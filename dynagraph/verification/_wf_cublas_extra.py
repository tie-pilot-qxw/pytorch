#!/usr/bin/env python3
"""Unpack the extra (CU_LAUNCH_PARAM_BUFFER_POINTER) param buffer of the nvjet node,
capture at two M values with exactly the same tensor pointers -> any difference can only be a shape-related field."""
import ctypes, gc, math, struct, sys
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

K = N = 512
MMAX = 512
DT = torch.bfloat16
torch.manual_seed(0)
x_full = torch.randn(MMAX, K, device="cuda", dtype=DT)
w = torch.randn(K, N, device="cuda", dtype=DT)
b = torch.randn(N, device="cuda", dtype=DT)
y_full = torch.zeros(MMAX, N, device="cuda", dtype=DT)
print("ptrs A=0x%x B=0x%x C=0x%x D=0x%x" % (x_full.data_ptr(), w.data_ptr(),
                                            b.data_ptr(), y_full.data_ptr()))

def run(M):
    return torch.addmm(b, x_full[:M], w, out=y_full[:M])

def read_extra(dp):
    ex = int(dp.extra)
    if ex == 0:
        return None, None
    ents = (ctypes.c_void_p * 8).from_address(ex)
    vals = [ents[i] or 0 for i in range(8)]
    ptr = size = None
    i = 0
    while i < 8:
        tag = vals[i]
        if tag == 0: break
        if tag == 1: ptr = vals[i + 1]; i += 2
        elif tag == 2:
            size = ctypes.c_size_t.from_address(vals[i + 1]).value; i += 2
        else: i += 1
    return ptr, size

def cap(M, keep=False):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): run(M)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): run(M)
    torch.cuda.synchronize()
    raw = g.raw_cuda_graph()
    nn = ck(cr.cudaGraphGetNodes(raw))[1]
    nd = ck(cr.cudaGraphGetNodes(raw, nn))[0][0]
    dp = ck(cd.cuGraphKernelNodeGetParams(nd))
    nm = ck(cd.cuFuncGetName(dp.func)).decode()
    ptr, size = read_extra(dp)
    _, psize = ck(cd.cuFuncGetParamInfo(dp.func, 0))
    blob = bytes((ctypes.c_ubyte * size).from_address(ptr)) if ptr else b""
    rec = dict(name=nm, func=int(dp.func), grid=(dp.gridDimX, dp.gridDimY, dp.gridDimZ),
               smem=int(dp.sharedMemBytes), extra_bufsize=size, paraminfo_size=int(psize),
               blob=blob, extra=int(dp.extra), bufptr=ptr, dp=dp)
    if keep: return rec, g, nd
    del g; gc.collect()
    return rec

for M in (104, 236):
    r = cap(M)
    print(f"M={M} {r['name']} grid={r['grid']} extra=0x{r['extra']:x} "
          f"buf=0x{r['bufptr']:x} bufsize={r['extra_bufsize']} "
          f"cuFuncGetParamInfo_size={r['paraminfo_size']}")

a = cap(104); bb = cap(236)
A, B = a["blob"], bb["blob"]
print(f"\ndiff {a['name']} M=104 vs 236, {len(A)}B  grid {a['grid']} -> {bb['grid']}")
segs = []
i = 0
while i < min(len(A), len(B)):
    if A[i] != B[i]:
        j = i
        while j < len(A) and A[j] != B[j]: j += 1
        segs.append((i, j - i)); i = j
    else: i += 1
print(f"  {len(segs)} segments / {sum(s[1] for s in segs)}B differ")
for o, l in segs:
    pa, pb = A[o:o+4].ljust(4, b"\0"), B[o:o+4].ljust(4, b"\0")
    print(f"    +{o:<5} {l:<3}B  {A[o:o+l].hex()} -> {B[o:o+l].hex()}"
          f"   i32@{o&~3}: {struct.unpack('<i',A[o&~3:(o&~3)+4])[0]} -> "
          f"{struct.unpack('<i',B[o&~3:(o&~3)+4])[0]}")

# try a patch on the same kernel: capture M=236, set the diffed fields to the M=104 values + swap the grid
rec, g, nd = cap(236, keep=True)
g.replay(); torch.cuda.synchronize()
gexec = g.raw_cuda_graph_exec()
buf = bytearray(rec["blob"])
for o, l in segs:
    buf[o:o+l] = A[o:o+l]
size = len(buf)
cbuf = (ctypes.c_ubyte * size).from_buffer(buf)
csize = ctypes.c_size_t(size)
extra_arr = (ctypes.c_void_p * 5)(
    ctypes.c_void_p(1), ctypes.cast(cbuf, ctypes.c_void_p),
    ctypes.c_void_p(2), ctypes.cast(ctypes.pointer(csize), ctypes.c_void_p),
    ctypes.c_void_p(0))
p = cd.CUDA_KERNEL_NODE_PARAMS()
p.func = rec["dp"].func
p.gridDimX, p.gridDimY, p.gridDimZ = a["grid"]
p.blockDimX = rec["dp"].blockDimX; p.blockDimY = rec["dp"].blockDimY
p.blockDimZ = rec["dp"].blockDimZ
p.sharedMemBytes = rec["dp"].sharedMemBytes
p.kernelParams = 0
p.extra = ctypes.addressof(extra_arr)
print("\npatch the M=236 graph -> M=104")
try:
    ck(cd.cuGraphExecKernelNodeSetParams(gexec, nd, p))
    y_full.zero_()
    g.replay(); torch.cuda.synchronize()
    ref = torch.addmm(b, x_full[:104], w).float()
    got = y_full[:104].float()
    err = (got - ref).abs().max().item()
    rel = err / max(ref.abs().max().item(), 1e-9)
    print(f"  max|err|={err:.4g} rel={rel:.3g} "
          f"{'OK' if rel < 5e-2 else '**WRONG**'}  residue 104..236 = "
          f"{y_full[104:236].abs().max().item():.4g}")
except RuntimeError as e:
    print("  SetParams failed:", str(e)[:150])
