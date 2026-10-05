#!/usr/bin/env python3
"""bf16/nvjet: (1) find groups of M that share a func; (2) capture with the same buffers and diff out the pure shape fields;
(3) try patching M + grid and check the numerics."""
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

def run(M):
    return torch.addmm(b, x_full[:M], w, out=y_full[:M])

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
    nodes = ck(cr.cudaGraphGetNodes(raw, nn))[0]
    assert nn == 1
    nd = nodes[0]
    dp = ck(cd.cuGraphKernelNodeGetParams(nd))
    nm = ck(cd.cuFuncGetName(dp.func)).decode()
    npar = 0
    while True:
        try:
            ck(cd.cuFuncGetParamInfo(dp.func, npar))
        except RuntimeError:
            break
        npar += 1
        if npar > 64: break
    kp, ex = int(dp.kernelParams), int(dp.extra)
    if npar == 0 or kp == 0:
        print(f"    M={M} {nm[:60]} npar={npar} kernelParams={kp} extra={ex} -> skipping param read", flush=True)
        if keep: return (nm, int(dp.func), (dp.gridDimX,dp.gridDimY,dp.gridDimZ), int(dp.sharedMemBytes), 0, b""), g, nd
        del g; gc.collect()
        return (nm, int(dp.func), (dp.gridDimX,dp.gridDimY,dp.gridDimZ), int(dp.sharedMemBytes), 0, b"")
    off, size = ck(cd.cuFuncGetParamInfo(dp.func, 0)); size = int(size)
    blob = bytes((ctypes.c_ubyte * size).from_address(
        (ctypes.c_void_p * 1).from_address(kp)[0]))
    info = (nm, int(dp.func), (dp.gridDimX, dp.gridDimY, dp.gridDimZ),
            int(dp.sharedMemBytes), int(off), blob)
    if keep:
        return info, g, nd
    del g; gc.collect()
    return info

# 1. find groups of M that share a func
byfunc = {}
for M in list(range(2, 273, 6)):
    print("  probe M=", M, flush=True)
    nm, f, gd, sm, off, blob = cap(M)
    byfunc.setdefault((nm, f, sm), []).append((M, gd))
print("bf16 addmm small-M range: func -> M list")
for (nm, f, sm), lst in byfunc.items():
    print(f"  {nm}  smem={sm}")
    print(f"     M={[m for m,_ in lst]}")
    print(f"     grid={sorted(set(g for _,g in lst))}")

# 2. take the largest group and diff
best = max(byfunc.items(), key=lambda kv: len(kv[1]))
(nm, f, sm), lst = best
Ma, Mb = lst[0][0], lst[-1][0]
ia = cap(Ma); ib = cap(Mb)
A, B = ia[5], ib[5]
print(f"\ndiff  {nm}  M={Ma} vs M={Mb}  struct={len(A)}B  "
      f"grid {ia[2]}->{ib[2]}  param_offset={ia[4]}")
segs = []
i = 0
while i < len(A):
    if A[i] != B[i]:
        j = i
        while j < len(A) and A[j] != B[j]: j += 1
        segs.append((i, j - i)); i = j
    else: i += 1
print(f"  {len(segs)} segments / {sum(s[1] for s in segs)}B differ in total")
for o, l in segs:
    a, bb = A[o:o+l], B[o:o+l]
    extra = ""
    if o % 4 == 0 and l <= 8:
        pad_a = A[o:o+4].ljust(4, b"\0"); pad_b = B[o:o+4].ljust(4, b"\0")
        extra = f"  i32@{o}: {struct.unpack('<i',pad_a)[0]} -> {struct.unpack('<i',pad_b)[0]}"
    print(f"    +{o:<5} {l:<3}B  {a.hex()} -> {bb.hex()}{extra}")

# 3. try a patch: turn the Mb graph into Ma
info, g, nd = cap(Mb, keep=True)
g.replay(); torch.cuda.synchronize()
gexec = g.raw_cuda_graph_exec()
dp0 = ck(cd.cuGraphKernelNodeGetParams(nd))
orig = bytearray(info[5])
size = len(orig)
cand = sorted({o for o, l in segs if o % 4 == 0})
print(f"\ntry patch: replace every diffed 4-byte-aligned field, M={Mb} value -> M={Ma} value, "
      f"and swap the grid too ({ib[2]} -> {ia[2]})")
buf = bytearray(B)
for o, l in segs:
    buf[o:o+l] = A[o:o+l]
cbuf = (ctypes.c_ubyte * size).from_buffer(buf)
arr = (ctypes.c_void_p * 1)(ctypes.cast(cbuf, ctypes.c_void_p))
p = cd.CUDA_KERNEL_NODE_PARAMS()
p.func = dp0.func
p.gridDimX, p.gridDimY, p.gridDimZ = ia[2]
p.blockDimX, p.blockDimY, p.blockDimZ = dp0.blockDimX, dp0.blockDimY, dp0.blockDimZ
p.sharedMemBytes = dp0.sharedMemBytes
p.kernelParams = ctypes.addressof(arr)
p.extra = 0
try:
    ck(cd.cuGraphExecKernelNodeSetParams(gexec, nd, p))
    y_full.zero_()
    g.replay(); torch.cuda.synchronize()
    ref = torch.addmm(b, x_full[:Ma], w).float()
    got = y_full[:Ma].float()
    err = (got - ref).abs().max().item()
    rel = err / max(ref.abs().max().item(), 1e-9)
    tail = y_full[Ma:Mb].abs().max().item()
    print(f"  result max|err|={err:.4g} rel={rel:.3g}  "
          f"{'OK' if rel<5e-2 else '**WRONG**'}   residue between Ma..Mb={tail:.3g}")
except RuntimeError as e:
    print("  SetParams failed:", str(e)[:120])
