#!/usr/bin/env python3
"""Practicality check: after harvesting (func, blob, grid, smem), destroy every graph used for harvesting,
then put them into a host graph and see whether the func handles and param blobs are still valid; also replay many times in a row."""
import ctypes, gc, sys, os
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

DT = torch.bfloat16
K = N = 512; MMAX = 4096
torch.manual_seed(0)
x_full = torch.randn(MMAX, K, device="cuda", dtype=DT)
w = torch.randn(K, N, device="cuda", dtype=DT)
b = torch.randn(N, device="cuda", dtype=DT)
y_full = torch.zeros(MMAX, N, device="cuda", dtype=DT)
def run(M): return torch.addmm(b, x_full[:M], w, out=y_full[:M])

def cap(M):
    print(f"  [cap {M}] warmup", flush=True)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): run(M)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    print(f"  [cap {M}] capture", flush=True)
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): run(M)
    torch.cuda.synchronize()
    print(f"  [cap {M}] getnodes", flush=True)
    raw = g.raw_cuda_graph()
    nn = ck(cr.cudaGraphGetNodes(raw))[1]
    nd = ck(cr.cudaGraphGetNodes(raw, nn))[0][0]
    print(f"  [cap {M}] nn={nn} getparams", flush=True)
    dp = ck(cd.cuGraphKernelNodeGetParams(nd))
    print(f"   cap M={M} extra=0x{int(dp.extra):x} kp=0x{int(dp.kernelParams):x} func=0x{int(dp.func):x}", flush=True)
    assert int(dp.extra) != 0, "extra is empty"
    ents = (ctypes.c_void_p * 8).from_address(int(dp.extra))
    vals = [ents[i] or 0 for i in range(8)]
    ptr = size = None
    i = 0
    while i < 8 and vals[i]:
        if vals[i] == 1: ptr = vals[i+1]; i += 2
        elif vals[i] == 2: size = ctypes.c_size_t.from_address(vals[i+1]).value; i += 2
        else: i += 1
    print(f"  [cap {M}] entries={[hex(v) for v in vals]} ptr={ptr and hex(ptr)} size={size}", flush=True)
    assert ptr and size
    rec = dict(M=M, funci=int(dp.func), name=ck(cd.cuFuncGetName(dp.func)).decode(),
               grid=(dp.gridDimX, dp.gridDimY, dp.gridDimZ),
               block=(dp.blockDimX, dp.blockDimY, dp.blockDimZ),
               smem=int(dp.sharedMemBytes),
               blob=bytes((ctypes.c_ubyte * size).from_address(ptr)))
    return rec, g, nd

Ms = [16, 104, 236, 512, 947]
table = {}
hold = []
for M in Ms:
    rec, g, nd = cap(M); table[M] = rec; hold.append((g, nd))
torch.cuda.synchronize()
rec0, gb, ndb = cap(947)          # build the host graph first
gb.replay(); torch.cuda.synchronize()
print("destroying all harvest graphs ...", flush=True)
hold.clear(); gc.collect(); torch.cuda.synchronize()
print("destroyed; only the int func handles and byte strings remain", flush=True)
gexec = gb.raw_cuda_graph_exec()
KEEP = []
def apply(rec):
    buf = bytearray(rec["blob"]); sz = len(buf)
    cb = (ctypes.c_ubyte*sz).from_buffer(buf); cs = ctypes.c_size_t(sz)
    ea = (ctypes.c_void_p*5)(ctypes.c_void_p(1), ctypes.cast(cb, ctypes.c_void_p),
                             ctypes.c_void_p(2), ctypes.cast(ctypes.pointer(cs), ctypes.c_void_p),
                             ctypes.c_void_p(0))
    KEEP.append((buf, cb, cs, ea))
    p = cd.CUDA_KERNEL_NODE_PARAMS()
    p.func = cd.CUfunction(init_value=rec["funci"])   # rebuilt from the int alone, does not depend on the original object
    p.gridDimX, p.gridDimY, p.gridDimZ = rec["grid"]
    p.blockDimX, p.blockDimY, p.blockDimZ = rec["block"]
    p.sharedMemBytes = rec["smem"]; p.kernelParams = 0; p.extra = ctypes.addressof(ea)
    ck(cd.cuGraphExecKernelNodeSetParams(gexec, ndb, p))

print("\niter  M      kernel                                          result")
for it in range(3):
    for M in Ms:
        apply(table[M])
        y_full.zero_(); gb.replay(); torch.cuda.synchronize()
        ref = torch.addmm(b, x_full[:M], w).float()
        err = (y_full[:M].float()-ref).abs().max().item()
        rel = err/max(ref.abs().max().item(),1e-9)
        print(f"{it:<5} {M:<6} {table[M]['name'][:46]:<46} "
              f"{'OK' if rel<5e-2 else '**WRONG**'} rel={rel:.3g} past_M={y_full[M:].abs().max().item():.3g}")
