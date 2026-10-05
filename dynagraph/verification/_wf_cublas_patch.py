#!/usr/bin/env python3
"""Decisive experiment: on an already instantiated graph, use cuGraphExecKernelNodeSetParams
to change the M of the cuBLAS cutlass node and check whether the numerics are correct."""
import ctypes, struct, sys, math
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

K = N = 512
MMAX = 1024
NT = N // 64          # cutlass 64x64 tile -> number of tiles along N
torch.manual_seed(0)

dev = "cuda"
x_full = torch.randn(MMAX, K, device=dev)
w = torch.randn(K, N, device=dev)
b = torch.randn(N, device=dev)
y_full = torch.zeros(MMAX, N, device=dev)

M_CAP = 947

def run(M):
    return torch.addmm(b, x_full[:M], w, out=y_full[:M])

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(5):
        run(M_CAP)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(g):
    run(M_CAP)
torch.cuda.synchronize()
g.replay(); torch.cuda.synchronize()

raw = g.raw_cuda_graph()
gexec = g.raw_cuda_graph_exec()
n = ck(cr.cudaGraphGetNodes(raw))[1]
nodes = ck(cr.cudaGraphGetNodes(raw, n))[0]
print(f"capture M={M_CAP} (out= variant): {n} nodes")
kn = []
for nd in nodes:
    t = int(getattr(ck(cr.cudaGraphNodeGetType(nd)), "value", 0))
    if t == 0:
        dp = ck(cd.cuGraphKernelNodeGetParams(nd))
        nm = ck(cd.cuFuncGetName(dp.func)).decode()
        print(f"   kernel {nm}  grid=({dp.gridDimX},{dp.gridDimY},{dp.gridDimZ}) "
              f"smem={dp.sharedMemBytes} extra={int(dp.extra)}")
        kn.append((nd, dp, nm))
    else:
        print(f"   node type={t}")

assert len(kn) == 1, "expected exactly one kernel node"
node, dp0, name0 = kn[0]
off, size = ck(cd.cuFuncGetParamInfo(dp0.func, 0))
size = int(size)
orig = bytes((ctypes.c_ubyte * size).from_address(
    (ctypes.c_void_p * 1).from_address(int(dp0.kernelParams))[0]))
print(f"   single param offset={int(off)} size={size}  @0={struct.unpack_from('<i',orig,0)[0]} "
      f"@12={struct.unpack_from('<i',orig,12)[0]}")

# keep alive
_keep = []

def patch_M(newM):
    buf = bytearray(orig)
    struct.pack_into("<i", buf, 0, newM)                       # problem_size.m
    struct.pack_into("<i", buf, 12, math.ceil(newM / 64))      # grid_tiled_shape.m
    cbuf = (ctypes.c_ubyte * size).from_buffer(buf)
    arr = (ctypes.c_void_p * 1)(ctypes.cast(cbuf, ctypes.c_void_p))
    _keep.append((buf, cbuf, arr))
    p = cd.CUDA_KERNEL_NODE_PARAMS()
    p.func = dp0.func
    p.gridDimX = math.ceil(newM / 64) * NT
    p.gridDimY = dp0.gridDimY
    p.gridDimZ = dp0.gridDimZ
    p.blockDimX = dp0.blockDimX
    p.blockDimY = dp0.blockDimY
    p.blockDimZ = dp0.blockDimZ
    p.sharedMemBytes = dp0.sharedMemBytes
    p.kernelParams = ctypes.addressof(arr)
    p.extra = 0
    ck(cd.cuGraphExecKernelNodeSetParams(gexec, node, p))

print("\nM        patch result       max|err|     vs eager")
for newM in (947, 64, 16, 1, 128, 129, 512, 1000, 1024, 947):
    y_full.zero_()
    try:
        patch_M(newM)
    except RuntimeError as e:
        print(f"{newM:<8} SetParams failed: {str(e)[:60]}")
        continue
    g.replay(); torch.cuda.synchronize()
    ref = torch.addmm(b, x_full[:newM], w)
    got = y_full[:newM]
    err = (got - ref).abs().max().item()
    rel = err / max(ref.abs().max().item(), 1e-9)
    tail_dirty = y_full[newM:].abs().max().item() if newM < MMAX else 0.0
    ok = "OK" if rel < 2e-3 else "**WRONG**"
    print(f"{newM:<8} {ok:<18} {err:<12.4g} rel={rel:.3g}  written_past_M={tail_dirty:.3g}")
