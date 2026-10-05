#!/usr/bin/env python3
"""Independent re-check of claim 6: on the fp32 cutlass node, changing only param@0(M)/@12(mtiles)+gridDimX,
can a graph captured at M=BASE be replayed as any M?
Stricter than the original author's version: (1) dense M sweep; (2) fill with a sentinel instead of 0, catching both out-of-bounds writes
and "should have written but did not"; (3) supports different BASE values; (4) optionally patch only @0 and not @12 as a control."""
import ctypes, math, os, struct, sys
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

K = N = 512
MMAX = 1024
BASE = int(os.environ.get("BASE", "947"))
MODE = os.environ.get("MODE", "both")      # both | m_only | tiles_only | nogrid
SENT = 12345.0
torch.manual_seed(0)
x_full = torch.randn(MMAX, K, device="cuda")
w = torch.randn(K, N, device="cuda")
b = torch.randn(N, device="cuda")
y_full = torch.empty(MMAX, N, device="cuda")

def run(M): return torch.addmm(b, x_full[:M], w, out=y_full[:M])

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): run(BASE)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(g): run(BASE)
torch.cuda.synchronize()
g.replay(); torch.cuda.synchronize()
raw, gexec = g.raw_cuda_graph(), g.raw_cuda_graph_exec()
cnt = ck(cr.cudaGraphGetNodes(raw))[1]
nds = ck(cr.cudaGraphGetNodes(raw, cnt))[0]
assert cnt == 1
node = nds[0]
dp0 = ck(cd.cuGraphKernelNodeGetParams(node))
name0 = ck(cd.cuFuncGetName(dp0.func)).decode()
off, size = ck(cd.cuFuncGetParamInfo(dp0.func, 0)); size = int(size)
base_ptr = (ctypes.c_void_p * 1).from_address(int(dp0.kernelParams))[0]
orig = bytes((ctypes.c_ubyte * size).from_address(base_ptr))
NT = int(dp0.gridDimX) // math.ceil(BASE / 64)
print(f"BASE={BASE} kernel={name0[:70]} grid={(dp0.gridDimX,dp0.gridDimY,dp0.gridDimZ)} "
      f"smem={dp0.sharedMemBytes} size={size} @0={struct.unpack_from('<i',orig,0)[0]} "
      f"@12={struct.unpack_from('<i',orig,12)[0]} NT={NT} MODE={MODE}")
_keep = []
def patch(newM):
    buf = bytearray(orig)
    if MODE in ("both", "m_only", "nogrid"):
        struct.pack_into("<i", buf, 0, newM)
    if MODE in ("both", "tiles_only", "nogrid"):
        struct.pack_into("<i", buf, 12, math.ceil(newM / 64))
    cbuf = (ctypes.c_ubyte * size).from_buffer(buf)
    arr = (ctypes.c_void_p * 1)(ctypes.cast(cbuf, ctypes.c_void_p))
    _keep.append((buf, cbuf, arr)); _keep[:] = _keep[-4:]
    p = cd.CUDA_KERNEL_NODE_PARAMS()
    p.func = dp0.func; p.kern = dp0.kern; p.ctx = dp0.ctx
    p.gridDimX = int(dp0.gridDimX) if MODE == "nogrid" else math.ceil(newM / 64) * NT
    p.gridDimY, p.gridDimZ = dp0.gridDimY, dp0.gridDimZ
    p.blockDimX, p.blockDimY, p.blockDimZ = dp0.blockDimX, dp0.blockDimY, dp0.blockDimZ
    p.sharedMemBytes = dp0.sharedMemBytes
    p.kernelParams = ctypes.addressof(arr); p.extra = 0
    ck(cd.cuGraphExecKernelNodeSetParams(gexec, node, p))

Ms = [int(v) for v in os.environ.get("MS", "").split(",") if v] or list(range(1, MMAX + 1))
bad = []; exact = 0; unwritten = 0; oob = 0; tested = 0
worst = (0.0, None)
for M in Ms:
    y_full.fill_(SENT)
    try:
        patch(M)
    except RuntimeError as e:
        bad.append((M, "SetParams " + str(e)[:50])); continue
    g.replay(); torch.cuda.synchronize()
    ref = torch.addmm(b, x_full[:M], w)
    got = y_full[:M]
    err = (got - ref).abs().max().item()
    rel = err / max(ref.abs().max().item(), 1e-9)
    nw = int((got == SENT).sum().item())
    tail = 0 if M == MMAX else int((y_full[M:] != SENT).sum().item())
    tested += 1
    if err == 0.0: exact += 1
    if nw: unwritten += 1
    if tail: oob += 1
    if rel > worst[0]: worst = (rel, M)
    if rel > 2e-3 or nw or tail:
        bad.append((M, f"rel={rel:.3g} unwritten={nw} out_of_bounds={tail}"))
print(f"tested {tested} M values: bit-exact {exact}, with unwritten elements {unwritten}, with out-of-bounds writes {oob}, "
      f"max rel={worst[0]:.3g} @M={worst[1]}")
print("anomalies:", bad[:25], "..." if len(bad) > 25 else "")
