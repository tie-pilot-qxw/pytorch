"""What is the cheapest way to get the launch of one DeepGEMM bf16 call:
  torch recording   _capture_launch._record_graph (CUDAGraph + private pool)
  raw capture       side stream cudaStreamBeginCapture(relaxed) -> call -> EndCapture -> read nodes -> destroy
Measure each M once (every M is new, JIT already warm), and see how the launch changes with M:
which of function/grid/cluster/param bytes change.
"""
import time
import torch
import vllm.third_party.deep_gemm as deep_gemm
from cuda.bindings import runtime as cr
from torch.cuda._utils import _check_cuda_bindings as ck
from torch.utils import _capture_launch as cl

N, K = 256, 256
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
MS = list(range(13000, 17000, 97))
a_full = torch.randn(max(MS), K, device="cuda", dtype=torch.bfloat16)
d_full = torch.empty(max(MS), N, device="cuda", dtype=torch.bfloat16)


def call(M):
    deep_gemm.bf16_gemm_nt(a_full[:M], w, d_full[:M])


for M in MS:  # JIT every config outside timing
    call(M)
torch.cuda.synchronize()

s = torch.cuda.Stream()
mode = cr.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed


def raw_capture(M):
    st = s.cuda_stream
    ck(cr.cudaStreamBeginCapture(st, mode))
    try:
        call(M)
    finally:
        g = ck(cr.cudaStreamEndCapture(st))
    n = int(ck(cr.cudaGraphGetNodes(g))[1])
    nodes = ck(cr.cudaGraphGetNodes(g, n))[0]
    out = [cl.read_node(nd) for nd in nodes]
    ck(cr.cudaGraphDestroy(g))
    return out


def timed(f, Ms):
    ts, outs = [], []
    for M in Ms:
        t0 = time.perf_counter()
        outs.append(f(M))
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2], outs


with torch.cuda.stream(s):
    t_raw, raws = timed(raw_capture, MS)
t_torch, _ = timed(lambda M: cl._record_graph(call, (M,), {})[0], MS)
t_call, _ = timed(call, MS)
print(f"median per call: direct call (host) {t_call:.0f} us, raw capture {t_raw:.0f} us, torch recording {t_torch:.0f} us")
r0 = raws[0]
print(f"nodes per call {set(len(r) for r in raws)}; first: {r0[0]._fields if r0 and r0[0] else r0}")
variants = {}
for M, r in zip(MS, raws):
    k = tuple((x.func, x.grid, x.block, x.smem, x.cluster) for x in r)
    variants.setdefault(k, []).append(M)
print(f"(func, grid, block, smem, cluster) variants: {len(variants)}")
for k, ms in variants.items():
    print(f"   M {ms[0]}..{ms[-1]} ({len(ms)} values): grid {k[0][1]} block {k[0][2]} smem {k[0][3]} cluster {k[0][4]}")
p0, p1 = raws[0][0].params, raws[1][0].params
print(f"{len(p0)} params, sizes {[len(x) for x in p0]}")
for j, (x, y) in enumerate(zip(p0, p1)):
    if x != y:
        d = [i for i in range(len(x)) if x[i] != y[i]]
        print(f"   param {j} ({len(x)} bytes) M {MS[0]} vs {MS[1]} differing bytes {d}")
