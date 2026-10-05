"""Patched DeepGEMM ($DG_DEPS/deepgemm-src, default /workspace/_deps/deepgemm-src): describe_begin/end vs bare capture, per M."""
import time
import torch
import deep_gemm
from torch.utils import _capture_launch as cl

print("deep_gemm from", deep_gemm.__file__)
N, K = 256, 256
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
MS = list(range(13000, 17000, 97))
A = torch.randn(max(MS), K, device="cuda", dtype=torch.bfloat16)
D = torch.empty(max(MS), N, device="cuda", dtype=torch.bfloat16)
call = lambda M: deep_gemm.bf16_gemm_nt(A[:M], w, D[:M])
for M in MS:
    call(M)
torch.cuda.synchronize()
r = A[:MS[0]].float() @ w.float().t()
print("numerics", ((D[:MS[0]].float() - r).abs().max() / r.abs().max()).item())

def describe(M):
    deep_gemm._C.describe_begin()
    try:
        call(M)
    finally:
        return deep_gemm._C.describe_end()

bad = 0
for M in MS:
    d = describe(M)
    rec = cl._record_raw(lambda: call(M), (), {})
    if len(d) != len(rec):
        bad += 1; print("count", M, len(d), len(rec)); continue
    for x, y in zip(d, rec):
        func, grid, block, smem, cluster, pdl, args = x
        same = (func == y.func and tuple(grid) == y.grid and tuple(block) == y.block and smem == y.smem
                and (None if cluster == 1 else (cluster, 1, 1)) == y.cluster and list(args) == list(y.params))
        if not same:
            bad += 1
            print("diff at M", M, func == y.func, grid, y.grid, smem, y.smem, cluster, y.cluster,
                  [i for i, (p, q) in enumerate(zip(args, y.params)) if p != q], [len(a) for a in args], [len(p) for p in y.params])
print(f"describe == capture on {len(MS) - bad}/{len(MS)} M values; pdl {set(x[5] for M in MS[:3] for x in describe(M))}")
torch.cuda.synchronize()
def t(f):
    ts = []
    for M in MS:
        t0 = time.perf_counter(); f(M); ts.append((time.perf_counter() - t0) * 1e6)
    return sorted(ts)[len(ts) // 2]
print(f"per call median: direct {t(call):.0f} us, describe {t(describe):.0f} us, bare capture {t(lambda M: cl._record_raw(lambda: call(M), (), {})):.0f} us")
