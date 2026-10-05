"""How the host cost of DeepGEMM describe varies with M: the first time an M is seen vs seeing it again, with ESM's real (N, K)."""
import random
import time

import deep_gemm
import torch

H, I, V = 480, 1920, 40
e = lambda *sh: torch.empty(*sh, device="cuda", dtype=torch.bfloat16)
MAXT = 8192


def gemms(T):
    out = []
    for n, k in ((H, H), (I, H), (H, I), (V, H)):
        out.append((e(T, k), e(n, k), e(T, n), "nk"))
        out.append((e(T, n), e(n, k).t(), e(T, k), "nk"))
        out.append((e(T, n).t(), e(T, k).t(), e(n, k), "n"))
    return out


def describe(g):
    a, b, d, dims = g
    t = time.perf_counter()
    deep_gemm._C.describe_begin()
    deep_gemm.bf16_gemm_nt(a, b, d, compiled_dims=dims)
    deep_gemm._C.describe_end()
    return time.perf_counter() - t


rng = random.Random(0)
Ms = rng.sample(range(500, MAXT), 60)
first, again = [], []
for T in Ms:
    gs = gemms(T)
    # really run once so any JIT happens outside the timing of "again"
    f = [describe(g) for g in gs]
    for a, b, d, dims in gs:
        deep_gemm.bf16_gemm_nt(a, b, d, compiled_dims=dims)
    torch.cuda.synchronize()
    ag = [describe(g) for g in gs]
    first += f
    again += ag
for name, xs in (("first", first), ("again", again)):
    xs = sorted(xs)
    print(f"{name}: median {xs[len(xs)//2]*1e6:.1f} us  p90 {xs[int(len(xs)*.9)]*1e6:.1f}  max {xs[-1]*1e6:.0f}  mean {sum(xs)/len(xs)*1e6:.1f}")
