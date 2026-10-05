#!/usr/bin/env python3
"""Are the 38 ms that torch.mm spends inside capture the cost of capturing, or the cost of cuBLAS seeing this shape for the first time?

No capture, plain eager calls: call the same M twice and time the first and the second call separately.
If the first is also ~38 ms and the second is fast, it is cuBLAS's heuristic query for a new shape,
unrelated to capture; eager/vLLM pays it too the first time it sees the shape.
"""
import statistics, time
import torch

K = N = 1024
w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
xmax = torch.randn(4096, K, device="cuda", dtype=torch.bfloat16)
torch.mm(xmax[:64], w)
torch.cuda.synchronize()
first, second = [], []
for M in list(range(1, 4096, 101))[:40]:
    x = xmax[:M]
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.mm(x, w, out=out); t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter(); torch.mm(x, w, out=out); t3 = time.perf_counter()
    torch.cuda.synchronize()
    first.append((t1 - t0) * 1e6); second.append((t3 - t2) * 1e6)
for nm, v in (("1st call of M", first), ("2nd call of M", second)):
    v = sorted(v)
    print(f"  eager {nm:<14} host time median {statistics.median(v):>8.0f} us  min {v[0]:>6.0f}  max {v[-1]:>7.0f}")
