#!/usr/bin/env python3
"""At tier 2, is each new shape a pure-CPU operation?

  1. Does any kernel actually run during capture: fill the output with a sentinel and
     check, after capture and before replay, whether the sentinel is still there
  2. Is warmup needed only once per stream: first capture on a brand-new stream with
     no warmup and see whether it fails; then warm up once, and capture all 40 new M
     after that with no warmup
  3. Host time of each capture
"""
import statistics, time
import torch

K = N = 1024
SENT = 12345.0
torch.manual_seed(0)
w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
# Use one shared pool, like DynaGraph's harvest does: with a private pool per graph, every
# capture has to open a new segment for cuBLAS's workspace, so we'd measure allocation, not capture.
POOL = torch.cuda.graph_pool_handle()
HOLD = []
INST = []
PARTS = []
xmax = torch.randn(4096, K, device="cuda", dtype=torch.bfloat16)
Ms = list(range(1, 4096, 101))[:40]


def capture(M, s, pool=None, keep=True):
    x = xmax[:M]
    out = torch.full((M, N), SENT, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=keep)
    t0 = time.perf_counter()
    with torch.cuda.stream(s):
        a0 = time.perf_counter()
        g.capture_begin(pool=POOL if pool is None else pool)
        a1 = time.perf_counter()
        try:
            torch.mm(x, w, out=out)
        finally:
            a2 = time.perf_counter()
            g.capture_end()
            a3 = time.perf_counter()
    PARTS.append(((a1 - a0) * 1e6, (a2 - a1) * 1e6, (a3 - a2) * 1e6))
    dt = (time.perf_counter() - t0) * 1e6
    t1 = time.perf_counter()
    if keep:
        g.instantiate()
    inst = (time.perf_counter() - t1) * 1e6
    INST.append(inst)
    torch.cuda.synchronize()
    untouched = bool((out == SENT).all().item())
    g.replay()
    torch.cuda.synchronize()
    err = (out.float() - (x.float() @ w.float())).abs().max().item()
    HOLD.append((g, out))
    return dt, untouched, err


print("\nII. Another brand-new stream: warm up once, then 40 new M with no warmup")
s1 = torch.cuda.Stream()
with torch.cuda.stream(s1):
    torch.mm(xmax[:64], w)
torch.cuda.synchronize()
times, bad, ran = [], 0, 0
for M in Ms:
    try:
        dt, untouched, err = capture(M, s1)
        times.append(dt)
        ran += not untouched
        bad += err > 1.0
    except Exception as e:
        print(f"    M={M} failed: {str(e)[:100]}")
        bad += 1
print(f"    {len(times)}/{len(Ms)} captures succeeded, {ran} had a kernel run during capture, "
      f"{bad} gave wrong numbers on replay")
if times:
    v = sorted(times)
    print(f"    capture only (keep_graph=True, what DynaGraph does): median {statistics.median(v):.0f} us, "
          f"min {v[0]:.0f}, max {v[-1]:.0f}")
    for nm, k in (("capture_begin", 0), ("torch.mm in capture", 1), ("capture_end", 2)):
        vv = sorted(p[k] for p in PARTS[-len(v):])
        print(f"      {nm:<18} median {statistics.median(vv):>8.0f} us  min {vv[0]:>7.0f}  max {vv[-1]:>7.0f}")
    vi = sorted(INST[-len(v):])
    print(f"    instantiate alone: median {statistics.median(vi):.0f} us, "
          f"min {vi[0]:.0f}, max {vi[-1]:.0f}")

print("\nI. Brand-new stream, no warmup at all, capture directly (run last, with its own pool: it fails, and the failure corrupts the pool)")
s0 = torch.cuda.Stream()
try:
    dt, untouched, err = capture(777, s0, pool=torch.cuda.graph_pool_handle())
    print(f"    succeeded  capture {dt:.0f} us  output still sentinel after capture={untouched}  replay error {err:.3g}")
except Exception as e:
    print(f"    failed: {str(e)[:120]}")

