#!/usr/bin/env python3
"""
Does cudagraph actually help? -- the experiment this research direction hinges on.

Background
----------
`workload_shapes/` has already shown that some workloads really do have many shapes (KITTI: 433 frames, 433 shapes),
and that pad-to-max waste is substantial (KITTI deep stages 94%). But all of that rests on an **untested premise**:

    cudagraph actually helps on this workload.

What cudagraph saves is only **the CPU overhead of each kernel launch**. If a step has few, large kernels
and the GPU is already the bottleneck, the launches are already hidden by async execution and cudagraph saves nothing --
then no matter how large the pad waste is, it cannot be recovered, because you would not use cudagraph at all.

**Anti-correlation hypothesis to test**: the cases where pad-to-max is expensive (proteins L^2/L^3, large kernels, GPU-bound)
and the cases where cudagraph helps (MD/MLIP small molecules, hundreds to thousands of small kernels, launch-bound)
may be **mutually exclusive**. If that holds, there is no room for this approach in between.

Method
------
Same model, same input, measure steady-state single-step time in three tiers:

    A. eager                      -- no compilation at all
    B. inductor without cudagraph -- mode="max-autotune-no-cudagraphs"
    C. inductor + cudagraph       -- mode="reduce-overhead"

**The B -> C speedup is cudagraph's contribution alone**; that is the number we want.
A -> B is Inductor's doing and unrelated to this topic, but it has to be measured to separate the two.

Also reports the kernel count per step (which sets the magnitude of launch overhead) and GPU utilization,
to judge whether the step is launch-bound or GPU-bound.

Usage
-----
    python cudagraph_benefit.py --model tv:resnet18
    python cudagraph_benefit.py --model extra:md --sizes 21,42,118,370
    python cudagraph_benefit.py --model tv:resnet18 --batches 1,2,8,32,128

**Before running, make sure the card is idle**:
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Measured (list_mode_options):
#   default         = {}                          -> triton.cudagraphs defaults to False
#   reduce-overhead = {"triton.cudagraphs": True}
# That switch is the **only** difference between the two, so default vs reduce-overhead isolates cudagraph cleanly.
# max-autotune-no-cudagraphs cannot be the baseline: it is
#   {"max_autotune": True, "coordinate_descent_tuning": True}, which changes kernel selection,
# so B and C would not run the same set of kernels and the B->C difference would include the effect of autotune.
MODES = [
    ("eager", None),
    ("inductor", "default"),
    ("cudagraph", "reduce-overhead"),
]


def count_kernels():
    """Count how many kernels this step launches.

    Counts device-side events with torch.profiler. More reliable than parsing the generated Triton source,
    because that misses cuBLAS / cuDNN / ATen built-in kernels.
    """
    from torch.profiler import profile, ProfilerActivity
    return profile, ProfilerActivity


def timeit(fn, args, kwargs, warmup, iters):
    """Steady-state single-step time; returns (median ms, interquartile range ms).

    cudagraph needs at least 3 calls to get past warmup: cudagraph_trees goes through CUDAWarmupNode the first time,
    actually records on the second, and only replays from the third on. The default warmup of 10 is enough.
    """
    import torch
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    q1, q3 = ts[len(ts) // 4], ts[3 * len(ts) // 4]
    return statistics.median(ts), q3 - q1


def profile_kernels(fn, args, kwargs):
    """Returns (device kernel count, total device time in ms)."""
    import torch
    from torch.profiler import profile, ProfilerActivity
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        fn(*args, **kwargs)
        torch.cuda.synchronize()
    # Use key_averages rather than events(): it does not depend on internal event field names, so it is stable across versions.
    n, dur = 0, 0.0
    for row in prof.key_averages():
        cuda_us = getattr(row, "device_time_total", None)
        if cuda_us is None:
            cuda_us = getattr(row, "cuda_time_total", 0.0)
        if cuda_us and cuda_us > 0 and row.key not in ("cudaLaunchKernel",
                                                       "cudaDeviceSynchronize",
                                                       "cudaStreamSynchronize"):
            n += row.count
            dur += cuda_us / 1e3
    return n, dur


def run_one(spec, mode_name, mode_arg, args_override, warmup, iters, train):
    import torch
    import models as _models
    torch._dynamo.reset()
    _models.TRAIN = train
    with _models.device_ctx():
        model, args, kwargs = _models.build(spec)
    model.train() if train else model.eval()
    if args_override is not None:
        args = args_override(args)

    fn = model if mode_arg is None else torch.compile(
        model, dynamic=True, mode=mode_arg)
    # Safety net: make sure the baseline really has cudagraph off (default mode does not set this key; it relies on the config default)
    if mode_arg == "default":
        import torch._inductor.config as _ic
        assert _ic.triton.cudagraphs is False, "baseline unexpectedly has cudagraph on"

    ctx = torch.enable_grad() if train else torch.no_grad()
    def step():
        with ctx:
            out = fn(*args, **kwargs)
        if train:
            loss = out.float().sum() if isinstance(out, torch.Tensor) else \
                sum(v.float().sum() for v in
                    (out.values() if isinstance(out, dict) else out)
                    if isinstance(v, torch.Tensor))
            loss.backward()
            model.zero_grad(set_to_none=True)

    try:
        med, iqr = timeit(step, (), {}, warmup, iters)
        nk, gpu_ms = profile_kernels(step, (), {})
    except Exception as e:
        return {"mode": mode_name, "error": f"{type(e).__name__}: {str(e)[:150]}"}
    return {"mode": mode_name, "ms": med, "iqr": iqr,
            "kernels": nk, "gpu_ms": gpu_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tv:resnet18")
    ap.add_argument("--batches", default="", help="comma-separated; sweep batch to find the launch-bound knee")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
    import torch
    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1

    batches = [int(x) for x in a.batches.split(",") if x.strip()] or [None]
    print(f"torch {torch.__version__}  model {a.model}  train={a.train}")
    print("A=eager  B=inductor(default, cudagraph off)  C=inductor+cudagraph")
    print("**B->C alone is cudagraph's contribution**; A->B is Inductor's and unrelated to this topic.\n")

    for bs in batches:
        def override(args, _bs=bs):
            if _bs is None:
                return args
            out = []
            for x in args:
                if isinstance(x, torch.Tensor) and x.dim() >= 2:
                    shp = list(x.shape); shp[0] = _bs
                    out.append(torch.randn(*shp, device=x.device, dtype=x.dtype))
                else:
                    out.append(x)
            return tuple(out)

        rows = [run_one(a.model, n, m, override, a.warmup, a.iters, a.train)
                for n, m in MODES]
        tag = f"batch={bs}" if bs else "default input"
        print(f"  {tag}")
        got = {}
        for r in rows:
            if "error" in r:
                print(f"    {r['mode']:<10} failed: {r['error']}")
                continue
            got[r["mode"]] = r
            launch_frac = (1 - r["gpu_ms"] / r["ms"]) * 100 if r["ms"] > 0 else 0
            print(f"    {r['mode']:<10} {r['ms']:>8.3f} ms (IQR {r['iqr']:.3f})  "
                  f"kernel {r['kernels']:>5}  GPU {r['gpu_ms']:>7.3f} ms  "
                  f"nonGPU frac {launch_frac:>5.1f}%")
        if "inductor" in got and "cudagraph" in got:
            sp = got["inductor"]["ms"] / got["cudagraph"]["ms"]
            verdict = ("cudagraph helps" if sp >= 1.15 else
                       "**cudagraph barely helps**" if sp < 1.05 else "marginal")
            print(f"    -> B->C cudagraph speedup {sp:.2f}x   {verdict}")
        if "eager" in got and "inductor" in got:
            print(f"    -> A->B Inductor speedup "
                  f"{got['eager']['ms'] / got['inductor']['ms']:.2f}x (unrelated to this topic)")
        print()

    print("""How to read
-----------
"nonGPU frac" = 1 - total GPU time / wall clock. A high fraction means the step is launch-bound
and cudagraph has room to help; a low one means the GPU is already saturated and cudagraph saves nothing.

If a workload has large pad-to-max waste but B->C is close to 1.00x,
that workload offers **no headroom** for this approach -- however large the waste, it cannot be recovered, because nobody would use cudagraph there.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
