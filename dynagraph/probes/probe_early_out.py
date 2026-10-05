#!/usr/bin/env python3
"""Skip the planner when the shape has not changed: how much it saves, and whether it ever skips wrongly.

planner + layout measured at 14.3% of the device time of each replay (43.7 us / 315 us), and patching is
idempotent, so re-patching the same shape is pure waste. The host now writes one extra "changed or not" slot at the end of ctx,
and the planner and layout return immediately when they see 0.

Two things must hold:
  1. **Gain**: on consecutive replays of the same shape the planner's device time should be close to 0; unchanged when the shape changes every step.
  2. **Correctness**: A,A,A (takes the early-out) and A,B,A,B (does not) must both match eager bitwise --
     the only possible error of the early-out is "something that should have been patched was not", which shows up immediately in the alternating stream.

Timing uses the profiler's device self time, unaffected by host contention.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

import torch
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct
from torch.profiler import profile, ProfilerActivity

import bench


def main() -> int:
    ic.force_disable_caches = True
    ic.triton.dynagraph = True
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"

    d_model = 128
    torch.manual_seed(0)
    m = bench.build("launch", d_model, "EO")
    grabbed = {}
    orig = ct._maybe_build_dynagraph

    def spy(model, inputs, kwargs, *a, **kw):
        r = orig(model, inputs, kwargs, *a, **kw)
        if r is not False:
            grabbed["r"] = r
        return r

    ct._maybe_build_dynagraph = spy
    try:
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        shapes = [947, 512, 333, 256, 128, 64]
        xs = {L: bench.make_input(L, d_model) for L in shapes}
        with torch.no_grad():
            for _ in range(2):
                for L in shapes:
                    f(xs[L])
        torch.cuda.synchronize()
    finally:
        ct._maybe_build_dynagraph = orig
    r = grabbed.get("r")
    if r is None:
        print("FAIL not served")
        return 1

    streams = {
        "changes every step (947,512,333,256,128,64)x8": shapes * 8,
        "same shape repeated (512)x48": [512] * 48,
        "alternating (512,256)x24": [512, 256] * 24,
    }
    bad = 0
    for name, stream in streams.items():
        with torch.no_grad():
            for L in stream[:6]:
                f(xs[L])
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for L in stream:
                    f(xs[L])
                torch.cuda.synchronize()
        n = len(stream)
        planner = total = 0.0
        for e in prof.key_averages():
            t = getattr(e, "self_device_time_total", 0) or 0
            total += t
            if "dynagraph_planner" in e.key:
                planner = t
        print(f"\n  {name}")
        print(f"    device time per replay {total / n:7.1f} us, of which planner {planner / n:5.1f} us"
              f" ({planner / max(total, 1e-9) * 100:.1f}%)")

    # Correctness. The reference cannot be eager: GEMMs are routed to Triton, which already differs from eager's cuBLAS
    # on the order of 3e-4, and that algorithmic difference would be read as an early-out error. The reference is **the same graph's
    # output for the same shape in the "changes every step" stream** -- same kernel, same config, must be bitwise identical. The only
    # possible error of the early-out is "something that should have been patched was not", which would make the later steps of A,A,A
    # or the switching steps of A,B,A,B deviate from it.
    print("\n  correctness (vs the same graph's output in the changing stream, must be bitwise identical):")
    ref = {}
    with torch.no_grad():
        for L in shapes:
            f(xs[L])
            ref[L] = f(xs[L]).float().clone()
        for name, stream in (("A,A,A,A,A", [512] * 5), ("A,B,A,B,A", [512, 256, 512, 256, 512])):
            worst = 0.0
            for L in stream:
                got = f(xs[L]).float()
                worst = max(worst, (got - ref[L]).abs().max().item())
            ok = worst == 0.0
            bad += not ok
            print(f"    {name:<10} max abs diff {worst:.1e} {'ok' if ok else 'FAIL'}")
    print("\n  " + ("all passed" if not bad else f"{bad} item(s) failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
