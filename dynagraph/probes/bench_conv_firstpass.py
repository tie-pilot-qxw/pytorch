#!/usr/bin/env python3
"""conv region: one graph (child route + SWITCH) vs cuDNN recording one graph per batch -- cost of the first pass and of steady state.

The `probe_conv_child` model (Conv2d 8->16->8 + relu + mean, (N, 8, 32, 32)), 12 batches
[16,1,2,3,4,8,32,64,96,128,5,48]. Compiled once with dynagraph off and once with it on (cannot interleave: the first pass is one-time),
every call timed with CUDA events, exclusive card. Reports: total first-pass time, number of recordings, per-step median of the second pass (steady state).
In the second pass the dynagraph-on side has seen every shape, so the planner only runs when the shape changes.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

import torch
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct

import probe_conv_child as pcc

BATCHES = [16, 1, 2, 3, 4, 8, 32, 64, 96, 128, 5, 48]


def run(dynagraph: bool, tag: str):
    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    ic.triton.dynagraph_partition_extern = False
    ic.triton.dynagraph_extern_child = True
    ic.triton.autotune_pointwise = False
    n_rec = {"n": 0}
    orig_rec = ct.CUDAGraphTreeManager.record_function

    def spy_rec(self, *a, **kw):
        n_rec["n"] += 1
        return orig_rec(self, *a, **kw)

    ct.CUDAGraphTreeManager.record_function = spy_rec
    try:
        torch.manual_seed(0)
        m = type(f"M_{tag}", (pcc.M,), {})().cuda().eval()
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        xs = {n: pcc.make_input(n) for n in BATCHES}
        with torch.no_grad():
            f(xs[BATCHES[0]])                      # compilation stays outside the timing
            torch.cuda.synchronize()
            passes = []
            for _ in range(3):
                per = []
                t0 = time.perf_counter()
                for n in BATCHES:
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record(); f(xs[n]); e1.record(); torch.cuda.synchronize()
                    per.append(e0.elapsed_time(e1) * 1000)
                passes.append((time.perf_counter() - t0, per))
    finally:
        ct.CUDAGraphTreeManager.record_function = orig_rec
    return n_rec["n"], passes


def main() -> int:
    print(f"  batch stream {BATCHES}, 3 passes per config (pass 1 = first pass)")
    for dyn, tag in ((False, "OFF"), (True, "ON")):
        rec, passes = run(dyn, tag)
        name = "dynagraph on (one graph + child + SWITCH)" if dyn else "cudagraph_trees, one recording per batch"
        print(f"\n  {name}: {rec} recordings")
        for k, (wall, per) in enumerate(passes):
            print(f"    pass {k + 1}  wall {wall * 1000:8.1f} ms   per-step device time median {statistics.median(per):7.1f} us  max {max(per):9.1f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
