#!/usr/bin/env python3
"""How much extra each replay costs with SWITCH: same region, plain child nodes vs after recapturing into SWITCH, timed interleaved.

Two independently compiled copies of the same model (the 4-layer GEMM from `probe_extern_child`, cuBLAS, 8 addmm call sites):
  A has only seen 512/256 -- 8 plain child nodes
  B has seen 512/256/64 -- the splitK at M=64 makes all 8 call sites get recaptured into SWITCH (2 bodies each),
    and after the recapture the pointers are rewritten along with every shape change
The two streams are timed interleaved (CUDA events, exclusive card):
  same shape repeated 512 -- the planner exits early; this measures the evaluation overhead of the SWITCH nodes themselves (cond_cost.cu says ~9 us each)
  alternating 512/256 -- the planner runs every time, and B additionally rewrites the pointers once more
"""
from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

import torch
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct
from torch._inductor import dynagraph as dg

import probe_extern_child as pec


def build(tag, shapes):
    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = True
    ic.triton.dynagraph_partition_extern = False
    ic.triton.dynagraph_extern_child = True
    ic.triton.dynagraph_topology = "switch"     # what this bench measures
    ic.max_autotune_gemm = False
    ic.triton.autotune_pointwise = False
    grabbed = {}
    orig = ct._maybe_build_dynagraph

    def spy(model, inputs, kwargs, *a, **kw):
        r = orig(model, inputs, kwargs, *a, **kw)
        if r is not False:
            grabbed["r"] = r
        return r

    ct._maybe_build_dynagraph = spy
    try:
        torch.manual_seed(0)
        m = type(f"M_{tag}", (pec.M,), {})(128, 4).cuda().eval()
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        xs = {L: torch.randn(L, 128, device="cuda") for L in (512, 256, 64)}
        with torch.no_grad():
            for _ in range(2):
                for L in shapes:
                    f(xs[L])
        torch.cuda.synchronize()
    finally:
        ct._maybe_build_dynagraph = orig
    r = grabbed.get("r")
    return f, xs, r


def timed(fns, stream, iters=200):
    """Interleaved: for each iteration, each fn once over the stream; per-call event times."""
    out = {k: [] for k in fns}
    with torch.no_grad():
        for _ in range(iters):
            for k, (f, xs) in fns.items():
                for L in stream:
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    f(xs[L])
                    e1.record()
                    torch.cuda.synchronize()
                    out[k].append(e0.elapsed_time(e1) * 1000)
    return {k: (statistics.median(v), min(v)) for k, v in out.items()}


def main() -> int:
    fA, xA, rA = build("A", [512, 256])
    fB, xB, rB = build("B", [512, 256, 64])
    if rA is None or rB is None:
        print("FAIL not served"); return 1
    xA, xB = getattr(rA, "ex", rA), getattr(rB, "ex", rB)   # per-graph state lives on the runner's `ex`
    print(f"  A: extern call sites {len(rA.extern_sites)}, recaptures {rA.recaptures}, SWITCH nodes {sum(1 for c in xA.site_cond if c)}")
    print(f"  B: extern call sites {len(rB.extern_sites)}, recaptures {rB.recaptures}, SWITCH nodes {sum(1 for c in xB.site_cond if c)}, pointers always dirty {not xB.ptr_dirty and int(xB.ctx[len(rB.symbols) + 1].item()) == 1}")
    fns = {"A plain child": (fA, xA), "B SWITCH": (fB, xB)}
    for name, stream in (("same shape repeated 512 (planner exits early)", [512]), ("alternating 512/256 (planner runs every time)", [512, 256])):
        # warm
        timed(fns, stream, iters=5)
        res = timed(fns, stream, iters=150)
        a, b = res["A plain child"], res["B SWITCH"]
        print(f"\n  {name}")
        print(f"    A plain child  median {a[0]:7.1f} us  min {a[1]:7.1f} us")
        print(f"    B SWITCH       median {b[0]:7.1f} us  min {b[1]:7.1f} us   diff {b[0] - a[0]:+.1f} us ({len(rB.extern_sites)} SWITCH nodes -> {(b[0] - a[0]) / max(len(rB.extern_sites), 1):.1f} us each)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
