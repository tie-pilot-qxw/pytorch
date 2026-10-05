#!/usr/bin/env python3
"""Steady-state cost of the two ways to handle a topology change: host-side exec selection (default) vs device-side SWITCH, timed interleaved.

Model from `bench_switch_cost` (4-layer GEMM, cuBLAS, 8 addmm call sites; M=64 goes to splitK, which is the second topology):
  A host   has only seen 512/256 -- 1 graph, 8 plain children
  B host   has seen 512/256/64 -- 2 graphs (combos (0,)*8 and (1,)*8), the host picks by shape
  C switch has seen 512/256/64 -- 1 graph, 8 SWITCHes with 2 bodies each
Three shape streams (CUDA events, exclusive card): same shape repeated 512 (planner early-out), alternating 512/256 (same graph, planner runs every call),
alternating 512/64 (B switches between its two graphs, each early-outs; C has one graph, planner runs every call + writes the body index).
"""
from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

import torch
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct

import probe_extern_child as pec


def build(tag, shapes, mode):
    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = True
    ic.triton.dynagraph_partition_extern = False
    ic.triton.dynagraph_extern_child = True
    ic.triton.dynagraph_topology = mode
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
    return f, xs, grabbed.get("r")


def timed(fns, stream, iters):
    """Per config, per shape: (median, min) over the interleaved samples. Per
    shape, not over the mixed stream: a median over two shape clusters lands
    on whichever cluster's edge the count favours and compares nothing."""
    out = {k: {L: [] for L in stream} for k in fns}
    with torch.no_grad():
        for _ in range(iters):
            for k, (f, xs) in fns.items():
                for L in stream:
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    f(xs[L])
                    e1.record()
                    torch.cuda.synchronize()
                    out[k][L].append(e0.elapsed_time(e1) * 1000)
    return {k: {L: (statistics.median(v), min(v)) for L, v in d.items()} for k, d in out.items()}


def main() -> int:
    fA, xA, rA = build("A", [512, 256], "host")
    fB, xB, rB = build("B", [512, 256, 64], "host")
    fC, xC, rC = build("C", [512, 256, 64], "switch")
    for name, r in (("A host", rA), ("B host", rB), ("C switch", rC)):
        if r is None:
            print(f"FAIL {name} not served"); return 1
        print(f"  {name:<9} call sites {len(r.extern_sites)}, graphs {len(r.execs)} {sorted(r.execs)}, "
              f"recaptures {r.recaptures}, SWITCHes {sum(1 for c in r.ex.site_cond if c)}")
    fns = {"A host 1 graph": (fA, xA), "B host 2 graphs": (fB, xB), "C switch": (fC, xC)}
    for name, stream in (("same shape repeated 512 (planner early-out)", [512]),
                         ("alternating 512/256 (same graph, planner runs every call)", [512, 256]),
                         ("alternating 512/64 (B switches graphs, C switches body)", [512, 64])):
        timed(fns, stream, iters=5)
        res = timed(fns, stream, iters=150)
        print(f"\n  {name}")
        for L in stream:
            base = res["A host 1 graph"][L][0]
            for k in fns:
                med, mn = res[k][L]
                print(f"    L={L:<4} {k:<12} median {med:7.1f} us  min {mn:7.1f} us   vs A {med - base:+.1f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
