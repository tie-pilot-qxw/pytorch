#!/usr/bin/env python3
"""Do device-side node updates persist across launches: writing pointers only once and the planner early-out both rely on this.

Once slots are fixed, the planner writes buffer pointers only on the first replay (`docs/notes/SOLUTION.md`, the "fixed slots" section),
and the early-out skips the whole planner when the shape has not changed. Both assume that the values written on the device by
cudaGraphKernelNodeSetParam / SetGridDim **stay in the exec graph** and are still there at the next launch. The header does not say so,
and `probe_early_out` never really tested it either -- it captures at the largest shape 947, so if the params reverted to the captured values,
the kernel would just compute a few extra rows, and the rows being checked would still be correct.

Here it is the other way around: capture at the **small** shape 300, then replay 512. If the params reverted to 300, everything after row 300 would be
garbage. Two streams:
  same shape repeated 512,512,512   -- the planner exits early and writes nothing
  alternating 512,333,512,333       -- the planner runs every time, but only changes grid and scalars, never pointers
Criterion: every output of each shape is bitwise identical to its first output (the one where the planner wrote everything).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

import torch
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct

import bench


def main() -> int:
    ic.force_disable_caches = True
    ic.triton.dynagraph = True
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"

    grabbed = {}
    orig = ct._maybe_build_dynagraph

    def spy(model, inputs, kwargs, *a, **kw):
        r = orig(model, inputs, kwargs, *a, **kw)
        if r is not False:
            grabbed["r"] = r
        return r

    ct._maybe_build_dynagraph = spy
    torch.manual_seed(0)
    m = bench.build("launch", 128, "UP")
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")
    xs = {L: bench.make_input(L, 128) for L in (300, 512, 333)}
    bad = 0
    try:
        with torch.no_grad():
            f(xs[300])                      # the first call builds, capturing at 300
            r = grabbed.get("r")
            if r is None:
                print("FAIL not served"); return 1
            first: dict[int, torch.Tensor] = {}
            for name, stream in (("same shape repeated", [512, 512, 512]),
                                 ("alternating", [512, 333, 512, 333, 512])):
                print(f"\n  {name} {stream}")
                for L in stream:
                    o = f(xs[L]).float().clone()
                    # `applied` is (shape key, lane, extern-read addresses); the shape is the first part.
                    if tuple(v for _, v in getattr(r, 'ex', r).applied[0]) != (L,) and L not in first:
                        print(f"    L={L} did not go through the runner (applied={getattr(r, 'ex', r).applied})"); bad += 1
                    ref = first.setdefault(L, o)
                    d = (o - ref).abs().max().item()
                    tail = (o - ref)[300:].abs().max().item()
                    ok = d == 0.0
                    bad += not ok
                    print(f"    L={L:<4} vs first {d:.1e} (after row 300 {tail:.1e})"
                          f"  planner {'exited early' if not getattr(r, 'ex', r).flag_on else 'ran'} {'ok' if ok else 'FAIL'}")
    finally:
        ct._maybe_build_dynagraph = orig
    print("\n  " + ("all passed: device-side updates persist across launches" if not bad else f"{bad} item(s) failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
