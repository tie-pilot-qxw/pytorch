#!/usr/bin/env python3
"""Negative control: when the planner is right on the recording shape but wrong on other shapes, does the runtime verification catch it.

The self-check at the end of `build()` only cross-checks on the **recording shape**, and the XBLOCK bug caught in this round was found only
because "the formula happened to be wrong on the recording shape too" -- a formula that is wrong only on other shapes is not covered by the self-check.
So `__call__` also runs an eager cross-check on the first few new shapes
(`config.triton.dynagraph_verify_shapes`), and on a mismatch retires this region and goes back to re-recording.

This layer of protection itself needs a negative control, otherwise "no error reported" could mean there really is no problem,
or that the verification is not running at all.

Method: cut `cudaGraphKernelNodeSetGridDim` out of the planner, so the grid stays at the value
recorded at capture time. When the shape **gets smaller** the extra blocks are masked off and the result is still correct;
when the shape **gets larger** there are not enough blocks and nobody computes the tail. So shapes must go **from small to large**:
record at 64 first (the self-check passes), then go to 96 (must be caught).

96 rather than 512, because the static input buffers only keep headroom-times slack (2x by default),
and 512 would first hit the `input-too-large` retirement before reaching the grid check -- that is a different path,
tested separately as the third case.

Three cases:
  1. intact planner: 0 recordings, no fallback, numerics correct
  2. grid patch cut out: `runtime-mismatch` appears, recording count > 0 (re-recording resumes after retirement)
  3. shape exceeds the input slack: `input-too-large` appears, but no retirement -- rebuilt once at the larger shape
     (`dynagraph_rebuilds`), after which it is still served by one graph, recordings still 0

**In all three cases the results must be correct** -- retirement is a fallback, not a pass-through.
This is also why "the output changed" cannot be used as evidence that the sabotage took effect: once caught, the output should be correct anyway.

Does not measure time; can run on a shared card.
"""
from __future__ import annotations

import logging
import os
import re
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

SHAPES = (64, 96)        # must be ascending and within the input slack, see above
OVERSIZE = (64, 512)     # third case: exceeds the input slack

# Match by structure, not by literal indentation: if the template moves this line into an if block and the indentation changes, the test should not break --
# but if **the call itself disappears** it must still fire, so a failed match is still an assertion.
_SET_GRID = re.compile(
    r"cudaGraphKernelNodeSetGridDim\(handles\[i\],\s*"
    r"dim3\(\(unsigned\)gx, \(unsigned\)gy, \(unsigned\)gz\)\);"
)
# The host patcher's equivalent: the grid assignment on a node.
_SET_GRID_HOST = re.compile(
    r"nd->p\.gridDimX = \(unsigned\)gx; nd->p\.gridDimY = \(unsigned\)gy; nd->p\.gridDimZ = \(unsigned\)gz;"
)
HOST = os.environ.get("TORCHINDUCTOR_DYNAGRAPH_UPDATE") == "host"


def run(sabotage: bool, shapes):
    import torch
    import torch._inductor.config as ic
    import torch._inductor.cudagraph_trees as ct
    from torch._inductor import dynagraph as dg

    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = True
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"

    msgs = []

    class Grab(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    grab = Grab()
    lg = logging.getLogger("torch._inductor.dynagraph")
    lg.setLevel(logging.INFO)
    lg.addHandler(grab)

    n_record = {"v": 0}
    orig_record = ct.CUDAGraphTreeManager.record_function

    def spy(self, *a, **kw):
        n_record["v"] += 1
        return orig_record(self, *a, **kw)

    ct.CUDAGraphTreeManager.record_function = spy

    orig_planner = dg.generate_planner
    orig_host = dg.generate_host_patcher

    def broken_host(*a, **kw):
        src = orig_host(*a, **kw)
        if not _SET_GRID_HOST.search(src):
            raise AssertionError("grid assignment not found in the host patcher, the test is no longer valid")
        return _SET_GRID_HOST.sub("(void)0;  /* negative control: grid left unchanged */", src, count=1)

    def broken(*a, **kw):
        src = orig_planner(*a, **kw)
        # Assert rather than silently replace: as soon as the template changes this test should fire,
        # rather than turning into a "pass" that broke nothing.
        if not _SET_GRID.search(src):
            raise AssertionError("SetGridDim not found in the planner template, the test is no longer valid")
        return _SET_GRID.sub("(void)0;  /* negative control: grid left unchanged */", src, count=1)

    if sabotage:
        dg.generate_planner = broken
        dg.generate_host_patcher = broken_host

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(128, 128)

        def forward(self, x):
            h = torch.relu(self.l(x))
            # A per-row reduction: when the grid is too small the rows at the tail are never computed at all,
            # whereas an elementwise op cannot reveal that the grid is short.
            return h - h.mean(dim=-1, keepdim=True)

    try:
        torch.manual_seed(0)
        m = M().cuda().eval()
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        outs = {}
        torch.manual_seed(1)
        for M_ in shapes:
            x = torch.randn(M_, 128, device="cuda")
            with torch.no_grad():
                f(x)
                outs[M_] = (f(x).float().cpu().clone(), m(x).float().cpu().clone())
    finally:
        ct.CUDAGraphTreeManager.record_function = orig_record
        dg.generate_planner = orig_planner
        dg.generate_host_patcher = orig_host
        lg.removeHandler(grab)

    tags = [t.split("[", 1)[1].split("]", 1)[0]
            for t in msgs if t.startswith("DynaGraph fallback [")]
    return n_record["v"], outs, tags


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("no CUDA device available")
        return 1

    # A shape larger than the recorded one: under the dynamic layout (the
    # default) the arena and the input copies grow, nothing falls back and
    # nothing is recorded; under the fixed layout the input copies still
    # grow but an arena slot sized for the recorded shape overflows
    # (`arena-too-small`) and the region is rebuilt, still with nothing
    # recorded.
    from torch._inductor import config as _cfg

    oversize_tag = (
        "arena-too-small" if _cfg.triton.dynagraph_layout == "fixed" else None
    )
    cases = (
        ("intact planner", False, SHAPES, None, 0),
        ("grid patch cut out", True, SHAPES, "runtime-mismatch", 1),
        ("shape exceeds the recording shape", False, OVERSIZE, oversize_tag, 0),
    )

    bad = 0
    for name, sabotage, shapes, want_tag, want_record in cases:
        n, outs, tags = run(sabotage, shapes)
        print(f"\n  {name}  shapes={shapes}")
        print(f"    {n} recordings, fallback tags {tags or '(none)'}")

        if want_tag is None:
            ok = not tags
            print(f"    expect no fallback -- {'ok' if ok else 'FAIL'}")
        else:
            ok = want_tag in tags
            print(f"    expect {want_tag} to appear -- {'ok' if ok else 'FAIL not caught'}")
        bad += not ok

        # want_record means "at least this many recordings": after retirement the remaining shapes go back to re-recording.
        ok = (n == 0) if want_record == 0 else (n >= want_record)
        print(f"    expect recordings{' == 0' if not want_record else f' >= {want_record}'}"
              f" -- {'ok' if ok else 'FAIL'}")
        bad += not ok

        # Retirement is a fallback, not a pass-through: whichever path is taken, the result handed to the caller must be correct.
        # Comparing with eager is enough -- what we need to distinguish here is "computed correctly" from "tail not computed",
        # and the algorithmic difference between Triton and cuBLAS is below 1e-3, so the magnitudes are well separated.
        for M_ in shapes:
            got, eager = outs[M_]
            rel = ((got - eager).abs().max().item()
                   / max(eager.abs().max().item(), 1e-9))
            ok = rel < 1e-3
            print(f"    M={M_:<4} rel vs eager {rel:.2e} {'ok' if ok else 'FAIL result is wrong'}")
            bad += not ok

    print("\n  " + ("all passed" if not bad else f"{bad} item(s) failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
