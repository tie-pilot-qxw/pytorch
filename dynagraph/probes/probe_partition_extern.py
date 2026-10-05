#!/usr/bin/env python3
r"""Cut-around fallback: partition at extern kernels so DynaGraph can serve the rest.

Today a single `extern_kernels.addmm` makes the whole region fall back with `extern-launch`, and Inductor
sends GEMMs to cuBLAS **by default** (`max_autotune_gemm` is off by default), so almost every real model
hits this right away.

cuBLAS nodes cannot be patched -- the kernels are closed source, and both the parameter layout and "which variant
a new shape would pick" are unknown. So this path does not patch them; it cuts on both sides instead: the extern call
runs eagerly, the segments before and after each become their own graph, and each is served by DynaGraph.
It relies on multi-partition support, which was just finished.

Switch: `TORCHINDUCTOR_DYNAGRAPH_PARTITION_EXTERN=1` (or the config.triton option of the same name).

**This probe does not assume cutting always pays off.** After cutting a GEMM-heavy model, Triton may be left with
only a few epilogues, the graph gets badly fragmented, and each segment carries its own cudagraph overhead. So it
reports facts, not a verdict: how many segments, how many served, which way the record count goes, whether numerics match.

Criteria:
  1. With the switch off it really hits `extern-launch` (the target is there)
  2. With the switch on, extern calls really are cut out (partition count goes up, and served segments
     contain no extern_kernels./torch.ops. call at all)
  3. Record count strictly drops (real coverage, not a cut for nothing)
  4. Numerics match the control
"""
from __future__ import annotations

import logging
import os
import re
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

import torch
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct
from torch._inductor import dynagraph as dg

SHAPES = [512, 256, 333, 64]
_PART_RE = re.compile(r"^def (partition_\d+)\(args\):", re.MULTILINE)
_EXTERN_RE = re.compile(r"\bextern_kernels\s*\.|\btorch\s*\.\s*ops\s*\.")


class Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = torch.nn.Linear(d, d)
        self.b = torch.nn.Linear(d, d)

    def forward(self, x):
        h = torch.relu(self.a(x))
        h = h - h.mean(dim=-1, keepdim=True)
        return x + self.b(h) * torch.sigmoid(h)


class M(torch.nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(d) for _ in range(n)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x.sum(dim=-1)


class Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs: list[str] = []

    def emit(self, rec):
        self.msgs.append(rec.getMessage())


def run(dynagraph: bool, cut: bool, tag: str):
    """Return (record count, per-shape outputs, observations)."""
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    ic.triton.dynagraph_partition_extern = cut
    # The numeric reference is **a separate compile**. Inductor autotunes reduction/pointwise configs on first run,
    # and on a shared card the winner changes per compile: when the mean kernel picks XBLOCK 8 instead of 1 the
    # summation order differs and the final value differs by 3e-5 (2 of 5 rounds). Pin the default config so both sides run the same kernels.
    ic.triton.autotune_pointwise = False
    # Key: do **not** enable max_autotune_gemm; that is Inductor's default, GEMMs go straight to cuBLAS.
    ic.max_autotune_gemm = False

    grab = Grab()
    lg = logging.getLogger("torch._inductor.dynagraph")
    lg.setLevel(logging.INFO)
    lg.addHandler(grab)

    seen: list[dict] = []
    orig_build = ct._maybe_build_dynagraph

    def spy_build(model, inputs, kwargs, *a, **kw):
        src = dg._wrapper_source(model) or ""
        r = orig_build(model, inputs, kwargs, *a, **kw)
        name = getattr(model, "__name__", None)
        body = dg._entry_source(src, name) if src else None
        seen.append(
            dict(
                name=name,
                n_part=len(_PART_RE.findall(src)),
                served=r is not False,
                # A served segment must never still contain an extern call
                extern_in_body=bool(body and _EXTERN_RE.search(body)),
                src=src,
            )
        )
        return r

    n_rec = {"n": 0}
    orig_rec = ct.CUDAGraphTreeManager.record_function

    def spy_rec(self, *a, **kw):
        n_rec["n"] += 1
        return orig_rec(self, *a, **kw)

    ct._maybe_build_dynagraph = spy_build
    ct.CUDAGraphTreeManager.record_function = spy_rec
    try:
        torch.manual_seed(0)
        cls = type(f"M_{tag}", (M,), {})
        m = cls(128, 4).cuda().eval()
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        outs = {}
        with torch.no_grad():
            # Run each shape twice: cudagraph_trees only does an eager warmup the first time and records the second.
            # With one pass the control's record count is 0 and the criterion is useless (hit this once).
            # Largest shape first: DynaGraph sizes the input buffer headroom by the first shape.
            for _ in range(2):
                for L in SHAPES:
                    x = torch.zeros(L, 128, device="cuda")
                    g = torch.Generator(device="cuda")
                    g.manual_seed(L)
                    x.copy_(torch.randn(L, 128, device="cuda", generator=g))
                    outs[L] = f(x).float().cpu().clone()
    finally:
        ct._maybe_build_dynagraph = orig_build
        ct.CUDAGraphTreeManager.record_function = orig_rec
        lg.removeHandler(grab)

    tags = sorted(
        {t.split("[", 1)[1].split("]", 1)[0] for t in grab.msgs if "fallback [" in t}
    )
    # Why upstream itself does not record; criterion 3 is only usable given this
    from torch._dynamo.utils import counters

    skips = {
        k: v
        for k, v in counters["inductor"].items()
        if "cudagraph" in k or "skip" in k or "partition" in k
    }
    return n_rec["n"], outs, dict(seen=seen, tags=tags, msgs=grab.msgs, skips=skips)


def main() -> int:
    bad = 0

    from torch._dynamo.utils import counters

    counters.clear()
    rec_ctl, out_ctl, obs_ctl = run(False, False, "CTL")
    rec_off, out_off, obs_off = run(True, False, "OFF")
    rec_on, out_on, obs_on = run(True, True, "ON")

    print(f"\n  control (dynagraph off)          recorded {rec_ctl}x, "
          f"partition {max([s['n_part'] for s in obs_ctl['seen']] or [0])}")
    print(f"  dynagraph on, no cut             recorded {rec_off}x, tags {obs_off['tags']}")
    print(f"  dynagraph on, cut at extern      recorded {rec_on}x, tags {obs_on['tags']}")
    print(f"\n  upstream cudagraph counters (control): {obs_ctl['skips'] or '(empty)'}")
    print(f"  after cutting: {obs_on['skips'] or '(empty)'}")

    # --- criterion 1: without cutting it really hits extern-launch ---------------------------
    print("\n  criterion 1: is the target there")
    if "extern-launch" not in obs_off["tags"]:
        print(f"    FAIL no extern-launch without cutting (tags {obs_off['tags']}) --"
              " this model never reached cuBLAS, the probe missed its target")
        bad += 1
    else:
        print("    ok without cutting it falls back with extern-launch")

    # --- criterion 2: extern really was cut out -------------------------------------
    print("\n  criterion 2: after cutting, do served segments still contain extern")
    n_part_on = max([s["n_part"] for s in obs_on["seen"]] or [0])
    n_part_off = max([s["n_part"] for s in obs_off["seen"]] or [0])
    print(f"    partition count {n_part_off} -> {n_part_on}")
    if n_part_on <= n_part_off:
        print("    FAIL partition count did not rise with the switch on -- the should_partition rule did not take effect")
        bad += 1
    served = [s for s in obs_on["seen"] if s["served"]]
    dirty = [s for s in served if s["extern_in_body"]]
    print(f"    cudagraphify asked {len(obs_on['seen'])}x, {len(served)} segments served")
    for s in obs_on["seen"]:
        mark = "served" if s["served"] else "fallback"
        print(f"      {s['name']}: {mark}"
              + ("  WARN segment still contains extern" if s["extern_in_body"] else ""))
    if dirty:
        print(f"    FAIL {len(dirty)} segments were served but still contain extern calls -- "
              "those would compute wrong results")
        bad += 1
    elif served:
        print("    ok served segments contain no extern calls")
    if not served:
        print("    FAIL no segment served after cutting -- the cut achieved nothing")
        bad += 1

    # --- criterion 3: the record count must really drop ----------------------------------------
    print("\n  criterion 3: record count")
    print(f"    control {rec_ctl} -> cut {rec_on}")
    if rec_ctl == 0:
        print("    FAIL the control never recorded, criterion invalid")
        bad += 1
    elif rec_on >= rec_ctl:
        print("    FAIL record count did not drop after cutting -- no coverage gained, and the graph is fragmented too")
        bad += 1
    else:
        print(f"    ok {rec_ctl - rec_on} fewer recordings")

    # --- criterion 4: numerics -------------------------------------------------------
    print("\n  criterion 4: numerics (reference is the same compile path with dynagraph off)")
    for L in SHAPES:
        a, b = out_ctl[L], out_on[L]
        if a.shape != b.shape:
            print(f"    L={L} shape {tuple(b.shape)} != {tuple(a.shape)}  FAIL")
            bad += 1
            continue
        rel = (a - b).abs().max().item() / max(a.abs().max().item(), 1e-9)
        # The reference is a separate compile+capture; cuBLAS picks its algorithm per capture (even with autotune pinned),
        # so a 1e-5-level difference comes and goes; the correctness gate is the runner's own bitwise check against eager.
        ok = rel < 1e-4
        print(f"    L={L:<4} rel vs control {rel:.2e} {'ok' if ok else 'FAIL'}")
        bad += not ok

    print("\n  " + ("all passed" if not bad else f"{bad} items failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
