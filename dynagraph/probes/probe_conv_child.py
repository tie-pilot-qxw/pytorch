#!/usr/bin/env python3
r"""conv stays in the graph: the path where an extern call allocates its own output (no out=).

Inductor emits conv as `buf0 = extern_kernels.convolution(...)` -- the output is allocated by cuDNN
itself, not through the wrapper's empty_strided_cuda, so it is not in the arena; the following relu then
writes it in place (`buf1 = buf0  # reuse`). The child path used to handle only `out=` calls (mm/addmm). Now:
at harvest time the output tensor of each shape is recorded (allocated during the small-graph capture and kept
alive with the small graph); during the main capture the one for the build shape is handed to the wrapper to
continue; the planner gets an extra "extern output pointer" patch, whose address the host writes into ctx per shape
(after the node state), rewritten on every shape change -- unlike arena pointers it is not fixed.

cuDNN's topology changes with batch (the verification agent measured 5 variants across 14 batches); shapes whose
topology changed are handed back upstream as `extern-topology`, and are only counted here, not treated as errors.

Criteria:
  1. The region is served (not rejected wholesale as extern-launch)
  2. Record count < control; every shape handed back upstream carries the extern-topology tag
  3. Numerics are bitwise identical to the control (same compile path, dynagraph off)
"""
from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

import torch
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct

BATCHES = [16, 24, 16, 20, 24, 12]


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(8, 16, 3, padding=1)
        self.d = torch.nn.Conv2d(16, 8, 3, padding=1)

    def forward(self, x):
        return self.d(torch.relu(self.c(x))).mean((1, 2, 3))


class Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs: list[str] = []

    def emit(self, rec):
        self.msgs.append(rec.getMessage())


def make_input(n):
    g = torch.Generator(device="cuda")
    g.manual_seed(n)
    return torch.randn(n, 8, 32, 32, device="cuda", generator=g)


def run(dynagraph: bool, tag: str):
    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    ic.triton.dynagraph_partition_extern = False
    ic.triton.dynagraph_extern_child = True
    ic.triton.autotune_pointwise = False

    grab = Grab()
    lg = logging.getLogger("torch._inductor.dynagraph")
    lg.setLevel(logging.INFO)
    lg.addHandler(grab)
    asked, served = [], []
    orig_build = ct._maybe_build_dynagraph

    def spy_build(model, inputs, kwargs, *a, **kw):
        r = orig_build(model, inputs, kwargs, *a, **kw)
        asked.append(getattr(model, "__name__", "?"))
        if r is not False:
            served.append(r)
        return r

    n_rec = {"n": 0}
    orig_rec = ct.CUDAGraphTreeManager.record_function

    def spy_rec(self, *a, **kw):
        n_rec["n"] += 1
        return orig_rec(self, *a, **kw)

    ct._maybe_build_dynagraph = spy_build
    ct.CUDAGraphTreeManager.record_function = spy_rec
    outs = {}
    try:
        torch.manual_seed(0)
        m = type(f"M_{tag}", (M,), {})().cuda().eval()
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        with torch.no_grad():
            for _ in range(2):
                for n in BATCHES:
                    outs[n] = f(make_input(n)).float().cpu().clone()
        torch.cuda.synchronize()
    finally:
        ct._maybe_build_dynagraph = orig_build
        ct.CUDAGraphTreeManager.record_function = orig_rec
        lg.removeHandler(grab)
    msgs = [t for t in grab.msgs if "fallback [" in t]
    tags = sorted({t.split("[", 1)[1].split("]", 1)[0] for t in msgs})
    return n_rec["n"], outs, dict(asked=asked, served=served, tags=tags, msgs=msgs)


def main() -> int:
    bad = 0
    rec_ctl, out_ctl, _ = run(False, "CTL")
    rec_on, out_on, obs = run(True, "ON")
    print(f"\n  batch stream {BATCHES} x 2 passes")
    print(f"  control recorded {rec_ctl}x; dynagraph on recorded {rec_on}x, asked {len(obs['asked'])} served {len(obs['served'])}, "
          f"tags {obs['tags'] or '-'}")
    for t in obs["msgs"][:6]:
        print(f"    {t[:120]}")
    if not obs["served"]:
        print("  FAIL not served"); bad += 1
    r = obs["served"][0] if obs["served"] else None
    if r is not None:
        print(f"  {len(r.extern_sites)} extern call sites {r.extern_sites}, self-allocated output buffers {dict(r.extern_outs_of)}, "
              f"harvested {r.harvests}x, skip {len(r.skip_keys)}")
    # A rebuild (input outgrown) makes a second runner; a batch skipped by any
    # runner for its cuDNN topology is recorded upstream once.
    skipped = set()
    for rr in obs["served"]:
        skipped |= {dict(k).get("s77") for k in rr.skip_keys}
    topos = sorted({t.split("]: ", 1)[1].split(" at ")[0] for t in obs["msgs"] if "extern-topology" in t})
    print(f"  cuDNN topology changes (node count build -> this shape): {topos or 'none'}; batches handed upstream {sorted(skipped) or 'none'}, "
          f"{len(obs['served'])} runners ({len(obs['served']) - 1} rebuilds)")
    if rec_on >= rec_ctl:
        print(f"  FAIL record count did not drop ({rec_ctl} -> {rec_on})"); bad += 1
    elif rec_on > len(skipped):
        print(f"  FAIL recorded {rec_on}x but only {len(skipped)} batches were handed upstream for topology"); bad += 1
    else:
        print(f"  ok records {rec_ctl} -> {rec_on} ({len(skipped)} batches handed upstream for a different cuDNN topology)")
    for n in sorted(set(BATCHES)):
        d = (out_ctl[n] - out_on[n]).abs().max().item()
        ok = d == 0.0
        bad += not ok
        print(f"    N={n:<3} max abs diff {d:.1e} {'ok' if ok else 'FAIL'}")
    print("\n  " + ("all passed" if not bad else f"{bad} items failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
