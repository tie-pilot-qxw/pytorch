#!/usr/bin/env python3
r"""Non-contiguous inputs: regions that receive a transposed view / strided slice across a graph break -- can they be served, and are they correct.

Previously `input-not-contiguous` rejected them outright. Now the input's own strides are kept when copying into the store (the kernel is specialized
for that geometry), and the extent is computed from the strides. Two cases: `h.transpose(0, 1)` across a break (the
transpose_across_break case in messy) and `x[:, ::2]` (a view with holes, extent > numel). With dynagraph off / on,
each runs 4 lengths x 2 passes; compares outputs (autotune pinned; bitwise or 1e-5), recording count, fallback tags.
"""
from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

import torch
import torch._dynamo
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct

D = 64
SHAPES = [64, 200, 128, 333]


class TransposeAcrossBreak(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(D, D)
        self.b = torch.nn.Linear(D, D)

    def forward(self, x):
        h = self.a(x).transpose(0, 1)
        if h.sum().item() > 1e9:
            h = h * 0
        return self.b(h.transpose(0, 1)).sum(-1)


class StridedSliceAcrossBreak(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(D, D)
        self.b = torch.nn.Linear(D // 2, D)

    def forward(self, x):
        h = self.a(x)[:, ::2]
        if h.sum().item() > 1e9:
            h = h * 0
        return self.b(h).sum(-1)


CASES = [("transpose", TransposeAcrossBreak), ("strided_slice", StridedSliceAcrossBreak)]


class Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs: list[str] = []

    def emit(self, rec):
        self.msgs.append(rec.getMessage())


def make_input(L: int) -> torch.Tensor:
    g = torch.Generator(device="cuda")
    g.manual_seed(L)
    return torch.randn(L, D, device="cuda", generator=g)


def run_case(cls, dynagraph):
    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    ic.triton.dynagraph_partition_extern = False
    ic.triton.dynagraph_extern_child = True
    ic.max_autotune_gemm = False
    ic.triton.autotune_pointwise = False
    grab = Grab()
    lg = logging.getLogger("torch._inductor.dynagraph")
    lg.setLevel(logging.INFO)
    lg.addHandler(grab)
    asked, served, regions = [], [], []
    orig_build = ct._maybe_build_dynagraph
    orig_impl = ct.cudagraphify_impl

    def spy_impl(*a, **kw):
        regions.append(1)
        return orig_impl(*a, **kw)

    def spy_build(model, inputs, kwargs, *a, **kw):
        r = orig_build(model, inputs, kwargs, *a, **kw)
        asked.append(1)
        if r is not False:
            served.append(1)
        return r

    n_rec = {"n": 0}
    orig_rec = ct.CUDAGraphTreeManager.record_function

    def spy_rec(self, *a, **kw):
        n_rec["n"] += 1
        return orig_rec(self, *a, **kw)

    ct._maybe_build_dynagraph = spy_build
    ct.CUDAGraphTreeManager.record_function = spy_rec
    ct.cudagraphify_impl = spy_impl
    outs, err = [], None
    try:
        torch.manual_seed(0)
        m = type(f"{cls.__name__}_{'ON' if dynagraph else 'OFF'}", (cls,), {})().cuda().eval()
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        with torch.no_grad():
            for _ in range(2):
                for L in SHAPES:
                    o = f(make_input(L))
                    torch.cuda.synchronize()
                    outs.append(o.detach().clone())
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:140]}"
    finally:
        ct._maybe_build_dynagraph = orig_build
        ct.CUDAGraphTreeManager.record_function = orig_rec
        ct.cudagraphify_impl = orig_impl
        lg.removeHandler(grab)
    tags = sorted({t.split("[", 1)[1].split("]", 1)[0] for t in grab.msgs if "fallback [" in t})
    return dict(asked=len(asked), served=len(served), rec=n_rec["n"], tags=tags, outs=outs, err=err,
                regions=len(regions))


def main() -> int:
    bad = 0
    print(f"\n  shape stream {SHAPES} x 2 passes; each row: asked/served, recordings off->on, fallback tags, output diff")
    for name, cls in CASES:
        off = run_case(cls, False)
        on = run_case(cls, True)
        if off["err"] or on["err"]:
            print(f"    {name:<14} ERR {off['err'] or on['err']}")
            bad += 1
            continue
        d = max((x - y).abs().max().item() for x, y in zip(off["outs"], on["outs"])) if len(off["outs"]) == len(on["outs"]) else float("inf")
        # A region without a symbolic input is never asked and records once
        # upstream; every asked region must be served with no recording.
        static_regions = on["regions"] - on["asked"]
        ok = on["served"] == on["asked"] >= 2 and on["rec"] <= static_regions and d <= 1e-5
        bad += not ok
        print(f"    {name:<14} regions {on['regions']} asked {on['asked']} served {on['served']}  recordings {off['rec']}->{on['rec']}  "
              f"{','.join(on['tags']) or '-':<24} output {d:.1e}  {'ok' if ok else 'FAIL'}")
    print("\n  " + ("all passed" if not bad else f"{bad} item(s) failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
