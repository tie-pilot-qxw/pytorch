#!/usr/bin/env python3
r"""Messy-code scan: patterns common in AI paper repos, and how many of each DynaGraph catches.

The fourth axis of the goal: we cannot consider only well-written standard implementations. Each case here is one "typical mess" --
a `.item()` branch, slicing with a python int taken from a tensor, modifying the input in place, append in a loop then cat,
boolean mask indexing, cpu/numpy round trips, pad to max length, arange positions, a non-contiguous input across a graph
break, dropout in train mode, scalar arithmetic on shapes, einsum, tolist loops, an L x L
triangular mask (the buffer is quadratic in the symbol), a python-side step counter, stashing intermediates on self.

Each case runs once with dynagraph off and once on (child route, extern kept in the graph), on the same shape stream of 4 lengths
x 2 passes, and records: how many graphs dynamo cut, how many graph breaks, how many regions cudagraphify asked about,
how many were served, fallback tags, recording count (off vs on), and the max absolute diff between the off and on outputs (both sides pin
autotune, same set of kernels, so non-random cases should be bitwise identical).

No timing. This is about coverage: which patterns one graph can serve, which get cut into pieces that are each served, which are refused entirely
and why. Results are written up in docs/notes/MESSY.md.
"""
from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

import torch
import torch.nn.functional as F
import torch._dynamo
import torch._dynamo.config as dcfg
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct
from torch._dynamo.utils import counters

D = 64
MAXL = 512
SHAPES = [64, 200, 128, 333]


# ----------------------------------------------------------------- cases
class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(D, D)
        self.b = torch.nn.Linear(D, D)


class ItemBranch(Base):
    def forward(self, x):
        h = torch.relu(self.a(x))
        if h.abs().mean().item() > 0.3:      # .item() branch: graph break
            h = h * 2
        return self.b(h).sum(-1)


class IntFromTensor(Base):
    def forward(self, x):
        n = int((x[:, 0] > 0).sum())         # python int slice: break + data-dependent length
        return self.a(x[:n]).sum(-1)


class InplaceInput(Base):
    def forward(self, x):
        x.mul_(0.5)                          # modify the input in place
        x[:, 0] = 1.0
        return self.a(x).sum(-1)


class CatLoop(Base):
    def forward(self, x):
        outs = []
        for i in range(3):                   # append in a loop, then cat
            outs.append(torch.tanh(self.a(x) * (i + 1)))
        return torch.cat(outs, dim=0).sum(-1)


class BoolMask(Base):
    def forward(self, x):
        y = x[x[:, 0] > 0]                   # boolean mask: unbacked
        return self.a(y).sum(-1)


class CpuRoundtrip(Base):
    def forward(self, x):
        x2 = torch.tensor(x.cpu().numpy(), device=x.device)   # cpu/numpy round trip
        return self.a(x2).sum(-1)


class PadToMax(Base):
    def forward(self, x):
        x = F.pad(x, (0, 0, 0, MAXL - x.shape[0]))            # pad to max
        return self.a(x).sum(-1)


class ArangePos(Base):
    def forward(self, x):
        pos = torch.arange(x.shape[0], device=x.device).float()[:, None]
        return self.a(x + pos * 0.01).sum(-1)


class TransposeAcrossBreak(Base):
    def forward(self, x):
        h = self.a(x).transpose(0, 1)
        if h.sum().item() > 1e9:             # break; the second segment gets a non-contiguous input
            h = h * 0
        return self.b(h.transpose(0, 1)).sum(-1)


class DropoutTrain(Base):
    def __init__(self):
        super().__init__()
        self.d = torch.nn.Dropout(0.1)

    def forward(self, x):
        return self.b(self.d(torch.relu(self.a(x)))).sum(-1)


class SizeArith(Base):
    def forward(self, x):
        return (self.a(x) / x.shape[0]).sum(-1) * (x.shape[0] ** 0.5)


class Einsum(Base):
    def forward(self, x):
        return torch.einsum("ld,dk->lk", x, self.a.weight).sum(-1)


class TolistLoop(Base):
    def forward(self, x):
        s = 0.0
        for v in x[0, :2].tolist():          # tolist loop
            s += v
        return self.a(x).sum(-1) + s


class TrilMask(Base):
    def forward(self, x):
        L = x.shape[0]
        m = torch.tril(torch.ones(L, L, device=x.device))     # L x L: quadratic in the symbol
        att = (self.a(x) @ x.t()) * m
        return att.sum(-1)


class StepCounter(Base):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1                      # python-side counter: the guard fails every call
        return self.a(x).sum(-1) * (1.0 if self.calls > 0 else 0.0)


class StashOnSelf(Base):
    def forward(self, x):
        h = torch.relu(self.a(x))
        self.cache = h                       # stash an intermediate on self
        return self.b(h).sum(-1)


CASES = [
    ("item_branch", ItemBranch, False, {}),
    ("int_from_tensor", IntFromTensor, False, {}),
    ("int_from_tensor_unbacked", IntFromTensor, False, {"capture_scalar_outputs": True}),
    ("inplace_input", InplaceInput, False, {}),
    ("cat_loop", CatLoop, False, {}),
    ("bool_mask", BoolMask, False, {}),
    ("bool_mask_unbacked", BoolMask, False, {"capture_dynamic_output_shape_ops": True}),
    ("cpu_roundtrip", CpuRoundtrip, False, {}),
    ("pad_to_max", PadToMax, False, {}),
    ("arange_pos", ArangePos, False, {}),
    ("transpose_across_break", TransposeAcrossBreak, False, {}),
    ("dropout_train", DropoutTrain, True, {}),
    ("size_arith", SizeArith, False, {}),
    ("einsum", Einsum, False, {}),
    ("tolist_loop", TolistLoop, False, {}),
    ("tril_mask", TrilMask, False, {}),
    ("step_counter", StepCounter, False, {}),
    ("stash_on_self", StashOnSelf, False, {}),
]


# --------------------------------------------------------------- harness
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


def run_case(name, cls, train, dyn_cfg, dynagraph: bool):
    torch._dynamo.reset()
    counters.clear()
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    ic.triton.dynagraph_partition_extern = False
    ic.triton.dynagraph_extern_child = True
    ic.max_autotune_gemm = False
    ic.triton.autotune_pointwise = False     # same set of kernels on both sides

    saved = {k: getattr(dcfg, k) for k in dyn_cfg}
    for k, v in dyn_cfg.items():
        setattr(dcfg, k, v)

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
            served.append(getattr(model, "__name__", "?"))
        return r

    n_rec = {"n": 0}
    orig_rec = ct.CUDAGraphTreeManager.record_function

    def spy_rec(self, *a, **kw):
        n_rec["n"] += 1
        return orig_rec(self, *a, **kw)

    ct._maybe_build_dynagraph = spy_build
    ct.CUDAGraphTreeManager.record_function = spy_rec
    outs, err = {}, None
    try:
        torch.manual_seed(0)
        m = type(f"{cls.__name__}_{'ON' if dynagraph else 'OFF'}", (cls,), {})().cuda()
        m.train(train)
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for _ in range(2):
                for L in SHAPES:
                    o = f(make_input(L))
                    o = o if isinstance(o, (tuple, list)) else (o,)
                    outs[L] = [t.detach().float().cpu().clone() for t in o]
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        ct._maybe_build_dynagraph = orig_build
        ct.CUDAGraphTreeManager.record_function = orig_rec
        lg.removeHandler(grab)
        for k, v in saved.items():
            setattr(dcfg, k, v)

    tags = sorted({t.split("[", 1)[1].split("]", 1)[0] for t in grab.msgs if "fallback [" in t})
    return dict(
        graphs=counters["stats"].get("unique_graphs", 0),
        breaks=sum(counters["graph_break"].values()),
        asked=len(asked), served=len(served), tags=tags,
        rec=n_rec["n"], outs=outs, err=err,
    )


def main() -> int:
    rows = []
    for name, cls, train, dyn_cfg in CASES:
        off = run_case(name, cls, train, dyn_cfg, False)
        on = run_case(name, cls, train, dyn_cfg, True)
        if on["err"] or off["err"]:
            num = "ERR"
        elif train:
            num = "(random)"
        else:
            worst = 0.0
            for L in SHAPES:
                a, b = off["outs"].get(L), on["outs"].get(L)
                if a is None or b is None or len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
                    worst = float("inf"); break
                worst = max(worst, max((x - y).abs().max().item() for x, y in zip(a, b)))
            num = f"{worst:.1e}"
        rows.append((name, on["graphs"], on["breaks"], on["asked"], on["served"],
                     off["rec"], on["rec"], ",".join(on["tags"]) or "-", num,
                     on["err"] or off["err"] or ""))

    print(f"\n  shape stream {SHAPES} x 2 passes; gr = dynamo graphs, ask/srv = regions cudagraphify asked about/served, rOff/rOn = cudagraph_trees record_function calls (dynagraph off/on)")
    print(f"  {'case':<26}{'gr':>3}{'break':>6}{'ask':>3}{'srv':>3}{'rOff':>5}{'rOn':>5}  {'fallback tags':<34}{'on-off':>8}")
    for r in rows:
        print(f"  {r[0]:<26}{r[1]:>3}{r[2]:>6}{r[3]:>3}{r[4]:>3}{r[5]:>5}{r[6]:>5}  {r[7]:<34}{r[8]:>8}  {r[9]}")
    n_full = sum(1 for r in rows if r[3] and r[4] == r[3] and r[6] == 0)
    n_part = sum(1 for r in rows if r[4] and (r[4] < r[3] or r[6] > 0))
    n_none = sum(1 for r in rows if r[3] and not r[4])
    print(f"\n  {len(rows)} cases: fully served with 0 recordings {n_full}, partially served {n_part}, fully refused {n_none}, "
          f"never reached cudagraphify {sum(1 for r in rows if not r[3])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
