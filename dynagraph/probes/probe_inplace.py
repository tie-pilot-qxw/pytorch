#!/usr/bin/env python3
r"""Regions that write inputs / buffers in place: once DynaGraph serves them, is the state changed exactly once?

The runner does several extra eager runs on build and on new shapes (warmup x3, self-check, verification, harvest),
and each one really executes the wrapper. If the region mutates an input in place (`x.mul_()`) or a static buffer
(BatchNorm's running_mean/var, a buffer used as a Python-side counter), those extra runs push the state forward
several more steps -- and verification comparing against replay does not match either (the runtime-mismatch of
inplace_input in the messy sweep). Now the extra runs either run on a copy or put the written tensors back afterwards.

Three cases: eval mode mutating an input in place; train-mode BatchNorm (running stats are static inputs, written
in place); a buffer counter `self.n.add_(1)`. Each runs the same stream of 4 lengths x 2 passes with dynagraph off / on
and compares: every output, the inputs / buffers after the run (all must be bitwise identical), record counts, fallback tags.
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


class InplaceInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(D, D)

    def forward(self, x):
        x.mul_(0.5)
        x[:, 0] = 1.0
        return self.a(x).sum(-1)


class BNTrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(D, D)
        self.bn = torch.nn.BatchNorm1d(D)
        self.b = torch.nn.Linear(D, D)

    def forward(self, x):
        return self.b(torch.relu(self.bn(self.a(x)))).sum(-1)


class Counter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(D, D)
        self.n = torch.nn.Buffer(torch.zeros(1, device="cuda"))

    def forward(self, x):
        self.n.add_(1.0)
        return self.a(x).sum(-1) + self.n


CASES = [("inplace_input", InplaceInput, False), ("bn_train", BNTrain, True), ("counter", Counter, False)]


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


def run_case(cls, train, dynagraph):
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
    asked, served = [], []
    orig_build = ct._maybe_build_dynagraph

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
    outs, ins, err = [], [], None
    try:
        torch.manual_seed(0)
        m = type(f"{cls.__name__}_{'ON' if dynagraph else 'OFF'}", (cls,), {})().cuda()
        m.train(train)
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        with torch.no_grad():
            for _ in range(2):
                for L in SHAPES:
                    x = make_input(L)
                    o = f(x)
                    torch.cuda.synchronize()
                    outs.append(o.detach().clone())
                    ins.append(x.detach().clone())
        state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:140]}"
        state = {}
    finally:
        ct._maybe_build_dynagraph = orig_build
        ct.CUDAGraphTreeManager.record_function = orig_rec
        lg.removeHandler(grab)
    tags = sorted({t.split("[", 1)[1].split("]", 1)[0] for t in grab.msgs if "fallback [" in t})
    return dict(asked=len(asked), served=len(served), rec=n_rec["n"], tags=tags,
                outs=outs, ins=ins, state=state, err=err)


def worst(a, b):
    if len(a) != len(b):
        return float("inf")
    return max((x - y).abs().max().item() for x, y in zip(a, b)) if a else 0.0


def main() -> int:
    bad = 0
    if os.environ.get("DG_PROBE_NOPROTECT"):
        # What the runner did before: every extra eager run on the real
        # tensors, nothing put back. For the before/after row in the docs.
        import contextlib
        from torch._inductor import dynagraph as dg
        dg.DynaGraphRunner._eager_args = lambda self, inputs: list(inputs)
        dg.DynaGraphRunner._unwritten = lambda self, inputs: contextlib.nullcontext()
        print("\n  [DG_PROBE_NOPROTECT] extra eager runs are unprotected (old behavior)")
    print(f"\n  shape stream {SHAPES} x 2 passes; per row: asked/served, records off->on, fallback tags, output diff, input diff, buffer diff")
    for name, cls, train in CASES:
        off = run_case(cls, train, False)
        on = run_case(cls, train, True)
        if off["err"] or on["err"]:
            print(f"    {name:<14} ERR {off['err'] or on['err']}")
            bad += 1
            continue
        d_out = worst(off["outs"], on["outs"])
        d_in = worst(off["ins"], on["ins"])
        keys = sorted(off["state"])
        d_st = worst([off["state"][k] for k in keys], [on["state"][k] for k in keys])
        # Outputs: 1e-6 between captures is cuBLAS picking per capture (docs/notes/BENCH.md);
        # the state -- what was written in place -- has to agree exactly.
        ok = on["served"] == on["asked"] == 1 and on["rec"] == 0 and d_out <= 1e-5 and d_in == 0 and d_st == 0
        bad += not ok
        print(f"    {name:<14} asked {on['asked']} served {on['served']}  rec {off['rec']}->{on['rec']}  "
              f"{','.join(on['tags']) or '-':<24} out {d_out:.1e}  in {d_in:.1e}  buffer {d_st:.1e}  "
              f"{'ok' if ok else 'FAIL'}")
        if name == "bn_train":
            rm_off, rm_on = off["state"]["bn.running_mean"], on["state"]["bn.running_mean"]
            print(f"      running_mean[:3] off {rm_off[:3].tolist()} on {rm_on[:3].tolist()}  "
                  f"num_batches_tracked off {int(off['state']['bn.num_batches_tracked'])} on {int(on['state']['bn.num_batches_tracked'])}")
        if name == "counter":
            print(f"      n off {float(off['state']['n'])} on {float(on['state']['n'])}")
    print("\n  " + ("all passed" if not bad else f"{bad} items failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
