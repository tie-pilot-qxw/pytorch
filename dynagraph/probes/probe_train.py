#!/usr/bin/env python3
r"""Training: two regions, forward + backward, with argument names primals_ / tangents_ / saved-intermediate names.

`_input_symbol_map` used to recognize only `argN_1 / sN / bufN`, and none of the training wrapper's args matched (no-symbol-args),
so DynaGraph had never served a single training region -- even though variable-length training was its original motivation. Here two
models, Linear -> Dropout -> Linear and Linear -> BatchNorm1d -> Linear, in train mode, each step forward + loss.backward(),
4 lengths x 2 passes. Compare dynagraph off / on: regions asked, regions served, recording count, fallback tags, plus
each step's loss and each parameter's .grad after the run (dropout is random, so only the BN model is compared).
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


def make_dropout():
    return torch.nn.Sequential(torch.nn.Linear(D, D), torch.nn.Dropout(0.1), torch.nn.Linear(D, D))


def make_bn():
    return torch.nn.Sequential(torch.nn.Linear(D, D), torch.nn.BatchNorm1d(D), torch.nn.ReLU(), torch.nn.Linear(D, D))


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


def run(make, dynagraph):
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
    losses, grads, err = [], {}, None
    try:
        torch.manual_seed(0)
        m = make().cuda().train()
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        for _ in range(2):
            for L in SHAPES:
                m.zero_grad(set_to_none=True)
                loss = f(make_input(L)).square().mean()
                loss.backward()
                torch.cuda.synchronize()
                losses.append(loss.detach().clone())
        grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
        state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:160]}"
        state = {}
    finally:
        ct._maybe_build_dynagraph = orig_build
        ct.CUDAGraphTreeManager.record_function = orig_rec
        lg.removeHandler(grab)
    tags = sorted({t.split("[", 1)[1].split("]", 1)[0] for t in grab.msgs if "fallback [" in t})
    return dict(asked=len(asked), served=len(served), rec=n_rec["n"], tags=tags,
                losses=losses, grads=grads, state=state, err=err)


def worst(a, b):
    if len(a) != len(b):
        return float("inf")
    return max((x - y).abs().max().item() for x, y in zip(a, b)) if a else 0.0


def main() -> int:
    bad = 0
    print(f"\n  shape stream {SHAPES} x 2 passes, forward + backward each step")
    for name, make, compare in (("dropout", make_dropout, False), ("batchnorm", make_bn, True)):
        off = run(make, False)
        on = run(make, True)
        if off["err"] or on["err"]:
            print(f"    {name:<10} ERR {off['err'] or on['err']}")
            bad += 1
            continue
        line = (f"    {name:<10} asked {on['asked']} served {on['served']}  recordings {off['rec']}->{on['rec']}  "
                f"{','.join(on['tags']) or '-':<28}")
        if compare:
            d_loss = worst(off["losses"], on["losses"])
            keys = sorted(off["grads"])
            d_grad = worst([off["grads"][k] for k in keys], [on["grads"].get(k, off["grads"][k] * 0 + 1e9) for k in keys])
            skeys = sorted(off["state"])
            d_state = worst([off["state"][k] for k in skeys], [on["state"][k] for k in skeys])
            ok = on["served"] == on["asked"] >= 2 and on["rec"] == 0 and d_loss <= 1e-5 and d_grad <= 1e-4 and d_state <= 1e-5
            line += f" loss {d_loss:.1e}  grad {d_grad:.1e}  buffer {d_state:.1e}"
        else:
            ok = on["served"] == on["asked"] >= 2 and on["rec"] == 0
            line += " (dropout is random, numbers not compared)"
        bad += not ok
        print(line + f"  {'OK' if ok else 'FAIL'}")
    print("\n  " + ("all passed" if not bad else f"{bad} checks failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
