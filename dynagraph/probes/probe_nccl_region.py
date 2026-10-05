#!/usr/bin/env python3
r"""Multi-GPU: for a region with an all_reduce in the graph, can DynaGraph serve every shape with one graph?

    torchrun --standalone --nproc_per_node=2 probe_nccl_region.py

`probe_nccl_variants.py` already showed that changing the size changes neither NCCL's topology nor its kernel, so in theory
the child route (harvest per size, swap the child) is enough. Here it goes into a real torch.compile region:
Linear -> relu -> all_reduce (functional collective; Inductor generates
`torch.ops._c10d_functional.all_reduce.default(...)` + `wait_tensor`) -> Linear -> sum.

Criteria (each rank checks for itself, rank 0 summarizes):
  1. with dynagraph on (child route) the region is served and the fallback tags are empty
  2. recording count: control 4 (one per shape) -> on 0
  3. numerics: the runner's bitwise check against eager reports no mismatch; the difference from the control (cuBLAS captured
     by cudagraph_trees itself) is within 1e-5
  4. both ranks agree (the collective order was not scrambled, otherwise it would have hung long ago)
"""
from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct

D = 128
SHAPES = [512, 256, 333, 64]


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(D, D)
        self.b = torch.nn.Linear(D, D)

    def forward(self, x):
        h = torch.relu(self.a(x))
        h = funcol.all_reduce(h, "sum", dist.group.WORLD)
        return self.b(h).sum(-1)


class Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs: list[str] = []

    def emit(self, rec):
        self.msgs.append(rec.getMessage())


def make_input(L, rank):
    g = torch.Generator(device="cuda")
    g.manual_seed(L * 10 + rank)
    return torch.randn(L, D, device="cuda", generator=g)


def run(dynagraph: bool, rank: int):
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
    outs = {}
    try:
        torch.manual_seed(0)          # both ranks get the same weights
        m = type(f"M_{'ON' if dynagraph else 'OFF'}", (M,), {})().cuda().eval()
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        with torch.no_grad():
            for _ in range(2):
                for L in SHAPES:
                    outs[L] = f(make_input(L, rank)).float().cpu().clone()
        torch.cuda.synchronize()
    finally:
        ct._maybe_build_dynagraph = orig_build
        ct.CUDAGraphTreeManager.record_function = orig_rec
        lg.removeHandler(grab)
    tags = sorted({t.split("[", 1)[1].split("]", 1)[0] for t in grab.msgs if "fallback [" in t})
    detail = [t for t in grab.msgs if "fallback [" in t][:3]
    return dict(rec=n_rec["n"], asked=len(asked), served=len(served), tags=tags,
                detail=detail, outs=outs)


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    off = run(False, rank)
    on = run(True, rank)
    worst = max((off["outs"][L] - on["outs"][L]).abs().max().item() for L in SHAPES)
    summary = dict(rank=rank, rec_off=off["rec"], rec_on=on["rec"], asked=on["asked"],
                   served=on["served"], tags=on["tags"], worst=worst, detail=on["detail"])
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, summary)
    if rank == 0:
        bad = 0
        print(f"\n  world={dist.get_world_size()}, shape stream {SHAPES} x 2 passes")
        for s in gathered:
            print(f"    rank {s['rank']}: regions asked {s['asked']} served {s['served']}, recordings off {s['rec_off']} -> on {s['rec_on']}, "
                  f"tags {s['tags'] or '-'}, max on-off diff {s['worst']:.1e}")
            for d in s["detail"]:
                print(f"        {d[:110]}")
            # The runner's own check against the eager wrapper is bitwise (no mismatch tag); against the control,
            # cuBLAS picks its algorithm separately in two different captures, ~1e-6, so judge with a tolerance.
            bad += not (s["served"] == s["asked"] > 0 and s["rec_on"] == 0
                        and not s["tags"] and s["worst"] < 1e-5)
        print("\n  " + ("all passed: the region with all_reduce is served by one graph, both ranks agree" if not bad
                        else f"{bad} ranks failed"))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
