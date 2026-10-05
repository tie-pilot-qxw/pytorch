#!/usr/bin/env python3
r"""Elastic expert parallelism: the same compiled region, changing the parallel width at runtime.

    torchrun --standalone --nproc_per_node=4 probe_elastic_ep.py

Elasticity is simulated with a fixed 4 ranks: the EP width W is the size of the subgroup doing all_to_all.
W=2 means two groups of 2 ranks each, W=4 means one whole group. When the width changes, three things change at once:

  1. the number of experts each rank holds, E/W -- the weight shape changes (a backed symbol)
  2. the all_to_all buffer size -- the communication shape changes
  3. the communicator itself -- the handle captured into the graph changes

On top of that, the token count differs every step (the shape axis, the one DynaGraph already serves).

Criteria (each rank checks its own, rank 0 summarizes):
  1. at every width the region is served, and the numerics match eager
  2. results are still correct after changing width -- the only evidence that "a stale communicator is not silently in use"
  3. recording count: the control group records one graph per shape; with DynaGraph on, at most one per width
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--experts", type=int, default=12)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--cap", type=int, default=32, help="capacity per expert (fixed length, no .item() for now)")
p.add_argument("--dg", type=int, default=1)
a = p.parse_args()

tags: list[str] = []


class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "DynaGraph fallback [" in m:
            tags.append(m.split("[", 1)[1].split("]", 1)[0])


logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph")
lg.setLevel(logging.INFO)
lg.addHandler(_Grab())

from torch._inductor import config as ic  # noqa: E402

ic.triton.dynagraph = bool(a.dg)
ic.triton.dynagraph_extern_child = True
ic.force_disable_caches = True

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)

# width -> (the subgroup this rank belongs to, index within the group). Widths are the factors of world that divide the expert count,
# so it also runs with 3 ranks (widths 1 and 3); 4 cards are not required.
GROUPS: dict[int, tuple[object, int]] = {}
WIDTHS = [w for w in range(1, world + 1) if world % w == 0 and a.experts % w == 0]
for W in WIDTHS:
    g = None
    for base in range(0, world, W):
        members = list(range(base, base + W))
        pg = dist.new_group(ranks=members)
        if rank in members:
            g = (pg, members.index(rank))
    GROUPS[W] = g


class MoE(torch.nn.Module):
    """One layer: route -> all_to_all dispatch -> per-expert GEMM -> all_to_all combine.

    Expert weights are split by width: at width W this rank holds E/W experts.
    """

    def __init__(self, experts: int, dim: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(experts, dim, dim, device=dev) * 0.05)
        self.router = torch.nn.Linear(dim, experts, device=dev)

    def forward(self, x, pg, gsize, my_experts):
        # fixed capacity: each expert receives cap tokens, truncated if more, zero-padded if fewer.
        cap = a.cap
        n_local = my_experts.numel()
        logits = self.router(x)
        top = logits.argmax(-1)
        # lay tokens out as (experts, cap, dim) by target expert; this step is pure Triton
        slot = torch.zeros(self.w.shape[0], cap, x.shape[-1], device=x.device, dtype=x.dtype)
        idx = torch.arange(x.shape[0], device=x.device) % cap
        slot.index_put_((top, idx), x, accumulate=True)
        # dispatch: each rank keeps only the slots of its own experts
        send = slot.view(gsize, n_local * cap, -1).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=pg)
        # per-expert GEMM
        h = torch.bmm(
            recv.view(n_local, gsize * cap, -1),
            self.w[my_experts],
        )
        h = torch.relu(h)
        # combine
        back = h.view(gsize, n_local * cap, -1).contiguous()
        out = torch.empty_like(back)
        dist.all_to_all_single(out, back, group=pg)
        return out.sum(0).sum(0)


def experts_of(W: int, gi: int, E: int) -> torch.Tensor:
    per = E // W
    return torch.arange(gi * per, (gi + 1) * per, device=dev)


def main() -> int:
    torch.manual_seed(0)
    m = MoE(a.experts, a.dim)
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")

    from torch._dynamo.utils import counters
    from torch._inductor import cudagraph_trees as ct

    n_comp = {"v": 0}
    from torch._inductor import compile_fx as _cfx
    _orig_cfx = _cfx.compile_fx

    def _count_cfx(*x, **kw):
        n_comp["v"] += 1
        return _orig_cfx(*x, **kw)

    _cfx.compile_fx = _count_cfx
    n_rec = {"v": 0}
    orig = ct.CUDAGraphNode.__init__

    def rec(self, *x, **kw):
        n_rec["v"] += 1
        return orig(self, *x, **kw)

    ct.CUDAGraphNode.__init__ = rec

    worst = 0.0
    plan = []
    for W in sorted(GROUPS):
        for T in (64, 96, 64):
            plan.append((W, T))
    # go back to the first width: this step is the real "is it still correct after switching back"
    if len(GROUPS) > 1:
        plan.append((sorted(GROUPS)[0], 96))

    with torch.no_grad():
        for W, T in plan:
            pg, gi = GROUPS[W]
            mine = experts_of(W, gi, a.experts)
            g = torch.Generator(device=dev)
            g.manual_seed(1000 + T + 7 * W + rank)
            x = torch.randn(T, a.dim, device=dev, generator=g)
            torch._dynamo.mark_dynamic(x, 0)
            ref = m(x, pg, W, mine)
            out = f(x, pg, W, mine)
            torch.cuda.synchronize()
            d = (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
            worst = max(worst, d)

    ct.CUDAGraphNode.__init__ = orig
    _cfx.compile_fx = _orig_cfx

    # Check that guard directly: if the communicator under the same group name changes, the harvest key must change with it.
    # (The case where the compiled artifact is the same, the shape is the same, and only the group was rebuilt relies on exactly this.)
    key_moves = None
    if len(GROUPS) > 1:
        from torch._inductor import dynagraph as dg

        cls = next(
            getattr(dg, n)
            for n in dir(dg)
            if isinstance(getattr(dg, n), type) and hasattr(getattr(dg, n), "_deps_key")
        )
        import torch.distributed.distributed_c10d as c10d

        stub = cls.__new__(cls)
        stub.site_deps = [("_c10d_functional::all_reduce", ("dg_probe_group",))]
        real = c10d._resolve_process_group
        ws = sorted(GROUPS)
        try:
            c10d._resolve_process_group = lambda nm: GROUPS[ws[0]][0]
            k0 = cls._deps_key(stub)
            c10d._resolve_process_group = lambda nm: GROUPS[ws[-1]][0]
            k1 = cls._deps_key(stub)
        finally:
            c10d._resolve_process_group = real
        key_moves = k0 != k1 and k0 != () and k1 != ()
    seen = sorted(set(tags))
    # Whether changing width recompiles the whole region is the precondition for this to work at all: if it really recompiles,
    # it is not "one graph across widths" but, like vLLM today, a restart on every resize.
    km = {None: "skipped", True: "changed", False: "unchanged(!)"}[key_moves]
    line = (
        f"  rank {rank}  widths {sorted(GROUPS)}  compiles {n_comp['v']}  recordings {n_rec['v']}"
        f"  max rel diff {worst:.1e}  harvest key vs communicator {km}  tags {seen or '-'}"
    )
    ok = worst < 1e-4 and (key_moves is not False)
    gather = [None] * world
    dist.all_gather_object(gather, (line, ok, n_comp["v"], seen))
    if rank == 0:
        for ln, _o, _n, _s in gather:
            print(ln, flush=True)
        allok = all(o for _l, o, _n, _s in gather)
        comps = {n for _l, _o, n, _s in gather}
        print(f"  compile counts {sorted(comps)} ({len(GROUPS)} widths; one per width means the width went into a guard)")
        print("  all passed: numerics still correct after changing width" if allok else "  FAILED")
        return 0 if allok else 1
    return 0


rc = main()
sys.stdout.flush()
os._exit(rc)
