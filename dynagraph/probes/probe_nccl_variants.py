#!/usr/bin/env python3
r"""First step on the multi-GPU axis: when an NCCL collective changes message size, do the nodes/kernels in the graph change?

    torchrun --nproc_per_node=2 probe_nccl_variants.py

`_c10d_functional.*` is a third kind of entry point (neither extern_kernels nor an aten fallback),
and today the whole region is rejected as extern-launch. To bring it into the child-graph route, the prerequisites are the same as for cuBLAS:
  (A) how many nodes one call produces, and whether they are kernel nodes;
  (C) whether changing size changes the kernel (NCCL picks algo/protocol by message size: LL / LL128 / Simple,
      ring / tree); a changed kernel can be handled by child replacement; a changed **node count** needs SWITCH.

Structural probe only, no timing: capture one all_reduce-only graph per size and list it with cudaGraphGetNodes /
cudaGraphNodeGetType / cuGraphKernelNodeGetParams + cuFuncGetName.
Tensors are small (<= 64 MB), so this does not disturb anyone else.

Output format: one line per size with [node types...] + kernel names, then "is the topology constant / is the kernel constant".
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def nodes_of(raw: int):
    from cuda.bindings import driver as cd, runtime as cr
    from torch.cuda._utils import _check_cuda_bindings as ck

    n = ck(cr.cudaGraphGetNodes(raw))[1]
    nds = ck(cr.cudaGraphGetNodes(raw, n))[0]
    out = []
    for nd in nds:
        t = ck(cr.cudaGraphNodeGetType(nd)).name.replace("cudaGraphNodeType", "")
        name = ""
        if t == "Kernel":
            try:
                p = ck(cd.cuGraphKernelNodeGetParams(nd))
                name = ck(cd.cuFuncGetName(p.func)).decode()
            except Exception as e:  # noqa: BLE001
                name = f"<{type(e).__name__}>"
        out.append((t, name))
    return out


def main() -> int:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    # Element counts: cover the typical switch range from LL to Simple (NCCL thresholds are in bytes; fp32 here)
    sizes = [256, 4096, 65536, 1 << 20, 4 << 20, 16 << 20]
    holds = []
    rows = {}
    for n in sizes:
        x = torch.ones(n, device=dev)
        # warmup: NCCL's first communication builds the communicator, which cannot happen inside a capture
        for _ in range(3):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph(keep_graph=True)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(g, stream=s):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        holds.append(g)
        rows[n] = nodes_of(g.raw_cuda_graph())
        print(f"    [rank {rank}] n={n} captured, {len(rows[n])} nodes", flush=True)
        # the captured graph really works and gives the right result
        x.fill_(1.0)
        g.replay()
        torch.cuda.synchronize()
        ok = bool((x == float(world)).all().item())
        rows[n].append(("check", "OK" if ok else "BAD"))

    if rank == 0:
        print(f"\n  world={world}  all_reduce fp32, one graph per size:")
        topos, kernels = set(), set()
        for n, r in rows.items():
            types = [t for t, _ in r if t != "check"]
            names = [nm for t, nm in r if t == "Kernel"]
            chk = [nm for t, nm in r if t == "check"][0]
            topos.add(tuple(types))
            kernels.add(tuple(names))
            print(f"    n={n:>9} ({n * 4 / 2**20:6.2f} MiB)  {len(types)} nodes {types}  "
                  f"kernel={[k[:48] for k in names]}  replay {chk}")
        print(f"\n  distinct topologies {len(topos)}, distinct kernel combinations {len(kernels)}")
        if len(topos) == 1 and len(kernels) == 1:
            print("  -> neither node count nor kernel changes with size: NCCL nodes can use the same child route as cuBLAS, "
                  "or even just parameter patching")
        elif len(topos) == 1:
            print("  -> constant node count, kernel changes with size: child-graph replacement works (same as cuBLAS)")
        else:
            print("  -> node count changes with size: only SWITCH or splitting the region works")
    # No barrier / destroy: in this container the NCCL teardown hangs past the
    # timeout after everything has printed (`_nccl_min.py`, not included in this repo),
    # so the ranks just flush and leave.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
