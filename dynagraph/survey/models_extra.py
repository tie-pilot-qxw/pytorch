"""
Probes to fill out the list: the categories that really are data-dependent.

Of the current 131 models, torchvision classification + TIMM make up 62%, and that is the most regular, least data-dependent kind.
The "arbitrary messy open-source training code" and LLM serving the user asked about are barely represented in the list.

This writes the **core pattern** of each of those categories in plain PyTorch, without depending on PyG / DGL / TorchSparse,
so we don't have to install a pile of extensions that need compiling just for a probe. Each keeps only the small piece that creates the data dependence.

After the full run, merge into `models.py` with spec prefix `extra:`.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn

DEV = os.environ.get("DYNAGRAPH_DEVICE", "cuda")


class GraphSAGELike(nn.Module):
    """
    GNN neighbor aggregation. The edge count changes every step; the output row count of `scatter_add` is set by the node count,
    and the node count comes from the sampling result -- the typical case the host cannot predict.
    """

    def __init__(self, dim=128):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        msg = self.lin_neigh(x)[src]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        return torch.relu(self.lin_self(x) + agg)


class NeighborListMD(nn.Module):
    """
    Molecular dynamics: build neighbor pairs by radius. The pair count is data-dependent and differs every step.
    """

    def __init__(self, dim=64, cutoff=1.0):
        super().__init__()
        self.cutoff = cutoff
        self.mlp = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, pos):
        d = torch.cdist(pos, pos)
        mask = (d < self.cutoff) & (d > 0)
        idx = mask.nonzero(as_tuple=False)          # row count is data-dependent
        dist = d[idx[:, 0], idx[:, 1]].unsqueeze(-1)
        feat = self.mlp(dist)
        out = torch.zeros(pos.shape[0], feat.shape[-1], device=pos.device, dtype=feat.dtype)
        out.index_add_(0, idx[:, 0], feat)
        return out


class SparseVoxel(nn.Module):
    """
    Point cloud / sparse conv: compute only on active voxels; the active count varies with the input.
    """

    def __init__(self, dim=64):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, feats, occupancy):
        active = (occupancy > 0.5).nonzero(as_tuple=True)[0]   # length is data-dependent
        sel = feats[active]
        out = torch.zeros_like(feats)
        out[active] = torch.relu(self.lin(sel))
        return out


class NMSHead(nn.Module):
    """Detection post-processing: filter boxes by score; how many remain is data-dependent."""

    def __init__(self, dim=256):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.box = nn.Linear(dim, 4)

    def forward(self, feats):
        s = self.score(feats).squeeze(-1).sigmoid()
        keep = (s > 0.5).nonzero(as_tuple=True)[0]
        return self.box(feats[keep]), s[keep]


class MoERouter(nn.Module):
    """
    MoE routing: how many tokens each expert gets is set by the top-k result.
    This is the core pattern of the LLM serving scenario with the worst graph memory cost.
    """

    def __init__(self, dim=256, n_expert=4):
        super().__init__()
        self.gate = nn.Linear(dim, n_expert)
        self.experts = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_expert))

    def forward(self, x):
        logits = self.gate(x)
        choice = logits.argmax(-1)
        out = torch.zeros_like(x)
        for i, e in enumerate(self.experts):
            idx = (choice == i).nonzero(as_tuple=True)[0]      # token count per expert is not fixed
            if idx.numel():
                out[idx] = e(x[idx])
        return out


class VarLenPack(nn.Module):
    """Variable-length sequences: pack by valid length; the total token count differs every batch."""

    def __init__(self, dim=256):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, lengths):
        b, t, d = x.shape
        ar = torch.arange(t, device=x.device)
        mask = ar.unsqueeze(0) < lengths.unsqueeze(1)
        packed = x[mask]                                       # row count is data-dependent
        return self.lin(packed).sum(0)


class NeighborSample(nn.Module):
    """
    The **sampling** stage of a GNN, not the aggregation stage.

    The `gnn` probe (pure index_add_ aggregation) is clean, because the output row count is set by the node count.
    The real trouble is here: starting from the seed nodes, filter the one-hop neighbors; **the edge count is data-dependent**
    and differs every batch. PyG/DGL's NeighborLoader does this at every step.
    """

    def __init__(self, dim=128):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, edge_index, seed_mask):
        keep = seed_mask[edge_index[1]]
        sub = edge_index[:, keep]                       # column count is data-dependent
        msg = self.lin(x)[sub[0]]
        out = torch.zeros_like(x)
        out.index_add_(0, sub[1], msg)
        return out


class DecodeStep(nn.Module):
    """
    One LLM serving decode step: every sequence in the batch has a different KV length, and it grows every step.

    This is the root reason vLLM / SGLang use piecewise cudagraph,
    and the target scenario the user cares about most. Only the core pattern "take the valid KV by each sequence's length" is kept.
    """

    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.q = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x, kv, kv_lens):
        b, t, _ = kv.shape
        ar = torch.arange(t, device=kv.device)
        mask = ar.unsqueeze(0) < kv_lens.unsqueeze(1)   # every sequence has a different length
        valid = kv[mask]                                 # total row count is data-dependent
        q = self.q(x)
        scores = q @ valid.t()
        w = scores.softmax(-1)
        return self.o(w @ valid)


def _rand_edges(n_node, n_edge, dev):
    return torch.randint(0, n_node, (2, n_edge), device=dev)


_BUILD = {
    "gnn": lambda: (GraphSAGELike(),
                    (torch.randn(512, 128), _rand_edges(512, 2048, "cpu")), {}),
    "md": lambda: (NeighborListMD(), (torch.randn(256, 3),), {}),
    "sparse": lambda: (SparseVoxel(),
                       (torch.randn(1024, 64), torch.rand(1024)), {}),
    "nms": lambda: (NMSHead(), (torch.randn(512, 256),), {}),
    "moe": lambda: (MoERouter(), (torch.randn(128, 256),), {}),
    "varlen": lambda: (VarLenPack(),
                       (torch.randn(8, 64, 256), torch.randint(1, 64, (8,))), {}),
    "gnn_sample": lambda: (NeighborSample(),
                           (torch.randn(512, 128), _rand_edges(512, 2048, "cpu"),
                            torch.rand(512) > 0.7), {}),
    "decode": lambda: (DecodeStep(),
                       (torch.randn(8, 256), torch.randn(8, 128, 256),
                        torch.randint(1, 128, (8,))), {}),
}


def build(name: str):
    """Returns (model, args, kwargs). The caller is responsible for wrapping it in a device context or meta."""
    if name not in _BUILD:
        raise ValueError(f"unknown extra probe: {name!r}, choices {sorted(_BUILD)}")
    return _BUILD[name]()


ALL = sorted(_BUILD)
