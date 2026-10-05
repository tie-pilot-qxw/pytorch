"""GraphSAGE + NeighborLoader training on ogbn-arxiv (the official PyG example's config: 3 layers, 256 hidden, fanout 15/10/5).

Every sampled batch has a different node count and edge count: this is the "200/200 batches all have different shapes"
scenario from the original survey.
All batches are sampled up front and put on the GPU; timing covers only model forward + backward + Adam.
Usage: CUDA_VISIBLE_DEVICES=N python e2e/sage.py [--batches 100] [--modes eager,compile,trees,dg]
Data: $DG_DATA/ogb (DG_DATA defaults to /workspace/_deps/data).
"""
import argparse
import os
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, os.path.dirname(__file__))
import harness
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, default=100)
ap.add_argument("--batch-size", type=int, default=1024)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--fanout", default="15,10")
ap.add_argument("--undirected", action="store_true")
ap.add_argument("--warm", type=int, default=20)
ap.add_argument("--modes", default=",".join(harness.MODES))
a = ap.parse_args()


class SAGE(torch.nn.Module):
    def __init__(self, cin, hid, cout, n):
        super().__init__()
        dims = [cin] + [hid] * (n - 1) + [cout]
        self.convs = torch.nn.ModuleList(SAGEConv(i, o) for i, o in zip(dims, dims[1:]))
        self.bns = torch.nn.ModuleList(torch.nn.BatchNorm1d(hid) for _ in range(n - 1))

    def forward(self, x, edge_index, nseed: int):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.bns):
                x = F.dropout(F.relu(self.bns[i](x)), p=0.0, training=self.training)
        return x[:nseed]


def load():
    from ogb.nodeproppred import PygNodePropPredDataset

    ds = PygNodePropPredDataset("ogbn-arxiv", root=os.path.join(os.environ.get("DG_DATA", "/workspace/_deps/data"), "ogb"))
    d = ds[0]
    d.y = d.y.view(-1)
    import torch_geometric.transforms as T

    if a.undirected:
        d = T.ToUndirected()(d)
    split = ds.get_idx_split()
    loader = NeighborLoader(d, num_neighbors=[int(x) for x in a.fanout.split(",")],
                            input_nodes=split["train"], batch_size=a.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(0))
    out = []
    for b in loader:
        if b.batch_size != a.batch_size:
            continue
        out.append((b.x.cuda(), b.edge_index.cuda(), b.y[: b.batch_size].cuda(), b.batch_size))
        if len(out) == a.batches:
            break
    return ds.num_features, ds.num_classes, out


def pad(batches):
    """Pad to max node count + 1: the extra last node receives only the padding self-loop edges, real nodes' neighbors are unchanged."""
    n = max(x.shape[0] for x, _, _, _ in batches) + 1
    e = max(ei.shape[1] for _, ei, _, _ in batches)
    out = []
    for x, ei, y, bs in batches:
        xp = x.new_zeros(n, x.shape[1])
        xp[: x.shape[0]] = x
        ep = ei.new_full((2, e), n - 1)
        ep[:, : ei.shape[1]] = ei
        out.append((xp, ep, y, bs))
    return out


def main():
    cin, ncls, batches = load()
    shapes = {(x.shape[0], e.shape[1]) for x, e, _, _ in batches}
    print(f"{len(batches)} batches, {len(shapes)} distinct (nodes, edges) shapes; nodes "
          f"{min(s[0] for s in shapes)}..{max(s[0] for s in shapes)}, edges {min(s[1] for s in shapes)}..{max(s[1] for s in shapes)}")

    def make():
        torch.manual_seed(0)
        m = SAGE(cin, a.hidden, ncls, len(a.fanout.split(","))).cuda().train()
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)

        def step(f, b):
            x, ei, y, bs = b
            with harness.amp():
                out = f(x, ei, bs)
                loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            return loss.detach()

        return m, step

    harness.run(make, batches, a.modes.split(","), warm=a.warm, label="sage", pad=pad)


if __name__ == "__main__":
    main()
