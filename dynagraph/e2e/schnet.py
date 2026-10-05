"""SchNet training on QM9 (PyG's SchNet, 6 interaction layers, 128 hidden, cutoff 10 A, predicting U0).

128 molecules per batch; the atom and edge counts differ in every batch (survey: 155 distinct across 156 batches).
The radius graph is precomputed in data preparation (coordinates do not change during QM9 training); distances and
everything after them are inside the compiled region.
Usage: CUDA_VISIBLE_DEVICES=N python e2e/schnet.py [--batches 100] [--modes ...]
"""
import argparse
import os
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, os.path.dirname(__file__))
import harness
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import SchNet

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, default=100)
ap.add_argument("--batch-size", type=int, default=128)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--warm", type=int, default=20)
ap.add_argument("--modes", default=",".join(harness.MODES))
a = ap.parse_args()


class Net(torch.nn.Module):
    """SchNet.forward with the interaction graph given (PyG computes it inside with radius_graph)."""

    def __init__(self):
        super().__init__()
        self.m = SchNet(hidden_channels=a.hidden, num_filters=a.hidden, num_interactions=6,
                        num_gaussians=50, cutoff=10.0)

    def forward(self, z, pos, edge_index, batch, n_graphs: int):
        m = self.m
        h = m.embedding(z)
        row, col = edge_index
        ew = (pos[row] - pos[col]).norm(dim=-1)
        ea = m.distance_expansion(ew)
        for interaction in m.interactions:
            h = h + interaction(h, edge_index, ew, ea)
        h = m.lin2(m.act(m.lin1(h)))
        # One graph more than there are molecules: pad_to_max parks its atoms there.
        return m.readout(h, batch, dim=0, dim_size=n_graphs + 1)[:n_graphs]


def load():
    from torch_geometric.datasets import QM9
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import radius_graph

    ds = QM9(os.environ.get("DG_DATA", "/workspace/_deps/data") + "/QM9")
    y = ds._data.y[:, 7]
    mean, std = y.mean().item(), y.std().item()
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, generator=torch.Generator().manual_seed(0))
    out = []
    for b in loader:
        if b.num_graphs != a.batch_size:
            continue
        pos, bt = b.pos.cuda(), b.batch.cuda()
        ei = radius_graph(pos, r=10.0, batch=bt, max_num_neighbors=32)
        out.append((b.z.cuda(), pos, ei, bt, ((b.y[:, 7] - mean) / std).cuda().view(-1, 1), b.num_graphs))
        if len(out) == a.batches:
            break
    return out


def pad(batches):
    """Pad atoms to the max + 2 and edges to the max: the extra atoms go to graph n_graphs (sliced off),
    padded edges join the two extra atoms, 100 A apart (past the cutoff, and not a zero distance whose
    norm has no gradient)."""
    n = max(z.shape[0] for z, *_ in batches) + 2
    e = max(ei.shape[1] for _, _, ei, *_ in batches)
    out = []
    for z, pos, ei, bt, y, g in batches:
        zp = z.new_zeros(n)
        zp[: z.shape[0]] = z
        pp = pos.new_zeros(n, 3)
        pp[: pos.shape[0]] = pos
        pp[n - 1, 0] = 100.0
        bp = bt.new_full((n,), g)
        bp[: bt.shape[0]] = bt
        ep = ei.new_empty(2, e)
        ep[0], ep[1] = n - 2, n - 1
        ep[:, : ei.shape[1]] = ei
        out.append((zp, pp, ep, bp, y, g))
    return out


def main():
    batches = load()
    shapes = {(b[0].shape[0], b[2].shape[1]) for b in batches}
    print(f"{len(batches)} batches, {len(shapes)} distinct (atoms, edges) shapes; atoms {min(s[0] for s in shapes)}..{max(s[0] for s in shapes)}, "
          f"edges {min(s[1] for s in shapes)}..{max(s[1] for s in shapes)}")

    def make():
        torch.manual_seed(0)
        m = Net().cuda().train()
        opt = torch.optim.Adam(m.parameters(), lr=5e-4)

        def step(f, b):
            z, pos, ei, bt, y, g = b
            with harness.amp():
                loss = F.mse_loss(f(z, pos, ei, bt, g).float(), y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            return loss.detach()

        return m, step

    harness.run(make, batches, a.modes.split(","), warm=a.warm, label="schnet", pad=pad)


if __name__ == "__main__":
    main()
