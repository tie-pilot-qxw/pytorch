"""MACE force-field inference (one MD step: energy + forces) on consecutive frames of the MD17 aspirin trajectory, one frame at a time.

The atom count is fixed (21); the neighbor edge count (r_max 5 A) changes from frame to frame. The model is randomly
initialized, with an architecture on the scale of MACE-OFF small (2 interaction layers, 64x0e+64x1o, max_ell 3,
correlation 3); compilation goes through MACE's own prepare/simplify path (the one mace.calculators' compile_mode uses).
fp32: MD needs the precision, so no AMP; the GEMMs are e3nn's small fp32 matmuls.
Usage: PYTHONPATH=${DG_DEPS:-/workspace/_deps}/mace_site:$PYTHONPATH python e2e/mace_md.py [--frames 100]
"""
import argparse
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import sys

sys.path.insert(0, os.path.dirname(__file__))
import harness
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=100)
ap.add_argument("--npz", default=os.environ.get("DG_DATA", "/workspace/_deps/data") + "/MD17/aspirin/raw/md17_aspirin.npz",
                help="MD22: $DG_DATA/MD22/md22_double-walled_nanotube.npz (370 atoms)")
ap.add_argument("--stride", type=int, default=1, help="use frames 0, s, 2s, ...")
ap.add_argument("--r-max", type=float, default=5.0)
ap.add_argument("--warm", type=int, default=20)
ap.add_argument("--modes", default="eager,compile,trees,dg,pad")
a = ap.parse_args()

from e3nn import o3
from mace import data as mdata
from mace import modules, tools
from mace.tools.compile import prepare

NPZ = np.load(a.npz)
Z = sorted(int(x) for x in set(NPZ["z"].tolist()))


def build_model():
    torch.manual_seed(0)
    return modules.MACE(
        r_max=a.r_max,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2,
        num_elements=len(Z),
        hidden_irreps=o3.Irreps("64x0e+64x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.zeros(len(Z)),
        avg_num_neighbors=10.0,
        atomic_numbers=Z,
        correlation=3,
        gate=torch.nn.functional.silu,
    ).cuda()


def load():
    import ase
    from mace.tools import torch_geometric

    R, z = NPZ["R"], NPZ["z"]
    table = tools.AtomicNumberTable(Z)
    out = []
    for i in range(0, a.frames * a.stride, a.stride):
        cfg = mdata.config_from_atoms(ase.Atoms(numbers=z, positions=R[i]))
        d = mdata.AtomicData.from_config(cfg, z_table=table, cutoff=a.r_max)
        b = next(iter(torch_geometric.dataloader.DataLoader([d], batch_size=1)))
        d = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.to_dict().items()}
        d["positions"].requires_grad_(True)  # forces are -dE/dpos
        out.append(d)
    return out


class Step(torch.nn.Module):
    """Energy and forces of one frame: what an MD integrator asks for every step."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, batch):
        out = self.m(batch, training=True, compute_force=True)
        return out["energy"], out["forces"]


def pad(frames):
    """Pad edges to the most any frame has (K=1, one static graph): the extra edges join two atoms
    that are further apart than the cutoff in every frame, so the radial cutoff makes their messages,
    and their share of every force, exactly zero. The atom count does not change along a trajectory."""
    e = max(f["edge_index"].shape[1] for f in frames)
    pos = torch.stack([f["positions"].detach() for f in frames])
    d = (pos[:, :, None, :] - pos[:, None, :, :]).norm(dim=-1).amin(0)
    i, j = divmod(int(d.argmax()), d.shape[0])
    assert float(d[i, j]) > 2 * a.r_max, float(d[i, j])
    out = []
    for f in frames:
        g = dict(f)
        k = e - f["edge_index"].shape[1]
        far = torch.tensor([[i], [j]], device=f["edge_index"].device, dtype=f["edge_index"].dtype)
        g["edge_index"] = torch.cat([f["edge_index"], far.expand(2, k)], 1)
        for key in ("shifts", "unit_shifts"):
            g[key] = torch.cat([f[key], f[key].new_zeros(k, 3)])
        out.append(g)
    if os.environ.get("PADCHECK"):
        m = Step(prepare(build_model)())
        for f, g in list(zip(frames, out))[:3]:
            e0, f0 = m(f)
            e1, f1 = m(g)
            print(f"pad check: energy {float(e0.sum()):.6f} vs {float(e1.sum()):.6f}, "
                  f"max |dF| {float((f0 - f1).abs().max()):.3e} (max |F| {float(f0.abs().max()):.3e}), "
                  f"edges {f['edge_index'].shape[1]} -> {e}", flush=True)
    return out


def main():
    frames = load()
    ne = [f["edge_index"].shape[1] for f in frames]
    print(f"{os.path.basename(a.npz)} {len(frames[0]['node_attrs'])} atoms, {len(frames)} frames, edges {min(ne)}..{max(ne)}, {len(set(ne))} distinct, changed between consecutive frames "
          f"{sum(x != y for x, y in zip(ne, ne[1:]))}/{len(ne) - 1}")

    def make():
        m = prepare(build_model)()
        s = Step(m)

        def step(f, b):
            # An inference loop tells the cudagraph runtime where a step starts (the previous step's
            # outputs are dead); without it every call of the loop counts as the same step.
            torch.compiler.cudagraph_mark_step_begin()
            e, forces = f(b)
            return e.detach().sum() + forces.detach().abs().sum() * 1e-3

        return s, step

    harness.run(make, frames, a.modes.split(","), warm=a.warm, label="mace", pad=pad)


if __name__ == "__main__":
    main()
