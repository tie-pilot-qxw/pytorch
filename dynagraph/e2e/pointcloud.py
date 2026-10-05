"""Point-cloud sparse-convolution inference: SECOND's VoxelBackBone8x (OpenPCDet) on real LiDAR frames from KITTI raw drive 0093.

Voxelization follows SECOND's KITTI config (0.05x0.05x0.1 m, range x[0,70.4] y[-40,40] z[-3,1]) using spconv's
PointToVoxel; each level's kernel map is computed during data preparation with spconv's own get_indice_pairs (the
coordinates within a frame do not change, same reasoning as QM9's radius graph). Inside the compiled region every
layer is the torchsparse / MinkowskiEngine dataflow:
    gather by kernel map -> GEMM grouped by the 27 offsets -> scatter-add -> BN -> ReLU
The grouped GEMM is DeepGEMM's m_grouped_bf16_gemm_nt_contiguous (sparse conv = an MoE with 27 experts);
each group's row count is padded to the alignment (128), and the padding rows have layout = -1 and scatter into a dump row.

Deviation from SECOND: DeepGEMM's grouped GEMM requires K % 64 == 0, so every channel count is padded to 64 wide (the
padded channels have zero weights and stay 0); the GEMMs of the first two levels (16/32 channels) therefore do 2~4x
the work. Every mode uses the same operator, so the comparison is fair.

Shapes: per frame, the active voxel counts of the 4 levels and the row counts of the 7 kernel maps all differ (survey:
433 frames, 433 distinct 4-tuples, deep-level max/mean up to 1.9).
Usage: PYTHONPATH=${DG_DEPS:-/workspace/_deps}/deepgemm-src:$PYTHONPATH python e2e/pointcloud.py [--frames 100]
Data: $DG_DATA/kitti/**/velodyne_points/data/*.bin (DG_DATA defaults to /workspace/_deps/data).
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import dgemm  # noqa: F401  registers dgemm::grouped_mm_out and its declaration
import harness
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=100)
ap.add_argument("--warm", type=int, default=20)
ap.add_argument("--modes", default=",".join(harness.MODES))
a = ap.parse_args()

W = 64  # channel width after padding (see above)
ALIGN = dgemm.deep_gemm.get_mk_alignment_for_contiguous_layout()
# (in level, out level, stride, padding, real Cin, real Cout); subm layers keep the level
LAYERS = [
    (0, 0, 1, 1, 4, 16), (0, 0, 1, 1, 16, 16),                                  # conv_input, conv1
    (0, 1, 2, 1, 16, 32), (1, 1, 1, 1, 32, 32), (1, 1, 1, 1, 32, 32),         # conv2
    (1, 2, 2, 1, 32, 64), (2, 2, 1, 1, 64, 64), (2, 2, 1, 1, 64, 64),         # conv3
    (2, 3, 2, (0, 1, 1), 64, 64), (3, 3, 1, 1, 64, 64), (3, 3, 1, 1, 64, 64),  # conv4
]


def kernel_maps(coords, shape):
    """Per level the active coordinates, and per layer (gather, scatter, layout) in grouped layout."""
    from spconv.core import ConvAlgo
    from spconv.pytorch import ops

    ind = torch.cat([coords.new_zeros(coords.shape[0], 1), coords], 1).int()
    levels = [(ind, shape)]
    maps = {}
    for lin, lout, stride, pad, _ci, _co in LAYERS:
        key = (lin, lout)
        if key in maps:
            continue
        ind_in, sh_in = levels[lin]
        pad3 = [pad] * 3 if isinstance(pad, int) else list(pad)
        subm = lin == lout
        outids, pairs, num = ops.get_indice_pairs(ind_in, 1, sh_in, ConvAlgo.Native, [3, 3, 3], [stride] * 3,
                                                   pad3, [1, 1, 1], [0, 0, 0], subm, False)
        if not subm:
            sh_out = ops.get_conv_output_size(sh_in, [3, 3, 3], [stride] * 3, pad3, [1, 1, 1])
            levels.append((outids, sh_out))
        n_out = (ind_in if subm else outids).shape[0]
        num = num.tolist()
        rows = sum(-(-c // ALIGN) * ALIGN for c in num)
        g = torch.zeros(rows, dtype=torch.int64, device="cuda")
        s = torch.full((rows,), n_out, dtype=torch.int64, device="cuda")  # padding rows -> dump row
        lay = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
        at = 0
        for k, c in enumerate(num):
            g[at:at + c] = pairs[0, k, :c]
            s[at:at + c] = pairs[1, k, :c]
            lay[at:at + c] = k
            at += -(-c // ALIGN) * ALIGN
        maps[key] = (g, s, lay, n_out)
    return [levels[i][0].shape[0] for i in range(4)], maps


def load():
    from spconv.pytorch.utils import PointToVoxel

    files = sorted(glob.glob(os.path.join(os.environ.get("DG_DATA", "/workspace/_deps/data"), "kitti/**/velodyne_points/data/*.bin"), recursive=True))
    gen = PointToVoxel(vsize_xyz=[0.05, 0.05, 0.1], coors_range_xyz=[0, -40, -3, 70.4, 40, 1], num_point_features=4,
                       max_num_voxels=200000, max_num_points_per_voxel=5, device=torch.device("cuda"))
    shape = [41, 1600, 1408]  # z, y, x (SECOND: grid[::-1] + [1, 0, 0])
    out = []
    for f in files[: a.frames]:
        pts = torch.from_numpy(np.fromfile(f, dtype=np.float32).reshape(-1, 4)).cuda()
        vox, coords, npts = gen(pts)
        feat = vox.sum(1) / npts.clamp_min(1).unsqueeze(1).float()
        x = torch.zeros(feat.shape[0], W, device="cuda", dtype=torch.bfloat16)
        x[:, :4] = feat.to(torch.bfloat16)
        counts, maps = kernel_maps(coords, shape)
        flat = [x]
        for key in [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]:
            g, s, lay, n_out = maps[key]
            flat += [g, s, lay, n_out]
        out.append(tuple(flat))
    return out


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        ws, bns = [], []
        for _li, _lo, _s, _p, ci, co in LAYERS:
            w = torch.zeros(27, W, W)
            w[:, :co, :ci] = torch.randn(27, co, ci) / (27 * ci) ** 0.5
            ws.append(torch.nn.Parameter(w.to(torch.bfloat16), requires_grad=False))
            bn = torch.nn.BatchNorm1d(W, eps=1e-3)
            bn.running_mean.uniform_(-0.1, 0.1)
            bn.running_var.uniform_(0.5, 1.5)
            bn.weight.data[co:] = 0
            bn.bias.data[co:] = 0
            bns.append(bn)
        self.w = torch.nn.ParameterList(ws)
        self.bn = torch.nn.ModuleList(bns)

    def forward(self, x, *maps):
        m = {}
        for i, key in enumerate([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]):
            m[key] = maps[4 * i: 4 * i + 4]
        for j, (lin, lout, _s, _p, _ci, _co) in enumerate(LAYERS):
            g, s, lay, n_out = m[(lin, lout)]
            rows = x[g]
            d = torch.empty(rows.shape[0], W, device=x.device, dtype=x.dtype)
            torch.ops.dgemm.grouped_mm_out(rows, self.w[j], lay, d)
            y = x.new_zeros(n_out + 1, W).index_add_(0, s, d)[:n_out]
            x = torch.relu(self.bn[j](y.float())).to(x.dtype)
        return x


def pad(batches):
    """Every size to its max over the frames: extra voxels are zero rows, extra kernel-map rows gather row 0
    with layout -1 and scatter into the dump row."""
    n = len(batches[0])
    mx = [max((b[i].shape[0] if torch.is_tensor(b[i]) else b[i]) for b in batches) for i in range(n)]
    for i in range(1, n, 4):  # rows of one map, padded to the alignment already
        mx[i] = mx[i + 1] = mx[i + 2] = max(mx[i], mx[i + 1], mx[i + 2])
    out = []
    for b in batches:
        x = b[0].new_zeros(mx[0], W)
        x[: b[0].shape[0]] = b[0]
        flat = [x]
        for i in range(1, n, 4):
            g, s, lay, n_out = b[i: i + 4]
            r, no = mx[i], mx[i + 3]
            gp = g.new_zeros(r)
            gp[: g.shape[0]] = g
            sp = s.new_full((r,), no)
            sp[: s.shape[0]] = torch.where(s == n_out, no, s)
            lp = lay.new_full((r,), -1)
            lp[: lay.shape[0]] = lay
            flat += [gp, sp, lp, no]
        out.append(tuple(flat))
    return out


def main():
    batches = load()
    lv = [(b[0].shape[0], b[8], b[16], b[24]) for b in batches]
    print(f"{len(batches)} frames, active voxels per level " + ", ".join(
        f"L{i} {min(v[i] for v in lv)}..{max(v[i] for v in lv)} (max/mean {max(v[i] for v in lv) / np.mean([v[i] for v in lv]):.2f})"
        for i in range(4)) + f", {len(set(lv))} distinct 4-level tuples")

    def make():
        m = Backbone().cuda().eval()

        def step(f, b):
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                y = f(*b)
                return y.float().abs().mean()

        return m, step

    harness.run(make, batches, a.modes.split(","), warm=a.warm, label="pointcloud", pad=pad)


if __name__ == "__main__":
    main()
