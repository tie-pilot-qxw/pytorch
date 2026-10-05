#!/usr/bin/env python3
"""
Bucketing cost curve -- the statistic that decides whether a workload supports this project's approach.

Why "how many distinct shapes" is the wrong statistic
-----------------------------------------------------
`results.log` (a log not included in this repo) measured KITTI at 433 shapes over 433 frames and GNN at 200 shapes over 200 steps. The numbers are real,
but they **cannot** justify the approach, because the real baseline is not "record one graph per shape" but
**K=1 = pad-to-max**: pad every input to the global maximum and run it all with one static graph.
pad-to-max needs neither the host to know the sizes nor any synchronization; `dynamic=False` does it
(`docs/notes/FEASIBILITY.md`, section "What was overturned, by severity"). So:

    If pad-to-max costs only 1.03x, the number of distinct shapes does not matter, and this workload is not a use case for this project.

The right statistic is the cost curve:

    cost(K) = E[pad_K(s)^p] / E[s^p]

s is the dimension that drives the shape, p is the kernel's real cost exponent
(1 for jagged pooling / scatter, 2 for attention and pair representation,
3 for triangle-type ops). The criterion requires **both**:

    cost(1) unacceptably expensive  and  cost(8) still unacceptable

Only then is "one capture covering the whole range" worth more than "pad to max" or "a few buckets".

Usage
-----
    python bucket_cost.py                 # run every workload whose data can be found
    python bucket_cost.py --only prot
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The data used to live in a session scratchpad (which gets cleared); it now lives in $DG_DATA.
# If it is missing, fetch it again with dl.sh / dl2.sh.
WORK = os.environ.get("DYNAGRAPH_WORKDATA", "")
if not WORK:
    for cand in (HERE, os.path.join(HERE, "work")):
        if os.path.exists(os.path.join(cand, "human_proteome.fasta.gz")):
            WORK = cand
            break
    else:
        WORK = os.environ.get("DG_DATA", "/workspace/_deps/data")

KS = (1, 2, 4, 8, 16, 32, 64)


def cost_curve(sizes, p, ks=KS):
    """Return (K, cost, bucket edges) for each K.

    Bucket edges are placed at **work-weighted quantiles**: this is how real systems tune them
    (put the buckets where the cost is concentrated), and it is **conservative** for this project --
    it makes bucketing look as good as possible, so only if bucketing is still expensive with this placement is the conclusion solid.
    The top bucket always equals the global maximum, so **no sample is ever dropped**
    (the original prot.py:24 `if b is None: continue` dropped the long sequences right here,
    and those are exactly the sequences that dominate the L^3 cost).
    """
    s = np.asarray(sizes, dtype=np.float64)
    s = s[s > 0]
    w = s ** p                      # real work per sample
    denom = w.mean()
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    cw = np.cumsum(w_sorted)
    cw /= cw[-1]

    out = []
    for k in ks:
        if k <= 1:
            edges = np.array([s_sorted[-1]])
        else:
            # take edges at k quantiles of cumulative work, dedup, then append the global max
            qs = np.linspace(0.0, 1.0, k + 1)[1:]
            idx = np.searchsorted(cw, qs)
            idx = np.clip(idx, 0, len(s_sorted) - 1)
            edges = np.unique(s_sorted[idx])
            if edges[-1] < s_sorted[-1]:
                edges = np.append(edges, s_sorted[-1])
        # pad each sample to the smallest bucket >= it; it always exists since the top bucket is the global max
        pos = np.searchsorted(edges, s_sorted, side="left")
        padded = edges[np.clip(pos, 0, len(edges) - 1)]
        out.append((k, float((padded ** p).mean() / denom), len(edges)))
    return out


def report(name, sizes, p, note=""):
    sizes = np.asarray(sizes, dtype=np.float64)
    sizes = sizes[sizes > 0]
    if sizes.size == 0:
        print(f"  {name}: no data, skipping"); return
    curve = cost_curve(sizes, p)
    c1 = curve[0][1]
    c8 = next(c for k, c, _ in curve if k == 8)
    verdict = ("**padding expensive at K=1 and K=8**" if (c1 >= 3.0 and c8 >= 2.0)
               else "pad-to-max cheap (cost(1) < 1.5x)" if c1 < 1.5
               else "marginal")
    print(f"\n  {name}   p={p}   n={sizes.size}   {note}")
    print(f"    sizes min={sizes.min():.0f} p50={np.percentile(sizes,50):.0f} "
          f"p99={np.percentile(sizes,99):.0f} max={sizes.max():.0f} "
          f"distinct={len(np.unique(sizes))}")
    print("    " + "  ".join(f"K={k}:{c:.2f}x" for k, c, _ in curve))
    print(f"    -> cost(1)={c1:.2f}x  cost(8)={c8:.2f}x   {verdict}")
    return c1, c8


# ------------------------------------------------------------------ workloads
def load_prot():
    path = os.path.join(WORK, "human_proteome.fasta.gz")
    if not os.path.exists(path):
        return None
    L, cur = [], 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    L.append(cur)
                cur = 0
            else:
                cur += len(line.strip())
    if cur:
        L.append(cur)
    return np.array(L)


def load_kitti():
    import glob
    files = sorted(glob.glob(os.path.join(WORK, "kitti", "**", "*.bin"), recursive=True))
    if not files:
        return None
    rng = ((0, 70.4), (-40, 40), (-3, 1))
    vs = (0.05, 0.05, 0.1)
    out = []
    for fp in files:
        pts = np.fromfile(fp, dtype=np.float32).reshape(-1, 4)[:, :3]
        m = np.ones(len(pts), bool)
        for i, (lo, hi) in enumerate(rng):
            m &= (pts[:, i] >= lo) & (pts[:, i] < hi)
        pts = pts[m]
        k = np.floor((pts - [r[0] for r in rng]) / vs).astype(np.int64)
        out.append(len(np.unique(k[:, 0] * 10**8 + k[:, 1] * 10**4 + k[:, 2])))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    print(f"data dir: {WORK}")
    print("criterion: cost(1) must be expensive and cost(8) still expensive. Only one being expensive does not count.\n")
    print("=" * 78)

    jobs = []
    if not a.only or a.only == "prot":
        p = load_prot()
        if p is not None:
            # AlphaFold: pair representation is L^2, triangle ops are L^3
            jobs.append(("protein structure prediction (UniProt human proteome) -- pair repr", p, 2,
                         "AF2/AF3 pair representation"))
            jobs.append(("protein structure prediction -- triangle ops", p, 3,
                         "the most expensive part of AF2/AF3"))
    if not a.only or a.only == "kitti":
        k = load_kitti()
        if k is not None:
            # sparse conv is linear in the active voxel count
            jobs.append(("KITTI sparse conv stride-1 active voxels", k, 1,
                         "control: 433/433 distinct shapes but linear cost"))

    for name, sizes, p, note in jobs:
        report(name, sizes, p, note)

    print("\n" + "=" * 78)
    print("""
How to read this
----------------
cost(1) is the extra factor paid for "pad every input to the global max and run it all with one static graph".
A workload where it is below 1.5x is not a use case for this project, however many shapes it has --
because pad-to-max needs no graph surgery; `dynamic=False` does it.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
