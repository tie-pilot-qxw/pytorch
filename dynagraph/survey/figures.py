#!/usr/bin/env python3
"""
Generate two figures, corresponding to Figure 1 and Figure 2 in the paper.

  python figures.py results_capture.jsonl [results_default.jsonl]

Figure 1  Per suite, the distribution of models by number of cudagraph partitions.
          Corresponds to the graph-break distribution rows of PyTorch 2 (ASPLOS'24) Table 1.
Figure 2  Which rule causes the graph split. The x axis is the number of affected models.

Labels are in English so they can go straight into the paper.
"""
from __future__ import annotations
import json, os, sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUITE_LABEL = {
    "tv": "torchvision\n(classification)",
    "tvdet": "torchvision\n(detection/seg)",
    "timm": "TIMM",
    "hf": "HuggingFace",
}
BUCKETS = ["1 (whole graph)", "2-9", ">=10", "failed to compile"]
COLORS = ["#2b8a3e", "#f59f00", "#e03131", "#868e96"]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def bucket(r):
    if not r.get("ok"):
        return BUCKETS[3]
    n = r.get("n_partitions_max") or 1
    if n <= 1:
        return BUCKETS[0]
    return BUCKETS[1] if n <= 9 else BUCKETS[2]


def fig1(rows, out):
    by = defaultdict(Counter)
    for r in rows:
        suite = r["model"].split(":", 1)[0]
        if suite == "builtin":
            continue
        by[suite][bucket(r)] += 1
    suites = [s for s in ("tv", "tvdet", "timm", "hf") if s in by]
    if not suites:
        print("  (no suites to plot)"); return

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bottom = [0.0] * len(suites)
    totals = [sum(by[s].values()) for s in suites]
    for b, c in zip(BUCKETS, COLORS):
        vals = [100.0 * by[s][b] / totals[i] for i, s in enumerate(suites)]
        ax.bar(range(len(suites)), vals, 0.6, bottom=bottom, label=b, color=c)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(i, bottom[i] + v / 2, f"{by[suites[i]][b]}",
                        ha="center", va="center", fontsize=9,
                        color="white" if c != "#f59f00" else "black")
        bottom = [bottom[i] + vals[i] for i in range(len(suites))]

    ax.set_xticks(range(len(suites)))
    ax.set_xticklabels([f"{SUITE_LABEL.get(s, s)}\n(n={totals[i]})"
                        for i, s in enumerate(suites)], fontsize=9)
    ax.set_ylabel("% of models")
    ax.set_ylim(0, 100)
    ax.set_title("CUDA Graph partitions per model", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"  Figure 1 -> {out}")


def fig2(rows, out):
    pm, pn = Counter(), Counter()
    for r in rows:
        if not r.get("ok"):
            continue
        for slug, n in (r.get("partition_reason_counts") or {}).items():
            pm[slug] += 1
            pn[slug] += n
    if not pm:
        print("  (no model was split, skipping Figure 2 -- that in itself is a result)")
        return
    items = pm.most_common()
    fig, ax = plt.subplots(figsize=(7.2, max(2.2, 0.42 * len(items) + 1.0)))
    ys = range(len(items))
    ax.barh(list(ys), [n for _, n in items], color="#1971c2")
    for i, (slug, n) in enumerate(items):
        ax.text(n, i, f"  {n} models / {pn[slug]} nodes", va="center", fontsize=8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([s for s, _ in items], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("# models affected")
    ax.set_title("Why the graph gets partitioned", fontsize=11)
    ax.set_xlim(0, max(n for _, n in items) * 1.45)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"  Figure 2 -> {out}")


if __name__ == "__main__":
    rows = load(sys.argv[1])
    print(f"loaded {len(rows)} rows")
    out_dir = os.environ.get("DG_OUT", "/tmp/dynagraph_out")
    os.makedirs(out_dir, exist_ok=True)
    fig1(rows, os.path.join(out_dir, "fig1_partitions.pdf"))
    fig2(rows, os.path.join(out_dir, "fig2_reasons.pdf"))
