#!/usr/bin/env python3
"""
Do fake mode and real tensors give the same compile-time conclusions?

This is the root question for the credibility of the whole survey: if fake mode systematically
changes partition results, the Figure 1 produced with it is wrong.

For each model, start two independent subprocesses (fake / non-fake) and compare the key fields.
"""
from __future__ import annotations
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("n_nodes_not_cudagraphable", "partition_reason_counts",
        "n_partitions_observed", "n_partitions_max",
        "n_partitions_skipping_cudagraph", "cudagraph_skips",
        "dynamo_graph_breaks")


def run(spec, fake):
    cmd = [sys.executable, os.path.join(HERE, "runner.py"), "--child", spec]
    if fake:
        cmd.append("--fake")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=HERE)
    except subprocess.TimeoutExpired:
        return {"error_type": "Timeout"}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("__RESULT__ "):
            return json.loads(line[len("__RESULT__ "):])
    return {"error_type": f"NoResult(rc={p.returncode})"}


specs = sys.argv[1:] or ["tv:resnet18", "tv:mobilenet_v2", "tv:vit_b_16",
                         "hf:BertForMaskedLM", "builtin:nonzero"]
print(f"{'model':<26}{'field':<34}{'fake':>22}{'real':>22}")
print("-" * 104)
n_same = n_diff = 0
for spec in specs:
    a, b = run(spec, True), run(spec, False)
    if a.get("error_type") or b.get("error_type"):
        print(f"{spec:<26}{'(one side failed)':<34}"
              f"{str(a.get('error_type') or 'ok'):>22}{str(b.get('error_type') or 'ok'):>22}")
    for k in KEYS:
        va, vb = a.get(k), b.get(k)
        same = va == vb
        n_same += same; n_diff += not same
        if not same:
            print(f"{spec:<26}{k:<34}{str(va):>22}{str(vb):>22}   <<< mismatch")
    print(f"{spec:<26}{'-- fields above':<34}{'':>22}{f'{len(KEYS)} compared':>22}")
print(f"\nmatch {n_same}, mismatch {n_diff}")
