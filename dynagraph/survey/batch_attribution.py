#!/usr/bin/env python3
"""
Batch attribution: run every model that has `cpu_ops` twice, and split out "one-off cases fixable upstream" from "structural CPU computation".

  python batch_attribution.py results_real_capture.jsonl > attribution.txt

Why this is required: the `nn.ReLU6` upstream bug, which a three-line change fixes, alone contributes 71
`cpu_ops` nodes to MobileNet-V2 (zero after removing it). If a chart reports these mixed in with the kind that
cannot be removed (as in detection models), a single "why don't you just fix that bug?" defeats the argument.

Method: replace `nn.ReLU6` / `nn.Hardtanh(0.0, 6.0)` in the model with an equivalent module whose bounds are literals,
leave everything else untouched, rerun, and take the difference. `origins` alone cannot tell them apart, because the
nodes ReLU6 produces have origin `aten.cat` / `aten.unsqueeze` (the result of symbolic materialization), which collides with other sources.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CHILD = r'''
import sys, os
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
import torch._inductor.config as ic, torch._dynamo.config as dc
import torch._inductor.scheduler as S
sys.path.insert(0, %(here)r)
import models

ic.force_disable_caches = True
dc.capture_dynamic_output_shape_ops = True
dc.capture_scalar_outputs = True

seen = {}
orig = S.Scheduler.should_partition
def patched(self, node, *a, **kw):
    out = orig(self, node, *a, **kw)
    if isinstance(out, str):
        try: n = node.get_name()
        except Exception: n = str(id(node))
        seen.setdefault(n, out)
    return out
S.Scheduler.should_partition = patched

class FixedReLU6(nn.Module):
    def forward(self, x):
        return F.hardtanh(x, 0.0, 6.0)

def swap(m):
    n = 0
    for name, child in list(m.named_children()):
        if isinstance(child, nn.ReLU6) or (
            type(child) is nn.Hardtanh
            and float(child.min_val) == 0.0 and float(child.max_val) == 6.0):
            setattr(m, name, FixedReLU6()); n += 1
        else:
            n += swap(child)
    return n

spec, mode = sys.argv[1], sys.argv[2]
with models.device_ctx():
    m, a, k = models.build(spec)
n_swapped = swap(m) if mode == "fixed" else 0
try:
    torch.compile(m, dynamic=True, mode="reduce-overhead")(*a, **k)
except Exception:
    pass
c = Counter(seen.values())
print("__RESULT__ " + __import__("json").dumps(
    {"n": len(seen), "counts": dict(c), "swapped": n_swapped}))
''' % {"here": HERE}


def run(spec, mode):
    p = subprocess.run([sys.executable, "-c", CHILD, spec, mode],
                       capture_output=True, text=True, timeout=1800, cwd=HERE,
                       env={**os.environ, "TORCHINDUCTOR_COMPILE_THREADS": "8"})
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("__RESULT__ "):
            return json.loads(line[len("__RESULT__ "):])
    return None


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    targets = [r["model"] for r in rows
               if r.get("ok") and (r.get("partition_reason_counts") or {}).get("cpu_ops")]
    print(f"{len(targets)} models have cpu_ops; running each twice\n")
    print(f"{'model':<42}{'original':>8}{'swap ReLU6':>12}{'residual':>10}{'swapped':>10}")
    print("-" * 82)
    tot_orig = tot_fixed = 0
    for spec in targets:
        a, b = run(spec, "orig"), run(spec, "fixed")
        if not a or not b:
            print(f"{spec:<42}{'(failed)':>8}")
            continue
        tot_orig += a["n"]
        tot_fixed += b["n"]
        print(f"{spec:<42}{a['n']:>8}{b['n']:>12}{b['n']:>10}{b['swapped']:>10}")
    print("-" * 82)
    print(f"{'total':<42}{tot_orig:>8}{tot_fixed:>12}{tot_fixed:>10}")
    if tot_orig:
        share = 100.0 * (tot_orig - tot_fixed) / tot_orig
        print(f"\n{share:.0f}% of the non-GPU nodes come from the nn.ReLU6 upstream bug; "
              f"the remaining {100 - share:.0f}% are structural.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_real_capture.jsonl")
