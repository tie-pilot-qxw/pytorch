#!/usr/bin/env python3
"""
Filter a model list by parameter count. With real tensors GPU memory is a hard constraint; fake mode does not need this.

  python filter_by_size.py models_all.txt 1e9 models_big.txt > models_small.txt

If a third argument is given, specs over the limit are written to that file (one per line, clean, no comments).
"""
import json, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sizes = json.load(open(os.path.join(HERE, "model_sizes.json")))
limit = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
big_out = sys.argv[3] if len(sys.argv) > 3 else None
big = []
kept = skipped = 0
for ln in open(sys.argv[1]):
    sp = ln.strip()
    if not sp or sp.startswith("#"):
        continue
    n = sizes.get(sp)
    if n is None or n <= limit:
        print(sp); kept += 1
    else:
        skipped += 1
        big.append(sp)
        print(f"# skip {sp}: {n/1e6:.0f}M params", file=sys.stderr)
if big_out:
    with open(big_out, "w") as f:
        f.write("\n".join(big) + ("\n" if big else ""))
print(f"kept {kept}, skipped {skipped} (limit {limit/1e6:.0f}M params)", file=sys.stderr)
