#!/usr/bin/env python3
"""Construct every model in the list once on the CPU to see which ones cannot even be built. Does not touch the GPU."""
import sys, os, traceback
os.environ.setdefault("DYNAGRAPH_DEVICE", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import models

specs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
ok, bad = [], []
for i, sp in enumerate(specs, 1):
    try:
        m, args, kw = models.build(sp)
        n = sum(p.numel() for p in m.parameters())
        ok.append((sp, n))
        print(f"[{i}/{len(specs)}] ok   {sp}  ({n/1e6:.1f}M params)", flush=True)
    except Exception as e:
        bad.append((sp, type(e).__name__, str(e)[:160]))
        print(f"[{i}/{len(specs)}] FAIL {sp}  {type(e).__name__}: {str(e)[:120]}", flush=True)

import json
with open("model_sizes.json", "w") as f:
    json.dump({sp: n for sp, n in ok}, f, indent=1)
print(f"\nbuildable {len(ok)}/{len(specs)}, failed {len(bad)}  (parameter counts written to model_sizes.json)")
if bad:
    print("\nfailed list:")
    for sp, t, msg in bad:
        print(f"  {sp:<45} {t}: {msg[:100]}")
    with open("failed_models.txt", "w") as f:
        f.write("\n".join(sp for sp, _, _ in bad) + "\n")
if ok:
    big = sorted(ok, key=lambda x: -x[1])[:5]
    print("\nlargest 5: " + ", ".join(f"{s}({n/1e6:.0f}M)" for s, n in big))
