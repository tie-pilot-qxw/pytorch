#!/usr/bin/env python3
"""Quick per-row view of a jsonl results file."""
import json, sys
for l in open(sys.argv[1]):
    d = json.loads(l)
    st = "ok" if d["ok"] else (d.get("error_type") or "?")
    print(f"{d['model']:<34} {st:<30} partitions={d.get('n_partitions_observed')} "
          f"reasons={d.get('partition_reason_counts')} {d.get('wall_s')}s")
