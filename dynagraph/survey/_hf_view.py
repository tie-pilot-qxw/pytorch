import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
pre = sys.argv[2] if len(sys.argv) > 2 else "hf:"
sub = [r for r in rows if r["model"].startswith(pre)]
ok = [r for r in sub if r.get("ok")]
split = [r for r in ok if (r.get("n_partitions_max") or 1) > 1]
print(f"{pre}  tried {len(sub)}, ok {len(ok)}, split {len(split)}")
for r in split:
    print(f"  {r['model']:<46} partitions={r.get('n_partitions_max')} "
          f"{r.get('partition_reason_counts')}")
