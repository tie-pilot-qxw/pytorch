import json, sys
from collections import Counter
rows = [json.loads(l) for l in open(sys.argv[1])]
ok = [r for r in rows if r.get("ok")]
print("max partition count distribution:", dict(sorted(Counter(r.get("n_partitions_max") or 1 for r in ok).items())))
print("segments per graph distribution:", dict(sorted(Counter(
    n for r in ok for n in (r.get("n_partitions_observed") or [])).items())))
sk = [r for r in ok if r.get("n_partitions_skipping_cudagraph")]
print(f"models with a segment that ended up skipping cudagraph: {len(sk)}/{len(ok)}")
for r in sk[:8]:
    print(f"   {r['model']:<44} {r.get('n_partitions_skipping_cudagraph')} segments skipped")
