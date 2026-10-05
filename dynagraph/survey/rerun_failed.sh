#!/bin/bash
# Rerun with real tensors the models that crashed (NoResult) in fake mode.
# Required: the fake-mode segfaults hit precisely the models that have CPU nodes; skipping the rerun is selection bias.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${DG_OUT:-/tmp/dynagraph_out}"; mkdir -p "$OUT"
source "$HERE/../setup/use_main.sh"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=5

python - <<'PY'
import json, os
out = os.environ.get("DG_OUT", "/tmp/dynagraph_out")
bad = [json.loads(l)["model"] for l in open(os.path.join(out, "results_capture.jsonl"))
       if json.loads(l).get("error_type") in ("NoResult", "Timeout")]
open(os.path.join(out, "rerun.txt"), "w").write("\n".join(bad) + "\n")
print(f"{len(bad)} to rerun -> {os.path.join(out, 'rerun.txt')}")
PY

# real tensors, concurrency down to 2, leave GPU memory headroom for others
python runner.py --list "$OUT/rerun.txt" --out "$OUT/results_rerun_real.jsonl" \
    --jobs 2 --timeout 1200
echo "rerun finished $(date)"
