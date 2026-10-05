#!/bin/bash
# Wait for the main group (capture=True) to finish, then automatically run the control group (PyTorch default config).
# The two groups answer different questions; see "Two more settings to run both ways" in README.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${DG_OUT:-/tmp/dynagraph_out}"; mkdir -p "$OUT"
source "$HERE/../setup/use_main.sh"
cd "$HERE"
while pgrep -f "runner.py --list models_all.txt --out .*results_capture" > /dev/null; do
    sleep 30
done
echo "=== main group finished, $(wc -l < "$OUT/results_capture.jsonl") rows. Starting control group $(date) ==="
CUDA_VISIBLE_DEVICES=5 python runner.py \
    --list models_all.txt --out "$OUT/results_default.jsonl" \
    --jobs 6 --fake --no-capture --timeout 600
echo "=== control group done $(date) ==="
