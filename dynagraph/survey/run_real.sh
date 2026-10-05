#!/bin/bash
# Full survey with real tensors.
#
# Why not fake: fake mode underestimates in two systematic ways
#   1. Crashes have selection bias -- it crashes specifically on models with CPU nodes (DeviceCopy triggers a segfault)
#   2. Even when it does not crash it silently truncates -- multi-subgraph models only count the first subgraph
#      (ssd300_vgg16: fake sees 1 subgraph, real tensors 9; cpu_ops 91 vs 98)
# See "Methodology alert" in ../docs/notes/FEASIBILITY.md.
#
# GPU memory: SSD300 measured single-process peak 5252 MiB. --jobs 3 is ~18 GB.
# Split off models over the LIMIT parameter count to run separately first, otherwise it OOMs.
#
# Lists, results and logs go under $DG_OUT. Redirect this script's stdout to $DG_OUT/run_real.log:
# finalize.sh polls that file for the phase markers echoed below.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${DG_OUT:-/tmp/dynagraph_out}"; mkdir -p "$OUT"
source "$HERE/../setup/use_main.sh"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=5

LIMIT=${LIMIT:-1e9}
# specs contain colons (hf:GPTJForCausalLM), so they cannot be scraped from the log with a regex; have the script write the file directly
python filter_by_size.py models_all.txt "$LIMIT" "$OUT/models_big.txt" > "$OUT/models_small.txt" 2> "$OUT/models_big.log"
echo "small models $(wc -l < "$OUT/models_small.txt"), big $(wc -l < "$OUT/models_big.txt" 2>/dev/null || echo 0)"

echo "########## main group (capture=True, real tensors) start $(date)"
python runner.py --list "$OUT/models_small.txt" --out "$OUT/results_real_capture.jsonl" \
    --jobs 3 --timeout 1200
echo "########## main group done $(date)  $(wc -l < "$OUT/results_real_capture.jsonl") rows"

echo "########## baseline group (PyTorch default) start $(date)"
python runner.py --list "$OUT/models_small.txt" --out "$OUT/results_real_default.jsonl" \
    --jobs 3 --no-capture --timeout 1200
echo "########## baseline group done $(date)  $(wc -l < "$OUT/results_real_default.jsonl") rows"

if [ -s "$OUT/models_big.txt" ]; then
    echo "########## big models run separately (jobs=1) $(date)"
    python runner.py --list "$OUT/models_big.txt" --out "$OUT/results_real_big.jsonl" \
        --jobs 1 --timeout 1800
fi

echo "########## summary"
cat "$OUT/results_real_capture.jsonl" "$OUT/results_real_big.jsonl" 2>/dev/null > "$OUT/results_real_all.jsonl"
python report.py "$OUT/results_real_all.jsonl"
echo
python compare.py "$OUT/results_real_default.jsonl" "$OUT/results_real_capture.jsonl"
