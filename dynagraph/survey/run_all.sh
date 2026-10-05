#!/bin/bash
# Two full survey runs, back to back.
#   main group     capture=True  data dependence reaches Inductor; shows how many segments and why
#   control group  capture=False PyTorch's shipped config; data dependence causes Dynamo graph breaks
# Looking at either group alone leads to the wrong conclusion; see README.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${DG_OUT:-/tmp/dynagraph_out}"; mkdir -p "$OUT"
source "$HERE/../setup/use_main.sh"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=5

echo "########## main group (capture=True) start $(date)"
python runner.py --list models_all.txt --out "$OUT/results_capture.jsonl" \
    --jobs 6 --fake --timeout 900
echo "########## main group done $(date)  $(wc -l < "$OUT/results_capture.jsonl") rows"

echo "########## control group (PyTorch default) start $(date)"
python runner.py --list models_all.txt --out "$OUT/results_default.jsonl" \
    --jobs 6 --fake --no-capture --timeout 900
echo "########## control group done $(date)  $(wc -l < "$OUT/results_default.jsonl") rows"

echo "########## summary"
python report.py "$OUT/results_capture.jsonl"
echo
python compare.py "$OUT/results_default.jsonl" "$OUT/results_capture.jsonl"
