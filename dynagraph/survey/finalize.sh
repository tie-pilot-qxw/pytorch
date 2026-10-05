#!/bin/bash
# Run this after both full groups have finished; produces the final tables, figures and attribution.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${DG_OUT:-/tmp/dynagraph_out}"; mkdir -p "$OUT"
source "$HERE/../setup/use_main.sh"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=5 DYNAGRAPH_COMPILE_THREADS=8

# Cannot use "runner.py process is gone" as the criterion: there is almost no gap between the main
# and baseline groups, and that instant would be mistaken for everything being done. Use the
# phase markers that run_real.sh prints itself.
# run_real.sh's stdout is expected to be redirected to $DG_OUT/run_real.log.
while ! grep -q "baseline group done" "$OUT/run_real.log" 2>/dev/null; do
    sleep 30
done
# The very-large-model round comes after the baseline group; wait for it too
while pgrep -f "runner.py --list .*models_big" > /dev/null; do sleep 20; done
echo "both groups done $(date)"
echo "  main      $(wc -l < "$OUT/results_real_capture.jsonl" 2>/dev/null || echo 0) rows"
echo "  baseline  $(wc -l < "$OUT/results_real_default.jsonl" 2>/dev/null || echo 0) rows"
echo "  big       $(wc -l < "$OUT/results_real_big.jsonl" 2>/dev/null || echo 0) rows"

cat "$OUT/results_real_capture.jsonl" "$OUT/results_real_big.jsonl" 2>/dev/null > "$OUT/results_real_all.jsonl"

echo; echo "################ main group stats"
python report.py "$OUT/results_real_all.jsonl"

echo; echo "################ two-group comparison"
python compare.py "$OUT/results_real_default.jsonl" "$OUT/results_real_capture.jsonl"

echo; echo "################ figures"
python figures.py "$OUT/results_real_all.jsonl"

echo; echo "################ attribution breakdown (each model with cpu_ops runs twice, takes a while)"
python batch_attribution.py "$OUT/results_real_all.jsonl"
