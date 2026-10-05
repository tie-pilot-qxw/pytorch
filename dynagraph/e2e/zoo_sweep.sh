#!/bin/bash
# usage: bash e2e/zoo_sweep.sh <tag> <bs> [models]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../setup/use_main.sh"
cd "$HERE/.."
outdir=${DG_OUT:-/tmp/dynagraph_out}; mkdir -p "$outdir"
tag=$1; bs=$2; models=${3:-"bert distilgpt2 xlnet t5 mobilebert debertav2 blenderbot albert electra deberta"}
for m in $models; do
  AMP=1 TORCHINDUCTOR_COMPILE_THREADS=8 timeout 1200 python e2e/zoo.py --model $m --bs $bs --modes ${MODES:-eager,compile,trees,pad} 2>&1 \
    | grep -E "^\[| steps, |fallback|Error|error" | grep -v -E "cudagraph_utils|pgo.py"
done > "$outdir/_zoo_$tag.log" 2>&1
echo done >> "$outdir/_zoo_$tag.log"
