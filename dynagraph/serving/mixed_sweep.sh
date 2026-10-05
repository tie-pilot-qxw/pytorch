#!/bin/bash
# Run the four cudagraph modes interleaved for N rounds (host load is the operating condition; only interleaved runs are comparable).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../setup/use_main.sh"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=${GPU:-4}
for i in $(seq 1 ${N:-2}); do
  for cg in NONE PIECEWISE FULL_AND_PIECEWISE FULL; do
    CG=$cg MODEL=${MODEL:-Qwen/Qwen3-0.6B} MBT=${MBT:-2048} NREQ=${NREQ:-256} GPU_UTIL=${GPU_UTIL:-0.5} STEPS=${STEPS:-0} \
      timeout 3000 python -W ignore probe_vllm_mixed.py 2>&1 | grep -aE '^\[mixed|^   steps |   token|Graph capturing finished|Traceback|Error' | cut -c1-260
  done
done
