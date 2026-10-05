#!/bin/bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../setup/use_main.sh"
export PYTHONPATH="${DG_DEPS:-/workspace/_deps}/mace_site:$PYTHONPATH" CUDA_VISIBLE_DEVICES=4
cd "$HERE"
mkdir -p "${DG_OUT:-/tmp/dynagraph_out}"
out="${DG_OUT:-/tmp/dynagraph_out}/_timing_$(date +%m%d_%H%M).log"
{
echo "== start $(date) util: $(nvidia-smi -i 4 --query-gpu=utilization.gpu,power.draw --format=csv,noheader)"
AMP=1 GEMM=deepgemm python sage.py --batches 80 --warm 20 --modes eager,compile,trees,dg,pad 2>&1 | grep -E '^\[|fallback|^[0-9]+ batches,'
echo "== util: $(nvidia-smi -i 4 --query-gpu=utilization.gpu --format=csv,noheader)"
AMP=1 GEMM=deepgemm python schnet.py --batches 60 --warm 20 --modes eager,compile,trees,dg,pad 2>&1 | grep -E '^\[|fallback|^[0-9]+ batches,'
echo "== util: $(nvidia-smi -i 4 --query-gpu=utilization.gpu --format=csv,noheader)"
GEMM=triton python mace_md.py --npz "${DG_DATA:-/workspace/_deps/data}/MD22/md22_double-walled_nanotube.npz" --frames 80 --warm 20 --modes eager,compile,trees,dg 2>&1 | grep -E '^\[|fallback| frames,'
echo "== end $(date) util: $(nvidia-smi -i 4 --query-gpu=utilization.gpu --format=csv,noheader)"
} > "$out" 2>&1
echo "$out"
