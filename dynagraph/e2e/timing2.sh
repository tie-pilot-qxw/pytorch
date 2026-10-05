#!/bin/bash
# $1 = card, $2 = which (gnn|mace)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../setup/use_main.sh"
export PYTHONPATH=${DG_DEPS:-/workspace/_deps}/deepgemm-src:${DG_DEPS:-/workspace/_deps}/mace_site:$PYTHONPATH CUDA_VISIBLE_DEVICES=$1
cd "$HERE"
mkdir -p "${DG_OUT:-/tmp/dynagraph_out}"
out="${DG_OUT:-/tmp/dynagraph_out}/_timing2_$2_$(date +%m%d_%H%M).log"
st () { echo "== $(date +%T) card $1: $(nvidia-smi -i $1 --query-gpu=utilization.gpu,memory.used --format=csv,noheader) procs $(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | wc -l)"; }
{
st $1
if [ $2 = gnn ]; then
  for order in eager,compile,trees,dg,pad pad,dg,trees,compile,eager; do
    AMP=1 GEMM=deepgemm python sage.py --batches 100 --warm 20 --modes $order 2>&1 | grep -E '^\[|fallback|^[0-9]+ batch'; st $1
    AMP=1 GEMM=deepgemm python schnet.py --batches 80 --warm 20 --modes $order 2>&1 | grep -E '^\[|fallback|^[0-9]+ batch'; st $1
  done
else
  for order in eager,compile,trees,dg dg,trees,compile,eager; do
    GEMM=triton python mace_md.py --npz "${DG_DATA:-/workspace/_deps/data}/MD22/md22_double-walled_nanotube.npz" --frames 100 --warm 20 --modes $order 2>&1 | grep -E '^\[|fallback| frames,'; st $1
  done
fi
} > "$out" 2>&1
echo "$out"
