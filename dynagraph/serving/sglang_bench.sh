#!/bin/bash
# usage: sglang_bench.sh <card> <tag> <env...>
card=$1; tag=$2; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
outdir=${DG_OUT:-/tmp/dynagraph_out}/bcg; mkdir -p "$outdir"
export PYTHONPATH=${DG_DEPS:-/workspace/_deps}/sglang-main/python CUDA_VISIBLE_DEVICES=$card FLASHINFER_DISABLE_VERSION_CHECK=1
export RES=1024x1024,512x512,768x1344 WARM_RES=1024x1024,512x512 STEPS=20 GUIDANCE=4.5 TORCH_LOGS=graph_breaks,recompiles
env "$@" OUT="$outdir/sgl_$tag.json" timeout 3600 python bcg_bench.py /fake/SANA1.5_1.6B_1024px_diffusers > "$outdir/sgl_$tag.log" 2>&1
echo "rc=$?" >> "$outdir/sgl_$tag.log"
