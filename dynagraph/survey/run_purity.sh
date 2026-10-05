#!/bin/bash
# Purity check on a few representative models: is the CPU compute chain independent of the GPU?
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../setup/use_main.sh"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=5 DYNAGRAPH_COMPILE_THREADS=2
for spec in extra:nms extra:sparse extra:md extra:varlen tv:mobilenet_v2; do
    echo "########## $spec"
    timeout 900 python _purity2.py "$spec" 2>&1 | grep -E "^  (CPU|these|->)|^\(run phase"
done
