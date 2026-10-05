#!/bin/bash
# dynamic ablation: separate "symbolic materialization fixable upstream" from "structural CPU computation".
# Ordered from fastest to slowest compile, to get a conclusion early. Each model gets its own timeout so the slowest one cannot stall the rest.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../setup/use_main.sh"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=5 DYNAGRAPH_COMPILE_THREADS=4
for spec in timm:botnet26t_256 tv:mobilenet_v2 timm:swin_base_patch4_window7_224 \
            tvdet:ssd300_vgg16 timm:cspdarknet53 timm:nfnet_l0; do
    echo "########## $spec"
    timeout 900 python _dyn_ablation.py "$spec" 2>&1 \
        | grep -E "^  dynamic|^  =>" || echo "  (timed out or failed)"
done
