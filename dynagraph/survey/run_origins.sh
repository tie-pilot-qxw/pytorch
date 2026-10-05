#!/bin/bash
# Find where the cpu_ops of a few non-MobileNet models come from: are they other instances of the ReLU6 bug?
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../setup/use_main.sh"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=5 DYNAGRAPH_COMPILE_THREADS=2
for spec in timm:nfnet_l0 timm:swin_base_patch4_window7_224 timm:botnet26t_256 timm:cspdarknet53; do
    echo "########## $spec"
    timeout 1200 python _cpuops_probe.py "$spec" 2>&1 | grep -E "^tv|^timm|^by (origin|ir_type|device)|^ +[0-9]+ " | head -14
done
