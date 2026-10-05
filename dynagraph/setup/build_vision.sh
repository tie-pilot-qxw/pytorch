#!/bin/bash
# Build torchvision main against our self-built torch 2.15.
# The container's own torchvision 0.25 was built for torch 2.11 and its ABI does not match;
# the symptom is "RuntimeError: operator torchvision::nms does not exist",
# and transformers' loss_utils hard-depends on it, so every HF model import fails along with it.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/use_main.sh" || exit 1
unset TORCHVISION_BUILD_VERSION TORCHVISION_VERSION
cd /workspace/torchvision-main
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="9.0a"
export MAX_JOBS=64
echo "=== start $(date) ==="
python -c "import torch;print('built against torch:', torch.__version__)"
pip install -e . -v --no-build-isolation --no-deps 2>&1
rc=$?
echo "=== end $(date) exit=$rc ==="
exit $rc
