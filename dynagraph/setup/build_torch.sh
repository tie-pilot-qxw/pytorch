#!/bin/bash
# Editable install of pytorch-main from source into the container venv
# Without the venv, pip would replace the container's NVIDIA torch, which cannot be reinstalled.
source /workspace/.venv/bin/activate || { echo "no venv at /workspace/.venv (see dynagraph/README.md)"; exit 1; }
cd /workspace/pytorch-main

# The NVIDIA container presets PYTORCH_BUILD_VERSION=2.11.0a0, which overrides version.txt (2.15.0a0),
# so the whole "#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0" block in shim.h is excluded,
# while shim_common.cpp uses symbols from inside that block unconditionally -> torch_cpu fails to compile. Must be unset.
unset PYTORCH_BUILD_VERSION PYTORCH_VERSION PYTORCH_BUILD_NUMBER
git config --global --add safe.directory /workspace/pytorch-main 2>/dev/null
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="9.0a"     # H100 only (including sm_90a: wgmma/TMA)
export USE_CUDA=1 USE_CUDNN=1 USE_NCCL=1 USE_SYSTEM_NCCL=1
export USE_ROCM=0 USE_XPU=0
export BUILD_TEST=0                     # skip the C++ unit tests
export USE_FLASH_ATTENTION=1 USE_MEM_EFF_ATTENTION=1
export CMAKE_BUILD_TYPE=Release
export MAX_JOBS=64
export USE_NUMA=0
echo "=== start $(date) ==="
# Launch this script with docker exec -d; otherwise ninja gets a signal and aborts as soon as the host-side session drops
pip install -e . -v --no-build-isolation 2>&1
rc=$?
echo "=== end $(date) exit=$rc ==="
exit $rc
