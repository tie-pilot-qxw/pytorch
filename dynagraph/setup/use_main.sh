# Use the self-built pytorch main. Source this before every invocation.
#   docker exec dynagraph bash -lc 'source /workspace/pytorch-main/dynagraph/setup/use_main.sh && python ...'
#
# Two NVIDIA container presets that must be dealt with:
#   PYTORCH_BUILD_VERSION  overrides version.txt, so the build treats main as 2.11 (see dynagraph/docs/notes/SETUP.md)
#   LD_LIBRARY_PATH        its first entry is the system torch's lib, which makes our freshly built _C.so (a 15KB stub)
#                          link against NVIDIA 2.11's libtorch_python.so; the symptom is
#                          "module 'torch._C' has no attribute '_has_gds'"
source /workspace/.venv/bin/activate || { echo "no venv at /workspace/.venv (see dynagraph/README.md)"; return 1; }
unset PYTORCH_BUILD_VERSION PYTORCH_VERSION PYTORCH_BUILD_NUMBER

# Drop the system torch / torch_tensorrt lib dirs from the search path, then put the self-built one first
LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
    | grep -vE 'dist-packages/(torch|torch_tensorrt)/lib' | paste -sd: -)"
export LD_LIBRARY_PATH="/workspace/pytorch-main/build/lib:${LD_LIBRARY_PATH}"

# torchvision must be the self-built one too (the container's 0.25 was built for 2.11).
# Its editable install uses a PEP 660 MetaPathFinder, and that finder sits after PathFinder
# in sys.meta_path, so it never wins over the old version in dist-packages. Put it first with PYTHONPATH directly.
export PYTHONPATH="/workspace/torchvision-main:${PYTHONPATH}"
