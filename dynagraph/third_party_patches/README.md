# Third-party patches (optional)

These patches add a **describe** entry point to two libraries. An operator that can describe its
launches can be inlined into DynaGraph's main graph and patched per shape like a Triton kernel
(tier 3; see `docs/notes/REGISTER.md` and `torch/utils/_capture_launch.py`). Without these patches,
the same calls still work: DynaGraph falls back to harvesting them into child graphs (tier 2).

They modify other people's libraries, so they are experiments, not the default path. Apply them to
private copies under `$DG_DEPS` (default `/workspace/_deps`), never to the copies vLLM or SGLang
use.

**If `$DG_DEPS/deepgemm-src` or `$DG_DEPS/fa3d-src` already exists** (a container you were given may
come with them), it is already patched and built against the same branch: skip the `cp` and patch
steps below. The patch scripts refuse to run twice (they assert the source is unpatched), and
`patch` asks "Reversed (or previously applied) patch detected! Assume -R?": answer `n`, because `y`
removes the patch. Rebuild only if your torch comes from a different commit (`./develop.sh` for
DeepGEMM; `rm -rf build` and the cmake and ninja lines for FA3). The `cp` lines record where the
sources came from; `/opt/vllm` exists only where vLLM was built from source.

## DeepGEMM: `patch_dgd.py`, `patch_dgd_c.py`

Base: the DeepGEMM bundled with vLLM (`/opt/vllm/.deps/deepgemm-src`, commit `8b1392b`).

- `patch_dgd.py` adds a describe sink to `launch_kernel`. Between `describe_begin()` and
  `describe_end()`, every launch is recorded (function, grid, block, smem, cluster, PDL, argument
  bytes) instead of issued.
- `patch_dgd_c.py` (apply after `patch_dgd.py`) exposes that as a C function with DynaGraph's
  describe ABI v1, so the C++ runtime can call it without going through Python.

```bash
source /workspace/pytorch-main/dynagraph/setup/use_main.sh
export DG_DEPS=${DG_DEPS:-/workspace/_deps}
cp -r /opt/vllm/.deps/deepgemm-src $DG_DEPS/deepgemm-src
cd /workspace/pytorch-main/dynagraph
python third_party_patches/patch_dgd.py && python third_party_patches/patch_dgd_c.py
cd $DG_DEPS/deepgemm-src && ./develop.sh      # builds deep_gemm/_C in place
```

Used by `e2e/dgemm.py` (an Inductor lowering of bf16 mm/addmm to DeepGEMM). Enable it with
`GEMM=deepgemm PYTHONPATH=${DG_DEPS:-/workspace/_deps}/deepgemm-src` (see `e2e/esm.py`).

## FlashAttention 3: `fa3d_describe.patch`

Base: vLLM's flash-attention fork (`/opt/vllm/.deps/vllm-flash-attn-src`, commit `06bdd47`).

This patch adds `hopper/launch_sink.h` and a `fwd_describe` op that records the forward launches
instead of issuing them. The build is trimmed to SM90a, BF16 and head dim 128 (about 2.5 minutes to
build), and the extension is renamed `_fa3d_C`, so it can be loaded next to vLLM's own
`_vllm_fa3_C`.

```bash
source /workspace/pytorch-main/dynagraph/setup/use_main.sh
export DG_DEPS=${DG_DEPS:-/workspace/_deps}
cp -r /opt/vllm/.deps/vllm-flash-attn-src $DG_DEPS/fa3d-src
cd $DG_DEPS/fa3d-src && patch -p1 < /workspace/pytorch-main/dynagraph/third_party_patches/fa3d_describe.patch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH=/workspace/.venv/lib/python3.12/site-packages/torch/share/cmake \
  -DVLLM_PYTHON_EXECUTABLE=/workspace/.venv/bin/python
ninja -C build
```

The cmake flags are the ones recorded in the original build's `CMakeCache.txt`. Pass
`CMAKE_PREFIX_PATH` explicitly; see `docs/METHODOLOGY.md` for why. The scripts load
`$DG_DEPS/fa3d-src/build/_fa3d_C*.so` (override with `FA3D_DIR`): `serving/vllm_launches.py` and
`verification/_wf_fa3_host_cost.py`.
