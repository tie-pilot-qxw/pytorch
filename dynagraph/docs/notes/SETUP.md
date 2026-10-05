# DynaGraph development environment setup log

Goal: run **our own clone of the PyTorch main branch** in the container, not the one packaged by NVIDIA.
The things we are going to change next (`should_partition`, Inductor's cudagraph partitioning, device-side node patching) all live in the PyTorch source,
and that cannot be done with a release wheel.

## Layout

| Location | Contents |
|---|---|
| host `/nvme2n1/xinwei/auto_cuda_graph` | project root, mounted in the container as `/workspace` |
| `pytorch-main/` | PyTorch main clone, HEAD `71b32515a2` (2026-09-17) |
| `.venv/` | venv inside the container, `--system-site-packages` |
| `microbench/` | all microbenchmark scripts |
| `pytorch-main/dynagraph/docs/notes/` | plans and measurement results |
| `pytorch-main/dynagraph/setup/build_torch.sh` | PyTorch build script |
| `pytorch-main/dynagraph/setup/build_vision.sh` | torchvision build script |
| `pytorch-main/dynagraph/setup/use_main.sh` | **environment script for using the self-built torch; source it before every invocation** |
| `survey/` | P0 coverage survey harness |
| `build.log` | log of the most recent build |

Container: `xinwei_autocudagraph`. **Always go in through `use_main.sh`**; it takes care of the venv,
NVIDIA's preset version variables and `LD_LIBRARY_PATH` all at once (see pitfall 6 and pitfall 8):

```bash
docker exec xinwei_autocudagraph bash -lc 'source /workspace/pytorch-main/dynagraph/setup/use_main.sh && ...'
```

## Why a venv instead of installing straight into the system

The container originally has NVIDIA's `torch 2.11.0a0+eb65b36914.nv26.02`, installed in `/usr/local/lib/python3.12/dist-packages`.
A direct `pip install -e .` would uninstall it, and that wheel is not on PyPI: **once uninstalled, there is no getting it back**.

So we create a `--system-site-packages` venv: numpy / timm / transformers / torchvision are all inherited,
and only torch and triton are shadowed by the versions in the venv. If the build fails there is a way back, and the two torches can be checked against each other.

## Version alignment

| Component | Container system | venv | Notes |
|---|---|---|---|
| torch | 2.11.0a0 (NV) | source, editable | the venv's site-packages comes earlier in sys.path |
| triton | 3.6.0 | **3.8.0** | main's pin is `.ci/docker/triton_version.txt` = 3.8.0, available on PyPI |
| timm | 1.0.29 | inherited | |
| transformers | 5.17.0 | inherited | |

CUDA 13.1 / gcc 13.3 / cmake 3.31.6 / cudnn 9.19 / nccl 2.29 all meet the requirements
(`cmake/public/cuda.cmake:73` requires CUDA >= 12.6, and CI is already running 13.x).

## Build switches (`dynagraph/setup/build_torch.sh`)

```bash
export TORCH_CUDA_ARCH_LIST="9.0a"   # H100 only, including sm_90a's wgmma/TMA
export BUILD_TEST=0                  # do not build the C++ unit tests
export USE_ROCM=0 USE_XPU=0
export MAX_JOBS=64                   # the machine has 256 cores but load is already 180; don't crowd others out
pip install -e . -v --no-build-isolation
```

Building only the single `9.0a` arch + turning off the C++ tests are the two biggest build-time savings.
Pick `MAX_JOBS` by looking at `uptime`; this machine always has other people's jobs running.

## Pitfalls we hit

1. **Not a single submodule was initialized.** The clone used `--filter=blob:none`, and all 37 submodules were bare gitlinks.
   You must run `git submodule update --init --recursive --depth 1 -j 16`.

2. **You cannot skip submodules to save space.** At first I skipped `composable_kernel` / `aiter` / `android/libs/fbjni`,
   and ran into two walls in a row:
   - `project.license-files` in `pyproject.toml` contains `third_party/**/LICENSE.rst`, and PEP 639 requires
     every pattern to match at least one file. That file is in `third_party/python-peachpy/`, a nested submodule of aiter.
     Without pulling it you get `metadata-generation-failed`.
   - `cmake/PreBuildSteps.cmake:64` checks every submodule directory one by one for
     `CMakeLists.txt / Makefile / setup.py / LICENSE*`; an empty directory is an immediate `FATAL_ERROR`.
     Even android's fbjni is required.

   Conclusion: **pull all of them, and use `--depth 1` to keep the size down**. The three that had to be pulled afterwards came to ~480 MB in total.

3. **`pip install -e .` is the only supported way to build** (pytorch-main's `CLAUDE.md` says so explicitly);
   do not use `setup.py build` or the like.

4. **Disk space is tight.** `/nvme2n1` has only ~134 GB left (99% full). Source + submodules is ~2.3 GB, and the build directory needs tens of GB more.
   Run `df -h` before building.

5. The container's apt sources do not have `ccache`, so it cannot be installed. It does not matter for the first build, but if we are going to modify the PyTorch source over and over later, it is worth finding a separate solution.

6. **The nastiest one: the `PYTORCH_BUILD_VERSION` preset by the NVIDIA container silently wrecks the build.**

   The container carries `PYTORCH_BUILD_VERSION=2.11.0a0+eb65b36` (and also `PYTORCH_VERSION`).
   `tools/generate_torch_version.py:74` gives this environment variable priority **over** `version.txt`,
   so our main (whose `version.txt` says `2.15.0a0`) gets built as 2.11.

   The consequence is not an ugly version number; it is that **whole conditional-compilation blocks disappear**:
   `torch/csrc/stable/c/shim.h` is divided into sections by `TORCH_FEATURE_VERSION` (= `TORCH_ABI_VERSION`,
   determined by `TORCH_VERSION_MINOR` in the generated `torch/headeronly/version.h`), and
   `torch_new_stable_ivalue` / `torch_delete_stable_ivalue` sit inside the
   `#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0` block. With the version pushed down to 2.11,
   that whole block is excluded, while `torch/csrc/shim_common.cpp:153` calls them unconditionally, so the build blows up at
   2344/3495:

   ```
   torch/csrc/shim_common.cpp:153:30: error:
       'torch_new_stable_ivalue' was not declared in this scope
   ```

   This error points nowhere near the real cause, and it is easy to mistake it for a bug in the main branch itself and go revert commits.

   Fix: before building
   ```bash
   unset PYTORCH_BUILD_VERSION PYTORCH_VERSION PYTORCH_BUILD_NUMBER
   ```
   and also run `git config --global --add safe.directory /workspace/pytorch-main`;
   otherwise git inside the container cannot get the sha because of dubious ownership, and the version becomes `2.15.0a0+gitUnknown`.
   After the change, **delete `build/CMakeCache.txt`** and re-run configure, because
   `CMAKE_PROJECT_VERSION` is cached as STATIC. `build/` was created by root inside the container,
   so it has to be deleted from inside the container.

   How to confirm:
   ```bash
   python -c "import sys;sys.path.insert(0,'tools');
   from generate_torch_version import get_torch_version;print(get_torch_version())"
   # expect 2.15.0a0+git71b3251, not 2.11.0a0+eb65b36
   ```

7. **As soon as the `docker exec` session drops, ninja inside the container gets interrupted.**

   The symptom is the build stopping halfway at
   ```
   ninja: build stopped: interrupted by user.
   ```
   but `grep -c FAILED build.log` is 0: not a single compile error. This is not a build problem;
   it is the process inside the container receiving a signal after the host-side `docker exec` was reaped.

   The build runs for tens of minutes, so it has to be detached from the host session:

   ```bash
   docker exec -d xinwei_autocudagraph bash -lc \
     'bash /workspace/pytorch-main/dynagraph/setup/build_torch.sh > /workspace/build.log 2>&1'
   ```

   `-d` means detached. The log is written to `/workspace/build.log`, because `/workspace` is the host's
   project root and can be read directly from outside. ninja is incremental, so a restart picks up where it was cut off; no work is wasted.

8. **`LD_LIBRARY_PATH` makes the freshly built torch load the old shared libraries.**

   The build succeeded and `pip` also said `Successfully installed torch-2.15.0a0+git71b3251`, but
   `import torch` reported

   ```
   AttributeError: module 'torch._C' has no attribute '_has_gds'
   ```

   Cause: the `torch/_C...so` produced by the editable install is only a 15 KB stub; the real symbols are in the
   35 MB `libtorch_python.so`. And the **first entry** of the NVIDIA container's `LD_LIBRARY_PATH` is
   `/usr/local/lib/python3.12/dist-packages/torch/lib`, so the stub linked against the 2.11 copy of the library.
   `ldd` makes it obvious at a glance:

   ```
   libtorch_python.so => /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_python.so
   ```

   For the fix see `dynagraph/setup/use_main.sh`: strip the system torch and torch_tensorrt lib directories out of `LD_LIBRARY_PATH`,
   then put `/workspace/pytorch-main/build/lib` first. **From now on, source it before every invocation.**

9. **The self-built torch breaks the container's preinstalled torchvision, and takes all of transformers down with it.**

   torchvision 0.25 was built for torch 2.11; the ABI does not match, and it reports

   ```
   RuntimeError: operator torchvision::nms does not exist
   ```

   The trap is that this error **does not only affect image models**: `transformers/modeling_utils.py` does
   `from .loss.loss_utils import LOSS_MAPPING`, and that chain depends on torchvision, so
   **every** HuggingFace model fails to import, and what gets reported is a
   `ModuleNotFoundError: Could not import module 'modeling_albert'` that has swallowed the underlying cause.

   The fix is to build torchvision main as well (`dynagraph/setup/build_vision.sh`, much faster than PyTorch).
   The same goes for torchaudio; deal with it when we need it.

## The survey itself does not need a GPU

`Scheduler.should_partition` is a **compile-time** decision and does not execute any kernel. The coverage survey should run entirely on fake tensors,
so it neither takes GPU memory nor steals compute from others (the 8 GPUs on this machine are always being used by other users).

On the container's bundled 2.11, the naive approach fails:

```
AssertionError: fake mode (...) from tracing context 0
               doesn't match mode (...) from fake tensor input 0
```

**main has already fixed this one itself**: `detect_fake_mode` in `torch/_guards.py` now authoritatively returns
the TracingContext's mode, and `process_inputs` in `torch/_functorch/_aot_autograd/frontend_utils.py`
re-fakifies inputs that come from a foreign mode. This is one more concrete reason why "we must use a self-built main".

Measured after installing 2.15: **compilation runs all the way through; only the cudagraph run phase fails**
(`FakeTensorDeviceMismatchError: cuda:0 and meta` at record time in `cudagraph_trees.py`).
That part is exactly what we do not need: `should_partition` and `graph_partition` have already run long before,
the data is complete, and `memory_allocated()` stays at 0 throughout.
So the harness treats "`graph_partition` has run" as the success criterion and records a `compiled_but_not_run` flag.

Note that wrapping things in `torch._guards.tracing(TracingContext(my_mode))` is **useless**; measured, Dynamo installs its own mode anyway.
The details and three alternative paths are recorded in the fake tensor section of `docs/notes/FEASIBILITY.md`.

The cost is a ~520 MiB CUDA primary context per process (0 tensor memory, no kernels launched),
and **there must be a real, visible GPU**: `CUDA_VISIBLE_DEVICES=""` fails inside AOTAutograd.

## GPU usage discipline

On this machine, other people are always running performance tests on the 8 GPUs. Only **GPU 4 and GPU 5** are authorized; before using one, run
`nvidia-smi --query-compute-apps=...` to see whose processes are on it. Right after a test, confirm that you left no GPU memory behind.
This project only takes timing numbers when no one else's job is running, or when the impact has been confirmed to be negligible.

---

## Pitfall 10: dependencies of the GEMM backends (installed 2026-09-18)

Inductor has nine GEMM candidate backends:
`ATEN / TRITON / CUTLASS / CUTEDSL / NVGEMM / CK / CKTILE / CPP / FLYDSL`
(all the values `_use_autotune_backend("...")` accepts). But **only three are enabled by default**:

```python
# config.py:673
max_autotune_gemm_backends = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON,CPP").upper()
```

Availability of each backend on this machine (H100 / sm_90a / CUDA 13.1):

| Backend | Status | Reason |
|---|---|---|
| ATEN | available | cuBLAS, on by default |
| TRITON | available | |
| CUTLASS | available | **but no fp32 support**, only fp16/bf16/int32 (see the comment at utils.py:2979, which points to pytorch#145952) |
| **NVGEMM** | **available only after installing** | see below |
| CUTEDSL | **unavailable** | requires `is_datacenter_blackwell_arch()`; H100 is Hopper, not Blackwell |
| CK / CKTILE | unavailable | ROCm only |

**NVGEMM needs two packages installed** (`ensure_nv_universal_gemm_available()` checks whether
`cutlass.operators` can be imported):

```bash
# the container ships nvidia-cutlass-dsl 4.3.5, which has only cutlass.cute and no cutlass.operators
pip install -U "nvidia-cutlass-dsl==4.7.1"      # operators requires >= 4.6
pip install /workspace/pytorch-main/third_party/cutlass/operators   # package name is nvidia-cutlass-operators
```

Install both into the **venv** (`/workspace/.venv`); it comes before
`/usr/local/lib/python3.12/dist-packages` in `sys.path`, so it shadows the container's bundled copy
without touching NVIDIA's original environment.

**`cutlass.operators` is not in the `nvidia-cutlass-dsl` on PyPI** -- it is in PyTorch's own
submodule `third_party/cutlass/operators/`. At the time I thought upgrading the DSL package would be enough;
after upgrading to 4.7.1, `cutlass.operators` still could not be imported, and only after searching around did I find it in the tree.
(The `cutlass/` directory has no `__init__.py`; it is an implicit namespace package, and once installed it does merge with the DSL's copy.)

Verification after installing:

```python
from torch._inductor.utils import ensure_cute_available, ensure_nv_universal_gemm_available
ensure_nv_universal_gemm_available.cache_clear()   # clear the cache after installing; the function has an lru_cache
assert ensure_nv_universal_gemm_available()
```

## Pitfall 11: a long-running container loses its GPU permissions (2026-09-19)

Symptom: the host is perfectly fine and other people's jobs keep running, but inside the container

```
Failed to initialize NVML: Unknown Error
torch.cuda.device_count() == 0        # accompanied by UserWarning: Can't initialize NVML
```

**Do not suspect the driver first.** The test is to open the device directly:

```bash
docker exec <container> python3 -c '
import os
for d in ("/dev/nvidiactl", "/dev/nvidia0"):
    try: os.close(os.open(d, os.O_RDWR)); print(d, "can be opened")
    except OSError as e: print(d, e.errno, e.strerror)'
```

If you get `errno=1 Operation not permitted`, the device nodes are present and the driver libraries are correct;
it is just that **the container's device cgroup was reprogrammed and the permission revoked**. Under cgroup v2 this is an eBPF program,
there is no `devices.list` to look at, so probing like this is the only way. The trigger is usually a
`systemctl daemon-reload` on the host: systemd resets the device-allow program of that scope,
wiping out the permissions that nvidia-container-runtime injected at creation time. **It only affects containers that are already running.**

Recreating the container with `--privileged` makes it immune. Copy the original container's configuration exactly:

```bash
docker rename <old-name> <old-name>_nogpu          # keep it so you can fall back; do not delete it outright
docker run -d --name <old-name> \
  --privileged --gpus all --ipc=host --shm-size=32g \
  -v /nvme2n1/xinwei/auto_cuda_graph:/workspace \
  -v /ssd2/xinwei/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace nvcr.io/nvidia/pytorch:26.02-py3 sleep infinity
```

Recreating is cheap, because **the venv and the self-built torch are both on the mounted disk**
(`/workspace/.venv`, `/workspace/pytorch-main`), not in the container layer.
After recreating, check the usual three things: `nvidia-smi -L`, `torch.__file__` points to `/workspace/pytorch-main`,
and `hasattr(torch._C._StaticCudaLauncher, "_begin_device_node_collection")`.

## Pitfall 11: building third-party CUDA extensions against this torch (hit on 2026-09-21 while building vLLM)

To build packages with CUDA extensions such as vllm / sgl-kernel / flashinfer inside the container, there are two layers to deal with;
miss either one and you build (or fail to build) something targeting **2.11**.

**Layer 1: `source use_main.sh`.** Without sourcing it, both `pip` and `python` are the system ones,
and `import torch` gets NVIDIA's 2.11 from dist-packages. The symptom is a compile-time "this API does not exist" error,
even though you can clearly grep for it in `/workspace/pytorch-main`. What it actually looked like (vLLM 0.29's
`csrc/libtorch_stable/cuda_view.cu`):

```
error: class "torch::stable::Tensor" has no member "layout"
error: no suitable conversion function from "lambda [](void *)->void" to "int64_t" exists
```

Both are things that are not yet in the 2.11 stable ABI (`Tensor::layout()` exists since 2.9 but that copy of the headers is old;
the `from_blob` overload taking a lambda deleter was only added in 2.11, see pytorch `62a71f1481`).
How to diagnose: pull nvcc's `-isystem` flags out of the build log and see which torch they point to.

```bash
grep -o "\-isystem /[^ ]*torch/include" build.log | sort -u
```

**Layer 2: `torch.utils.cmake_prefix_path` is broken in this environment and cannot be trusted.**
It computes `os.path.dirname(os.path.dirname(torch.__file__)) + "/share/cmake"`,
i.e. `/workspace/pytorch-main/torch/share/cmake` -- the source tree is an in-place build and
**has no installed C++ layout** (`torch/include`, `torch/share/cmake/Torch`, `torch/lib/libtorch*.so`
none of them exist; the libraries are in `build/lib`). When CMake's `find_package(Torch)` cannot find it, it silently falls back to
the 2.11 in dist-packages.

The usable 2.15 headers and `TorchConfig.cmake` are in **the venv's site-packages** (the same set that
`torch.utils.cpp_extension.include_paths()` returns, so extensions built through cpp_extension are fine; only those built with plain CMake
get bitten):

```
/workspace/.venv/lib/python3.12/site-packages/torch/{include,lib,share/cmake}
```

So point CMake there explicitly before building:

```bash
export CMAKE_PREFIX_PATH=/workspace/.venv/lib/python3.12/site-packages/torch/share/cmake:$CMAKE_PREFIX_PATH
```

**Full recipe for building vLLM** (0.29.0, sm_90, against our torch):

```bash
source /workspace/pytorch-main/dynagraph/setup/use_main.sh
export PATH=$HOME/.cargo/bin:$PATH          # 0.29 has Rust components and the container has no rust; install rustup first
export CMAKE_PREFIX_PATH=/workspace/.venv/lib/python3.12/site-packages/torch/share/cmake:$CMAKE_PREFIX_PATH
export VLLM_TARGET_DEVICE=cuda TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=64 NVCC_THREADS=4
cd /opt/vllm && python use_existing_torch.py      # strips torch==2.13.0 out of requirements
pip install --no-build-isolation --no-deps -e . -v
```

`--no-build-isolation` is there so it builds against our torch; the cost is that you have to install the build backends into the venv yourself
(`setuptools-rust`, `setuptools-scm`, `wheel`, `jinja2`, `cmake`, `ninja`).
`--no-deps` is there so pip does not replace our torch.

**Disk**: the 7T disk behind `/workspace` is full (only ~37G left), so the vLLM source + build artifacts live on the overlay at
`/opt/vllm` (which still has ~140G). The cost is that they are gone when the container is recreated.

**Good news**: most vLLM kernels now go through `libtorch_stable` (`TORCH_TARGET_VERSION=2.11`),
so of the 427 objects only one actually hit the version drift; the rest do not care which torch version is used.
