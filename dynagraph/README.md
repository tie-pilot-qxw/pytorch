# DynaGraph: one CUDA graph for every shape

DynaGraph (DG) is a research prototype built into this PyTorch fork (branch `dynagraph`).
It lets a `torch.compile`d program with dynamic shapes run from a single captured CUDA graph per
region, patching the graph's kernel-node parameters for each call's shapes. The usual alternative
records one graph per shape or pads inputs to a few fixed sizes.

This directory holds everything except the library code: notes, the end-to-end harness, workloads,
regression probes, microbenchmarks and serving experiments. Start here, then read
[docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) and [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Goal

CUDA graphs remove CPU launch overhead, which dominates many modern workloads: small batches,
decode steps, small DiTs, GNNs, and tensor-parallel shards where each kernel is tiny. But a CUDA graph
bakes in shapes and pointers. Today a dynamic-shape program has three options, each with a cost:

- **Record one graph per shape.** This is PyTorch's cudagraph trees under `mode="reduce-overhead"`.
  Every new shape pays for a warmup and a recording, and memory grows with the number of shapes.
- **Pad to a few fixed sizes (bucketing)** and record one graph per bucket. This is what vLLM and
  SGLang do. It wastes compute on padding and needs a compile and capture per bucket, and each model
  needs engineering work: capture sizes, padders, and allow-lists.
- **Run without graphs.** This keeps paying the launch overhead.

**DynaGraph's goal is to make CUDA graphs work for dynamic-shape PyTorch programs automatically:
compile once, capture once, and serve every shape from that graph at close to graph-replay speed,
with no per-model engineering.** Xinwei set four axes. They pull against each other, and
every change should say which one it spends and which one it buys:

1. **Coverage.** Serve as much real code as possible from the graph: Inductor Triton kernels,
   library calls (cuBLAS, cuDNN, FlashAttention, DeepGEMM), user Triton kernels, `torch.cond`,
   data-dependent sizes, training steps, and collectives. Every fallback is a coverage hole.
2. **Latency.** Turning DynaGraph on must not make a step slower than the alternatives. Per-call
   host overhead and any GPU work DynaGraph adds count against it.
3. **Single and multi GPU together.** Tensor and data parallelism make kernels smaller and
   launch overhead relatively larger.
4. **Badly written code counts.** Evaluate on messy research code as well as clean reference
   implementations. Such code typically has no shape discipline and no CUDA graph support of its
   own.

### The question for the next phase

The mechanism works and is correct on real models (see "Status" below). The open research question
is **where it pays off**. Concretely: for which real workloads, under which deployment conditions, does
serving all shapes from one patched graph beat the practical alternatives? The alternatives are
eager, `torch.compile` without graphs, bucketing with per-bucket graphs, per-shape recording, vLLM
piecewise graphs and SGLang breakable CUDA graphs. "Beat" can mean any of:

- **steady-state latency or throughput**;
- **cold start** (compile and capture count);
- **graph memory**;
- **engineering effort** (how much per-model work the alternative needs).

[docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) records everything measured so far, with the exact
setups, so it can be reproduced, extended or challenged. The "Open directions" section below lists
starting points; it is not a plan.

## How DynaGraph works (short version)

1. `torch.compile(model, dynamic=True, mode="reduce-overhead")` produces Inductor regions whose
   kernels take symbolic sizes. With `torch._inductor.config.triton.dynagraph = True`, the
   cudagraph-trees wrapper hands each region to a `DynaGraphRunner` instead of recording per shape
   (`torch/_inductor/cudagraph_trees.py`, `deferred_cudagraphify`).
2. The runner captures the region once. It moves all intermediates into an **arena** it owns and
   lays out per shape, and builds a table of every kernel node: which arguments are pointers, sizes
   or grids, and how each is computed from the symbolic sizes. It reads this from the generated
   wrapper code.
3. On every call it computes the arena layout for the actual shapes and **patches the graph**:
   grids, scalar arguments and pointers. This happens either on the host
   (`cuGraphExecKernelNodeSetParams`, from a generated C++ runtime) or on the device (a "planner"
   kernel inside the graph uses the device graph-update API). Then it launches.
4. Library calls whose kernels are opaque are handled in tiers:
   - **Tier 1.** Inductor Triton kernels. They are patched directly.
   - **Tier 2.** Extern calls (cuBLAS, cuDNN, ...). Each one is captured ("harvested") into a
     child graph per shape signature and swapped in with `cudaGraphExecChildGraphNodeSetParams`. If
     a library picks a different node topology for a new shape, DG keeps one main graph per
     topology combination, or uses a SWITCH node.
   - **Tier 3.** Operators that **declare** their launches through a small C ABI
     (`torch/utils/_capture_launch.py`) are inlined into the main graph and patched like Triton
     kernels.
5. For the first `dynagraph_verify_shapes` shapes, every replay is checked against eager. A region
   that disagrees, or that DynaGraph cannot model, falls back to ordinary per-shape recording and
   logs a tagged reason.

The longer version is in [docs/notes/TIERS.md](docs/notes/TIERS.md),
[docs/notes/EXTERN.md](docs/notes/EXTERN.md), [docs/notes/REGISTER.md](docs/notes/REGISTER.md) and
[docs/notes/ENGINE.md](docs/notes/ENGINE.md).

### Terms used throughout

| term | meaning |
|---|---|
| region | one Inductor-compiled graph partition that cudagraph trees would record; DG serves it with one `DynaGraphRunner` |
| arena | the memory block a runner owns for the region's intermediates, laid out anew for each shape |
| planner | the generated kernel that patches the graph from inside it (device path) |
| site | one extern call (cuBLAS, cuDNN, a custom op) inside a region |
| harvest | capturing a site on its own into a child graph for one shape signature (tier 2) |
| signature, hkey | the key a runner caches per-shape work under (symbolic sizes plus what else the sites depend on) |
| topology | the node structure a library call produces for a shape; one main graph is kept per combination of site topologies, or a SWITCH node selects among bodies |
| SWITCH | a CUDA conditional node with several bodies, selected on the device |
| describe | a library entry point that reports the launches a call would make instead of issuing them (tier 3) |
| lane | an extra arena for a region called several times in one step (e.g. once per layer) |
| fallback | DG declines a region or a shape and hands it to ordinary per-shape recording; logged as `fallback [tag]` |
| oracle, bucket, pad, trees | baselines; see [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) |

"Tier" means something else in two of the notes: in `docs/notes/TIERS.md` and sections 15-19 of
`docs/notes/EXTERN.md` it is about where a size lives (static, a host symbol, or a value produced on
the GPU), not the library tiers above.

## Status

What exists and runs:

- **Correctness on real models.** The training and inference workloads below run fully served (no
  fallbacks) and match eager or `torch.compile` within normal bf16/fp32 tolerance:
  - ESM-2 35M variable-length training;
  - GraphSAGE, SchNet, MACE (MD22), point clouds;
  - BERT with random (B, L) per step (the other HF models in `e2e/zoo.py` were only run without
    DG);
  - SANA 1.5 1.6B DiT (901/901 calls served).
- **Host path in C++.** For regions patched from the host, every per-call step (cache key, layout,
  node patches, output tensors, launch) runs in `torch/_inductor/dynagraph_rt.cpp`, and Python only
  builds, verifies and falls back. Known shapes cost about 0.15 ms of DG host time per call on
  BERT. Regions on the device path, and regions with device-resolved sizes, TMA descriptors or
  declared dependencies, still run Python per call (`_rt_region` in `dynagraph.py`).
- **Device path.** A generated planner kernel patches the graph from inside it. With
  `dynagraph_unbacked="device"` it also handles data-dependent (unbacked) sizes without a host
  sync, and `torch.cond` as a conditional node.
- **Tier 2 and tier 3.** Tier 2 harvests cuBLAS, cuDNN and SDPA calls into child graphs. Tier 3 is
  implemented, with a DeepGEMM describe as the worked example (see `third_party_patches/`). FA3/CuTe
  varlen training is served.
- **Regression suites.** `probes/regress_quick.sh` runs 12 probes and `probes/regress.sh` runs 45.
- **Multi-GPU.** Only probes so far (NCCL inside a region; DDP training step); there is no
  multi-GPU workload measurement yet.

Known limitations to be aware of:

- **Topology changes in libraries are costly.** When a cuBLAS or cuDNN call picks a different
  kernel set for a new shape, DG harvests a new child graph and may re-capture the main graph.
  On a BERT run with cuBLAS GEMMs this dominated new-shape cost: 14.7 ms per step. Routing GEMMs to
  Inductor's Triton templates avoids it (`GEMM=triton` in the harness; in your own script,
  `ic.max_autotune_gemm = True` and `ic.max_autotune_gemm_backends = "TRITON"`).
- **The per-call cost is mostly per-node `SetParams`** (~0.57 us/node).
- **`dynamic=True` can still specialize sizes.** Duck sizing, guards on Python state, and ops that
  require a constant size all do this. A specialized size forces a recompile that DynaGraph cannot
  hide. See docs/METHODOLOGY.md.

## Getting started

**If you were given a container to log in to over SSH as a normal user**, you are already inside
it: `/workspace` is your project directory (the `<host-project-dir>` below), so skip `docker run`,
clone and run git inside the container, and run the commands shown as
`docker exec dynagraph bash -lc '...'` directly (for `docker exec -d`, use `nohup ... > log 2>&1 &`).
You cannot see who owns the processes on a GPU from inside a container, so ask whoever gave you the
container which cards to use. Anything that needs docker or root (restarting the container,
installing system packages) also goes through them.

### 1. Environment

Everything was developed on 8x H100 80GB (sm_90a), CUDA 13.1, inside the NVIDIA container
`nvcr.io/nvidia/pytorch:26.02-py3`. A project directory on the host is mounted at `/workspace` and
holds the fork checkout, a torchvision checkout, a venv and third-party sources:

```
/workspace/
  pytorch-main/        this repo (branch dynagraph)
  torchvision-main/    torchvision main, built against this torch
  .venv/               python -m venv --system-site-packages (shadows only torch/triton/torchvision)
  _deps/               optional third-party sources and data ($DG_DEPS), datasets in _deps/data ($DG_DATA)
```

```bash
docker run -d --name dynagraph --gpus all \
  $(for d in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do echo --device $d; done) \
  --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v <host-project-dir>:/workspace -v <hf-cache>:/root/.cache/huggingface \
  -w /workspace nvcr.io/nvidia/pytorch:26.02-py3 sleep infinity
```

List the device nodes with `--device` as well as `--gpus all`. `--gpus all` alone adds the GPU
permissions when the container is created, but not to its device rules, so a host
`systemctl daemon-reload` (systemd, cgroup v2) revokes them and the container loses its GPUs
("Failed to initialize NVML: Unknown Error"). Devices listed with `--device` are part of the
container's configuration and survive. `--device nvidia.com/gpu=all` (CDI) does the same, but on
this host the CDI spec lives in `/var/run` and is regenerated at boot with no ordering against
docker, so a CDI container may not come back after a reboot. The notes used `--privileged` for the
same reason; it works too, but it gives root in the container full access to the host.

One-time setup. Clone on the host (git stays on the host, see below), then create the venv inside
the container:

```bash
# on the host, in <host-project-dir>
git clone -b dynagraph https://github.com/tie-pilot-qxw/pytorch.git pytorch-main
git -C pytorch-main submodule update --init --recursive --depth 1 -j 16    # all of them; skipping any breaks the build
git clone https://github.com/pytorch/vision.git torchvision-main             # main; 7b0e250 was used

# in the container
docker exec dynagraph bash -lc 'python -m venv --system-site-packages /workspace/.venv && /workspace/.venv/bin/pip install triton==3.8.0'
```

The venv inherits the container's packages (numpy, transformers, ...) and shadows only torch,
triton and torchvision. Never `pip install` torch outside it: the container's NVIDIA torch wheel is
not on PyPI and cannot be reinstalled.

### 2. Build

Read [docs/notes/SETUP.md](docs/notes/SETUP.md) first: it lists the traps that each cost hours. The
short path:

```bash
docker exec -d dynagraph bash -lc 'bash /workspace/pytorch-main/dynagraph/setup/build_torch.sh > /workspace/build.log 2>&1'
# then, once torch is built:
docker exec dynagraph bash -lc 'bash /workspace/pytorch-main/dynagraph/setup/build_vision.sh'
```

Use `docker exec -d` (over SSH, `nohup ... &`): an interrupted session kills ninja. Every later command
starts with:

```bash
source /workspace/pytorch-main/dynagraph/setup/use_main.sh
```

This script activates the venv, unsets the NVIDIA `PYTORCH_BUILD_VERSION` variables (they silently
break the build) and puts the self-built libraries first on `LD_LIBRARY_PATH`.

**Do not run git as root inside the container.** A git command run as root leaves files in `.git`
owned by root, and the next commit from your own account fails. With a root container, run git on
the host. If root's git refuses with "detected dubious ownership", do not add a `safe.directory`
exception; run git as the user who owns the checkout.

### 3. Smoke test

```bash
GPU=<card> bash /workspace/pytorch-main/dynagraph/probes/regress_quick.sh   # 12 probes, all should PASS
```

Logs go to `$DG_OUT/_regress_logs` (`DG_OUT` defaults to `/tmp/dynagraph_out`). A failing probe's
log ends with the DynaGraph fallback tag. Before picking a card, see who else is on it
([docs/METHODOLOGY.md](docs/METHODOLOGY.md), "The machine"): other people's jobs distort timings,
and yours can make theirs OOM. If every probe fails with `CUDA_ERROR_UNKNOWN` or "devices busy or
unavailable" on a card that looks idle, see "A container can lose its GPUs" in METHODOLOGY.

**Clean the Inductor cache regularly.** Most probe and survey scripts set
`force_disable_caches = True`, and with it Inductor leaves one `tmp*` directory per compilation
under its cache directory (`/tmp/torchinductor_<user>` unless `TORCHINDUCTOR_CACHE_DIR` is set) and
never deletes it. A week of probe runs left about 100 GB there. Prune it now and then:

```bash
find "${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_$USER}" -maxdepth 1 -name 'tmp*' -mmin +60 -exec rm -rf {} +
```

### 4. Run a workload

```bash
cd /workspace/pytorch-main/dynagraph
# HF model, random (B, L) every step: eager / compile / per-shape graphs / pad / pow2 buckets / replay-only oracle / DG
AMP=1 python e2e/zoo.py --model bert --bs 1 --modes eager,compile,trees,pad,bucket,oracle,dg
# same, GEMMs through Triton templates and eager attention (no tier-2 harvest)
GEMM=triton ATTN=eager AMP=1 python e2e/zoo.py --model bert --bs 1,2,4,8,16,32 --modes compile,dg,bucket,oracle
```

The harness (`e2e/harness.py`) runs every mode from the same initialization over the same batches.
It times three segments:

- **warm**: compilation and first recordings;
- **new**: shapes not seen before; this is the realistic steady state when shapes keep changing;
- **replay**: the new segment again, every shape now seen.

For each mode it reports recordings, DG serves and fallbacks, graph breaks, peak memory and loss
deltas. Set `NOSYNC=1` to time without a sync per step (host and GPU overlap).

### 5. Turn it on in your own script

```python
import torch, torch._inductor.config as ic
ic.triton.dynagraph = True                 # or TORCHINDUCTOR_DYNAGRAPH=1
ic.triton.dynagraph_extern_child = True    # keep cuBLAS/cuDNN calls in the graph (tier 2)
f = torch.compile(model, dynamic=True, mode="reduce-overhead")
```

To route GEMMs to Triton templates as the harness's `GEMM=triton` does, also set
`ic.max_autotune_gemm = True` and `ic.max_autotune_gemm_backends = "TRITON"`.

Set `logging.getLogger("torch._inductor.dynagraph").setLevel(logging.INFO)` to see what is served
and why something fell back. All knobs live in `torch/_inductor/config.py` under `triton.dynagraph_*`.
Each one has a comment, and each can be set with an environment variable
(`TORCHINDUCTOR_DYNAGRAPH_*`). The ones you will touch:

| knob | meaning |
|---|---|
| `dynagraph_update` (`auto`/`host`/`device`) | who patches the graph per call |
| `dynagraph_extern_child` | tier 2: harvest extern calls into child graphs |
| `dynagraph_declared_launches` | tier 3: inline operators that declare their launches |
| `dynagraph_topology` (`host`/`switch`) | how a library topology change is followed |
| `dynagraph_max_graphs` | main graphs kept per region (LRU) |
| `dynagraph_verify_shapes` | shapes checked against eager before replays are trusted |
| `dynagraph_layout` (`dynamic`/`fixed`) | arena layout policy |
| `dynagraph_unbacked` (`host`/`device`) | where data-dependent sizes are resolved; `device` is needed to keep `.item()`-sized work and `torch.cond` inside the graph |

Environment variables used by the scripts in this directory:

| variable | default | used for |
|---|---|---|
| `DG_DEPS` | `/workspace/_deps` | third-party sources (FA4 site, DeepGEMM, MACE site, SGLang checkout, ...) |
| `DG_DATA` | `/workspace/_deps/data` | datasets (`survey/workload_shapes/dl.sh` and `dl2.sh` download them) |
| `DG_OUT` | `/tmp/dynagraph_out` | logs and generated artifacts |
| `GPU` | `0` | card for the regression scripts |

### 6. Optional third-party dependencies

`probes/regress_quick.sh` and the HF-model zoo need nothing beyond the build above. `probes/regress.sh`
also needs FA3 and fa4site (for `probe_train_varlen_fa`). Other scripts need the following; install
only what you use. In a container you were given, `$DG_DEPS` may already hold these sources (and
the datasets in `$DG_DATA`): check before downloading, and see `third_party_patches/README.md`
before patching anything there.

| dependency | used by | how it was set up |
|---|---|---|
| PyG stack, schnetpack, mace-torch, spconv, ogb, ase | `e2e/sage.py`, `schnet.py`, `mace_md.py`, `pointcloud.py` | installed into the venv; versions and the pyg-lib source build are in `docs/notes/E2E.md` section 1 |
| `$DG_DEPS/mace_site` | `e2e/mace_md.py` | `pip install --target $DG_DEPS/mace_site --no-deps e3nn==0.4.4` (MACE needs this older e3nn; put it first on `PYTHONPATH`, as the usage line in `e2e/mace_md.py` shows) |
| `$DG_DEPS/flash-attention` | the two rows below | `git clone https://github.com/Dao-AILab/flash-attention.git` at `edb5c76` |
| FA3 (`flash_attn_interface`) in the venv | `probes/probe_train_varlen_fa.py` (`--attn fa3_varlen`, the default) | `cd $DG_DEPS/flash-attention/hopper && pip install --no-build-isolation .` with the venv active |
| `$DG_DEPS/fa4site` (FA4, `flash_attn.cute`) | `probes/probe_train_varlen_fa.py`, `serving/probe_vllm_*.py` | `pip install --no-deps --target $DG_DEPS/fa4site $DG_DEPS/flash-attention/flash_attn/cute einops==0.8.2 apache-tvm-ffi==0.1.14.post0 torch-c-dlpack-ext==0.1.5 quack-kernels==0.6.5`, then `touch $DG_DEPS/fa4site/flash_attn/__init__.py`. Use `--no-deps`: otherwise pip puts a PyPI torch into the directory, and these scripts put it first on `sys.path`. `nvidia-cutlass-dsl` comes from the venv (`docs/notes/SETUP.md`, pitfall 10). The empty `flash_attn/__init__.py` shadows the container's flash_attn 2.7.4, which is built against its own torch and fails to import |
| vLLM, built from source | `serving/probe_vllm_*.py`, `serving/vllm_launches.py` | `docs/notes/SETUP.md`, the last pitfall (building third-party CUDA extensions against this torch). It was built in `/opt/vllm`, which needs root; without root, build it under `$DG_DEPS/vllm` |
| SGLang | `serving/sglang_bench.sh`, `serving/bcg_bench.py` | Needs its own container, set up by whoever administers the machine. A separate container with stock torch 2.11 (a local image, `sglang-ab:base`, with SGLang and its kernels installed), not the self-built torch. The SGLang code came from `$DG_DEPS/sglang-main` (`2a1c477` plus `serving/sglang_experiment_knobs.patch`, `git apply`) via `PYTHONPATH`; the diffusion extras were installed with `--no-deps`. Fake weights: `python serving/fake_hf_repo.py <hf-repo> <out-dir>`, with `<out-dir>` mounted at `/fake`. Details in `docs/notes/DIFFUSION.md` |
| ShareGPT | `serving/probe_vllm_mixed.py` | `hf download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset`; set `SHAREGPT` to the file if it is not in the HF cache under `$HF_HOME` (default `~/.cache/huggingface`) |
| `$DG_DEPS/deepgemm-src` | `GEMM=deepgemm` runs (`e2e/sage.py`, `schnet.py`, `esm.py`), and always `e2e/pointcloud.py` (its grouped GEMM) | [third_party_patches/README.md](third_party_patches/README.md) |
| `$DG_DEPS/fa3d-src` | FA3 describe (tier 3): `serving/vllm_launches.py` | [third_party_patches/README.md](third_party_patches/README.md) |

Datasets for the `e2e/` workloads live under `$DG_DATA` in this layout:

| path under `$DG_DATA` | used by | source |
|---|---|---|
| `ogb/ogbn_arxiv/` | `e2e/sage.py` | downloaded by `ogb` on first use |
| `QM9/` | `e2e/schnet.py` | downloaded by `torch_geometric.datasets.QM9` on first use |
| `MD17/aspirin/raw/md17_aspirin.npz` | `e2e/mace_md.py` (default) | `python -c "from torch_geometric.datasets import MD17; MD17('$DG_DATA/MD17', name='aspirin')"` |
| `MD22/md22_double-walled_nanotube.npz`, `MD22/md22_buckyball-catcher.npz` | `e2e/mace_md.py --npz ...` | `survey/workload_shapes/dl2.sh` saves them as `$DG_DATA/md22_nanotube.npz` and `md22_bucky.npz`; move them: `mkdir -p MD22 && mv md22_nanotube.npz MD22/md22_double-walled_nanotube.npz && mv md22_bucky.npz MD22/md22_buckyball-catcher.npz` |
| `kitti/2011_09_26/2011_09_26_drive_0093_sync/velodyne_points/data/*.bin` | `e2e/pointcloud.py` | KITTI raw drive 0093, unzipped (URL in `dl2.sh`) |
| `human_proteome.fasta.gz` | `e2e/esm.py`, `probes/bench.py` | UniProt UP000005640 (URL in `survey/workload_shapes/dl.sh`) |

The survey scripts in `survey/workload_shapes/` read raw files from the same directory; see
`survey/workload_shapes/README.md`.

## Where things are

Library code, in the fork:

| file | what |
|---|---|
| `torch/_inductor/dynagraph.py` | everything Python-side. `DynaGraphRunner` (build, capture, harvest, `__call__`); `extract_kernel_table` (reads the generated wrapper); `generate_planner` (device path); `generate_host_patcher`; arena layout (`plan_slots`, `slot_sizes`, ...) |
| `torch/_inductor/dynagraph_rt.cpp` | the C++ per-call runtime (`Region`): cache key, layout, node patches, output tensors, launch. It is built on first use with `load_inline` |
| `torch/_inductor/cudagraph_trees.py` | the hook (`deferred_cudagraphify`) that routes a region to DynaGraph |
| `torch/_inductor/config.py` | `triton.dynagraph*` knobs |
| `torch/utils/_capture_launch.py` | tier-3 declared-launch ABI (`Declaration`, `register`, C `describe` ABI v1) |
| `torch/utils/_capture_deps.py`, `_capture_tma.py` | declarations of what a capture reads, and host-built TMA descriptors |
| `torch/csrc/inductor/static_launcher/cuda.cpp`, `torch/_inductor/scheduler.py`, `codegen/wrapper.py`, `ir.py`, `lowering.py` | supporting changes: device-updatable launches, partitioning, static user-Triton launches |
| `torch/distributed/distributed_c10d.py` | collectives inside a captured region (probes only) |

This directory:

| path | what |
|---|---|
| `docs/MEASUREMENTS.md` | every measurement so far, with setups and commands |
| `docs/METHODOLOGY.md` | how to measure without fooling yourself; read before timing anything |
| `docs/notes/` | the original working notes, translated (see `docs/notes/README.md` for an index) |
| `setup/` | build scripts and `use_main.sh` |
| `probes/` | regression probes (`regress_quick.sh`, `regress.sh`), unit-style tests, micro-probes of single features |
| `e2e/` | end-to-end harness (`harness.py`) and workloads: `zoo.py` (HF models, random shapes), `esm.py`, `sage.py`, `schnet.py`, `mace_md.py`, `pointcloud.py`, `sana.py`. `e2e/probes/` holds focused probes (GEMM pad cost, per-call breakdown, ...) |
| `serving/` | vLLM probes (mixed prefill+decode batches, decode latency), SGLang diffusion benchmark (`bcg_bench.py`, `sglang_bench.sh`), the fake-weights generator (`fake_hf_repo.py`), and `sglang_experiment_knobs.patch` (local SGLang changes used in those experiments) |
| `microbench/` | CUDA-graph API microbenchmarks: what can be changed in an instantiated graph, and what each change costs |
| `survey/` | early coverage survey: why Inductor partitions or refuses graphs across ~100 vision/HF models. Data not included; regenerate with the scripts |
| `verification/` | one-off verification scripts for library behaviour inside captured graphs (cuBLAS, cuDNN, ATen, DeepGEMM). They back `docs/notes/EXTERN.md` |
| `third_party_patches/` | optional DeepGEMM patch that adds a describe entry point (tier-3 example). It modifies a third-party library, so it is not the default path |

The `.lintrunner.toml` excludes `dynagraph/**`: these are research scripts, not library code. Lint
`torch/` changes as usual (`lintrunner -a`) before committing.

## Open directions

These are starting points that came up during the work. None of them is settled.

- **Find the workloads.** List workloads where shapes change often and the same shape repeats
  rarely, so per-shape recording does not amortize; where padding is not free; and where per-model
  engineering is expensive. Candidates touched so far are listed in docs/MEASUREMENTS.md
  (HF encoders with random (B, L), protein LMs, GNN/MD, LLM mixed batches, diffusion DiTs,
  speculative decoding verify, elastic parallelism). None has been studied exhaustively, and
  multi-GPU (TP shards) has not been measured at all. The `e2e/` bucket baseline is a stand-in
  (one static compile per power-of-two bucket); a vLLM-style baseline (one `dynamic=True` compile,
  one graph per capture size) has not been run against DG on these workloads.
- **Library GEMMs as fixed variants.** Instead of harvesting cuBLAS per exact M, capture it at
  bucketed M (rows are independent, so padding M only needs buffer capacity) and slice the output.
  Measured: padding M to a multiple of 128 costs 1.01x to 1.05x GPU time over 17 sizes
  (`e2e/probes/probe_gemm_pad.py`), versus 1.23x to 1.34x for powers of two. This would remove most
  tier-2 harvests.
- **Cut the per-call cost.** The host patcher already skips nodes whose parameter bytes did not
  change, but on a shape change grids and pointers move, so nearly every node is touched (see the
  end of `docs/notes/BENCH.md`). The open part is moving sizes into device memory: kernels read them
  from a small buffer and launch a max grid with early exit, so a call costs one memcpy instead of
  N `SetParams`.
- **Graph breaks and non-Inductor code.** Serving frameworks (SGLang) avoid `torch.compile` and
  capture eager code in segments ("breakable CUDA graphs"). The DG idea, patching instead of
  re-capturing, could apply at that level too. See docs/notes/DIFFUSION.md for what SGLang does
  and which PT2 issues blocked the automatic path there.
- **Attention inside the graph.** Serving systems break the graph at attention because its
  metadata (varlen lengths, paged KV) changes per batch. A declared-launch (tier-3) attention kernel
  could keep it in the graph.
- **Cold start and memory.** Count compiles, captures and graph memory against bucketing, for
  example 30 compiles for 2-D (B, L) buckets versus 1.
