# Working notes (translated)

These are the project's working notes, written as the work happened between 2026-09-14 and
2026-09-29 and translated from Chinese. They are logs, not specifications:

- Later sections often correct earlier ones, and some early numbers were measured with bugs that
  were fixed later. When a note and the code disagree, trust the code. When two notes disagree,
  trust the later date.
- `../MEASUREMENTS.md` has the current numbers and `../METHODOLOGY.md` has the measurement lessons.
  Read those first and come here for the history and reasoning behind a design decision.
- Card numbers ("GPU 6, exclusive") record which H100 a run used on the shared machine. Rules in
  the notes about which cards may be used, and SETUP.md's container recipe (`--privileged`), are
  historical; follow the top-level README and `../METHODOLOGY.md`.

## Start here

| note | what it covers |
|---|---|
| [FEASIBILITY.md](FEASIBILITY.md) | The original proposal: glossary, motivation (serving and badly written open-source models), the first survey data, opaque kernels (GEMM vs attention), incremental update mechanisms, system design, novelty and the list of hard problems. Long; the glossary and TL;DR are the useful entry points. |
| [SETUP.md](SETUP.md) | How the dev environment was built (venv on top of the NVIDIA container, version alignment, build flags) and the traps hit along the way: GEMM backend deps, containers losing GPUs, building third-party CUDA extensions against this torch. |

## Design and implementation

| note | what it covers |
|---|---|
| [SOLUTION.md](SOLUTION.md) | Implementation log: where DG hooks into PyTorch, the first working end-to-end run, record-at-max and arena relayout, integration with cudagraph trees, cross-shape verification and the stress probes that found bugs. |
| [TIERS.md](TIERS.md) | When the device-side planner is needed and when host-side patching is enough ("where does the value live"), how to make the planner cheap, and the mechanisms for topology changes (SWITCH, linear pre-placement + SetEnabled, why ExecUpdate is not a substitute). |
| [EXTERN.md](EXTERN.md) | Extern kernels (cuBLAS, cuDNN, NCCL, custom ops): what measurement ruled out, the tiered design, child-graph harvesting, SWITCH for topology changes, multi-GPU regions with all_reduce, training regions and non-contiguous inputs. |
| [REGISTER.md](REGISTER.md) | Tier 3: what a library has to declare (variant key, prepare, plan, bind) so its launches can be inlined into the main graph; the C describe ABI and the inventory of launch types. |
| [ENGINE.md](ENGINE.md) | Plugging DG into vLLM's decode path: four-way comparison, per-forward node-patch latency, a lifetime bug that looked like GPU overhead, and checking the annotation approach against real serving kernels. |

## Measurements and workloads

| note | what it covers |
|---|---|
| [FINDINGS.md](FINDINGS.md) | The 125-model survey (2026-09-17): methodology corrections, attribution, list representativeness, why "number of shapes" is the wrong statistic, and how often CUDA graphs help at all. |
| [BENCH.md](BENCH.md) | The first micro-benchmarks of DG itself (launch-bound vs GPU-bound transformer stacks), per-call host cost, planner cost, and the cost of partitioning around fallbacks. |
| [MESSY.md](MESSY.md) | Scans of AI paper repositories ("badly written code"): which coding patterns DG can serve and which it cannot, re-scanned after each fix. |
| [E2E.md](E2E.md) | End-to-end workloads (GraphSAGE, SchNet, MACE, point cloud, ESM-2): the harness, status per workload, upstream bugs and workarounds. |
| [DIFFUSION.md](DIFFUSION.md) | SGLang diffusion serving: breakable CUDA graphs (BCG) vs compile vs the automatic PT2 path, varying prompt lengths, and every place BCG is used in SGLang. |

## Workload search and related work

| note | what it covers |
|---|---|
| [WORKLOAD.md](WORKLOAD.md) | First search for an elastic-serving workload: what to look for and which traces to use. |
| [WORKLOAD2.md](WORKLOAD2.md) | Second search: tree verification in speculative decoding, elastic parallelism frequency, pure decode, and the HF-model zoo with a new shape every step. |
| [ELASTIC.md](ELASTIC.md) | Elastic parallelism (shape axis vs parallel-width axis): how SGLang elastic EP and vllm-omni handle it today, and what DG could cover. |
| [UPSTREAM.md](UPSTREAM.md) | Comparison with Elias Ellison's parametrized CUDA graph runtime checkpoint (2026-09-14). |

## Old paths in these notes

The notes refer to the layout used while the work was done. In this directory:

| in the notes | now |
|---|---|
| `plan/<NAME>.md`, `plan/README.md` | `docs/notes/<NAME>.md`, `docs/notes/FEASIBILITY.md` |
| other top-level `dynagraph/*.py` (`test_*`, `probe_*`, `bench_*`, ...) | `probes/` |
| `dynagraph/_wf_*` | `verification/` |
| `dynagraph/_regress_quick.sh` | `probes/regress_quick.sh` |
| `dynagraph/e2e/*.py` | `e2e/` (`e2e/_probe_*.py` is now `e2e/probes/probe_*.py`) |
| `dynagraph/probe_vllm_*.py`, `dynagraph/bcg_bench.py`, `dynagraph/bcg/` | `serving/` |
| `microbench/`, `survey/` | `microbench/`, `survey/` |
| `/workspace/use_main.sh`, `/workspace/build_torch.sh` | `setup/` |
| `/workspace/pytorch-main` | the root of this fork |
| containers `xinwei_autocudagraph`, `xinwei_dflash`, `xinwei_bcg` | the container from the top-level README (called `dynagraph` there); `xinwei_bcg` ran SGLang |
| `_deps/patch_dgd.py`, `_deps/patch_dgd_c.py` | `third_party_patches/` |

Some notes link to files that were not carried over: logs, datasets, a few one-off scratch
scripts, and an earlier summary document.
