# DynaGraph microbenchmarks

Standalone CUDA and Python experiments from 2026-09-17, written before DynaGraph existed in
PyTorch. They are the measurements behind the feasibility study in
[../docs/notes/FEASIBILITY.md](../docs/notes/FEASIBILITY.md): what can be changed in an
instantiated CUDA graph, and what each change costs.

Sources only; "Running them" below shows how to build and run them.

## What each one measures

| File | Question it answers | Result on 2026-09-17 |
|---|---|---|
| `devupdate_test.cu` | Does a planner kernel's `cudaGraphKernelNodeSetGridDim`/`SetParam` on a downstream node take effect in the *same* graph launch? | Yes. 4 values of L all correct, no `cudaGraphUpload` needed per launch. |
| `devupdate_scale.cu` | What does patching N nodes per launch cost? | 3000 nodes = +148 us/launch, 0 errors. |
| `switch_test.cu` | Can a SWITCH conditional node be inserted mid stream-capture with the condition set on device? | Yes, works. |
| `cond_cost.cu` | What does a SWITCH node cost, and does it scale with body size? | ~9 us per SWITCH, independent of body length. Layer-granularity only. |
| `triton_ctx_test.py` | Baseline: Triton kernel reading its pointers/sizes from a device ctx buffer instead of kernel args. | Works; establishes the ctx-indirection pattern. |
| `triton_ctx_test2.py` | Pointer-alignment hints placed after the int64->pointer cast. | Needed to keep vectorized loads. |
| `triton_ctx_test3.py` | ctx-indirection + persistent grid, correctness across sizes and addresses. | Correct across 5 size/address combos. |
| `triton_retime.py` | Timing cost of ctx-indirection + persistent grid vs direct args. | +0.2 us on tiny kernels, +5% on large ones. |
| `inductor_vec_check.py` | Do Inductor `dynamic=True` pointwise kernels already lose vectorized loads? | Yes, no `ld.global.v4` already, so ctx costs nothing extra there. |
| `cublas_variants.py` | How many distinct cuBLAS kernels does one bf16 GEMM shape select over a range of M? | 61 distinct nvjet kernels over 361 M values. **Undersampled**: a dense M=1..4096 scan on 2026-09-19 found 94 kernels, 3 cluster shapes, interleaved (357 changes between adjacent M). Kills "enumerate variants + SWITCH" for GEMM -- but `cudaGraphExecUpdate` / child-graph replacement swaps func+cluster without enumerating, so it is no longer fatal. |
| `planner_batched.cu` | Is `cudaGraphKernelNodeUpdatesApply` cheaper than separate SetEnabled/SetGridDim/SetParam calls? And how much does DynaGraph's per-node `switch` cost? | **No.** At N=48: separate 0.083 us/node, batched 0.099. Per-node switch 0.145, nested switch (dg_eval inside) 0.244. All of it ~4-12 us at N=48 -- the real planner's 43.7 us is the *count* of per-argument calls plus divergence, not any one API. Killed the "batching = 18x" lead from devupdate_scale.cu. |
| `planner_cta.cu` | Do more threads / more CTAs make the planner's device-update calls cheaper? Same 48x6 calls in three layouts. | **No.** A (1 thread/node, 6 sequential calls) 4.9 us, B (6 threads/node, 1 block) 2.9, C (6 threads/node, 48 blocks) 2.9. B == C: calls serialize in the device runtime, spreading over SMs buys nothing; threads only break the per-thread chain (1.7x). And all 288 calls are ~5 us -- the real planner's 43 us is not the call count of this shape. |
| `planner_real.cu` | Time a *dumped* DynaGraph planner cubin (`dynagraph/_dump_bench_planner.py`, not included in this repo) on dummy nodes, so it can be bisected. | 48 nodes, 143 SetParam (95 pointer patches), 952-byte stack frame: full 53.8 us, without pointer patches 26.1, without scalar params 36.8, grid only 12.9. Pointer patches are half -> fix slot offsets at build so pointers are written once. |
| `host_update.cu` | What a host-side patcher pays per call: `cudaGraphExecKernelNodeSetParams` per node, whole-graph `cudaGraphExecUpdate`, child swaps, a launch. | 48 nodes: 24.7 us (0.52/node); edit graph + ExecUpdate 23.9; 24 child swaps 10.6 (0.44 each; 2.4 from Python); launch 1.6 us host. Host patching is linear in nodes but overlaps the GPU. |
| `switch_patch.cu` | Can nodes *inside* a SWITCH body be patched after instantiation? | **Yes, both ways.** Device-side (planner sets the conditional and patches the chosen body's grid in the same launch) and host-side (`cudaGraphExecKernelNodeSetParams` on a body node). When both are used, the device patch runs later and wins. Topology variation can therefore live inside one graph. |

## Running them

They were run in a container of the same image as the top-level README (`nvcr.io/nvidia/pytorch:26.02-py3`:
nvcc 13.1, the container's own torch 2.11 and triton 3.6), without the self-built torch.
nvcc is not installed on the host.

Inside the container the repo is mounted at `/workspace/pytorch-main`, so this directory is
`/workspace/pytorch-main/dynagraph/microbench`. Build outputs go to `$DG_OUT`
(default `/tmp/dynagraph_out`) so the source tree stays clean.

CUDA files need relocatable device code and the device runtime:

```bash
docker exec dynagraph bash -c \
  'cd /workspace/pytorch-main/dynagraph/microbench && OUT="${DG_OUT:-/tmp/dynagraph_out}" && mkdir -p "$OUT" && nvcc -O2 -arch=sm_90 -rdc=true devupdate_test.cu -o "$OUT/devupdate_test" -lcudadevrt && "$OUT/devupdate_test"'
```

Python files need `TRITON_CACHE_DIR` set when they inspect generated PTX
(`triton_ctx_test.py`, `inductor_vec_check.py`). `closure_check.py` sets its own,
under `$DG_OUT/closure_triton`.

## Before trusting any timing number

All 8 H100s on this box are capped at 550 W against a 700 W default and spend roughly
half their time power-limited, so SM clocks vary by up to 30% between cards and over
time. Check the target card first:

```bash
nvidia-smi --query-gpu=index,power.limit,power.draw,clocks.sm,clocks_event_reasons.sw_power_cap --format=csv
```

The original runs used GPU 7. The box is shared, so ask before changing a power limit.
