# Measurements so far

Measurements through 2026-10-15, with setups and commands, so they can be reproduced, extended
or challenged. The working notes in `notes/` have the full history, including dead ends and fixed
bugs. The code changed a lot during these two weeks, so each table says when it was measured, on
which code and with which GEMM backend. Where a workload was not re-measured on the final code, the
latest numbers are given and marked as such.

## Conditions

- 8x H100 80GB HBM3 (sm_90a). The September measurements below used a **550 W power cap**
  and were power-limited a large fraction of the time. Later dated sections state their own
  conditions; the cuBLAS bucket experiment on 2026-10-15 used 700 W.
- Self-built PyTorch main `2.15.0a0+git71b3251` plus the `dynagraph` branch. CUDA 13.1, NVIDIA
  container `nvcr.io/nvidia/pytorch:26.02-py3`.
- Shared machine. A card was considered free when no other process was on it, but the host CPU
  usually carried other people's load. **Ratios are more reliable than absolute numbers.** Each
  table says which card and conditions were used.
- Unless stated otherwise, timings sync the GPU before and after every step. This measures
  single-request latency, where host and GPU time add up. With `NOSYNC=1`, the harness syncs only at
  segment ends, so the host work of step n+1 overlaps the GPU work of step n. Only means are
  available in that mode.

## Baselines used everywhere

| name | what it is |
|---|---|
| eager | no compilation |
| compile | `torch.compile(dynamic=True)`, no CUDA graphs |
| trees | `torch.compile(dynamic=True, mode="reduce-overhead")`: upstream cudagraph trees, one recorded graph per distinct shape (warmup on first sight, record on second) |
| pad | every batch padded to the global maximum shape, `dynamic=False` + reduce-overhead: one static graph |
| bucket | every dimension padded up to the next power of two, one static (`dynamic=False`) compile and one graph per bucket. This is a stand-in for what serving frameworks do by hand; see the note below |
| oracle | the compile(dynamic) code, one CUDA graph captured per shape, timing **replay only** |
| DG | DynaGraph: `reduce-overhead` + `triton.dynagraph=True` |

The oracle has no Dynamo guard evaluation, no AOT/Inductor wrapper and no recording cost, so no
`torch.compile`-based scheme can reach it. Treat it as a floor, not a target. On BERT, about
0.35 ms per step of host work outside DynaGraph separates any compiled scheme from it (see
"Per-call cost" below).

The bucket baseline here differs from the real thing. vLLM compiles once with dynamic shapes and
records one graph per capture size ([1, 2, 4] + range(8, 256, 8) + range(256, max, 16), capped at
512). SGLang's breakable CUDA graphs do not compile at all. Neither of those was run as a baseline
for the `e2e/` workloads; for serving, see the vLLM and SGLang sections below.

## Per-call cost of DynaGraph

BERT-base, bs=1, L in [16, 512], GEMMs through Triton templates, eager attention. cProfile over 80
replay steps, GPU 3, 2026-09-25 (code at `f0e7639038`):

| component | per step |
|---|---|
| GPU execution of the patched graph | ~1.04 ms (same as the oracle) |
| whole host path into the compiled function | ~0.52 ms |
| - Dynamo frame + guard evaluation | ~0.15-0.2 ms (every compiled mode pays this) |
| - AOT + Inductor wrapper | ~0.05 ms (every compiled mode pays this) |
| - DG `__call__` (key, layout, node patches, outputs, launch; C++ runtime) | ~0.15 ms, dominated by per-node `cuGraphExecKernelNodeSetParams` (~0.57 us x ~200 nodes) |
| - not attributed by the profile | ~0.12-0.17 ms |

Unit costs measured on ESM-2 (`e2e/proto_units.py`):

| operation | cost |
|---|---|
| `SetParams` from C++ | 0.57 us/node |
| output tensor built on the arena | 1.3 us |
| `cudaGraphLaunch` | ~2 us |
| DeepGEMM describe (Python) | 8.7 us |

Host-path history on ESM-2 35M (DG `__call__` per-step median, new / seen shapes). These runs used
DeepGEMM through the local describe patch (`GEMM=deepgemm`, tier 3), 2026-09-24, GPU 7 shared; see
`notes/E2E.md` sections 6.5-6.7:

| stage | new shapes | seen shapes |
|---|---|---|
| after the correctness fixes | 40.6 ms | 19.7 ms |
| Python-side optimizations (signature grouping, batched node updates, compiled layout exprs; `677ef29e48`) | 20.7 ms | 4.1 ms |
| per-call path moved to C++ (`dynagraph_rt.cpp`; `82f1e4bb3e`) | 10.1 ms | 2.9 ms |

## HF models with a new shape every step (`e2e/zoo.py`)

These are the HuggingFace models from GraCE (OSDI'26), randomly initialized, bf16 inference. Every
step draws a new (B, L), with L log-uniform in [16, 512]. cuDNN SDPA is disabled (see METHODOLOGY:
it rebuilds a plan per new shape). 100 steps: 20 warm, 80 timed. GPUs 3-5, mostly exclusive.
Measured 2026-09-25 on code `f0e7639038`.

**bs=1, per-step median (ms), cuBLAS GEMM + SDPA:**

| model | compile | trees | pad to 490 | bucket (6 buckets) | oracle |
|---|---|---|---|---|---|
| bert | 4.32 | 7.17 | 1.96 | 1.49 | 1.05 |
| distilgpt2 | 1.53 | 4.04 | 1.41 | 0.92 | 0.63 |
| xlnet | 2.88 | 6.74 | 1.86 | 1.36 | 1.15 |
| t5 | 3.60 | 7.61 | 1.99 | 1.49 | 1.05 |
| mobilebert | 11.27 | 29.7 | 4.80 | (5.55, contended run) | 3.08 |
| debertav2 | 4.98 | 10.2 | 2.68 | - | 2.33 |
| blenderbot | 5.33 | 10.8 | 2.97 | 2.32 | 1.62 |
| albert | 2.94 | 6.41 | 1.45 | 1.04 | 0.82 |
| electra | 2.67 | 7.21 | 1.42 | 1.19 | 0.77 |

**B in {1,2,4,8,16,32}, L in [16, 512], per-step mean (ms) over new shapes.** Padding to (32, 511) is
10.4x the mean token count.

| model | compile | trees | pad | oracle mean / median | bucket: seen-shape median (29 buckets, 30 compiles) |
|---|---|---|---|---|---|
| bert | 4.60 | 11.41 | 14.87 | 2.17 / 1.22 | 2.00 (warm 97 s) |
| distilgpt2 | 3.05 | 4.79 | 10.38 | 1.51 / 0.91 | 1.46 (warm 54 s) |
| xlnet | 4.73 | 8.71 | 27.58 | 3.06 / 1.40 | 1.87 (warm 170 s) |
| t5 | 4.42 | 8.92 | 19.42 | 2.34 / 1.21 | 1.98 (warm 157 s) |
| mobilebert | 14.87 | 36.84 | 11.86 | 3.65 / 3.21 | - |
| blenderbot | 5.66 | 10.45 | 19.98 | 3.04 / 1.86 | - |
| albert | 2.75 | 5.42 | 11.67 | 1.61 / 0.93 | - |
| electra | 3.61 | 7.14 | 5.44 | 1.14 / 0.84 | - |

The bucket column is a median over seen shapes, unlike the other columns (means over new shapes).
In the 2-D setting, buckets keep compiling during the timed segment as new buckets appear, so
bucketing's mean per-step time over new shapes is 0.7-2.4 s, mostly compile time.

**DynaGraph on BERT:**

- bs=1, cuBLAS GEMMs (tier 2): new-shape step 14.7 ms (each new shape harvests cuBLAS child graphs;
  256 new topologies logged); seen shapes 1.78 ms.
- bs=1, Triton GEMM + eager attention (no harvest; per-step median):

| mode | new shapes | seen shapes |
|---|---|---|
| compile | 1.58 | 1.57 |
| DG | 1.50 | 1.49 |
| bucket (6) | 1.39 | 1.39 |
| oracle | - | 1.05 |

- B in {1..32}, Triton GEMM + eager attention. GPU 5 was exclusive; the host was loaded.
  2026-09-25, code `f0e7639038` plus the LRU fix committed later (the run log is not included).

| mode | compiles / warm time | sync: new median / mean | sync: seen median | no sync: per-step mean (new / seen) |
|---|---|---|---|---|
| compile | 1, 111 s | 2.46 / 3.58 | 2.92 | 3.06 / 3.07 |
| bucket (29) | 30, 292 s | 5.63 / 4035 (still compiling new buckets) | 2.69 | - / 2.92 |
| DG | 1, 114 s | 2.27 / 3.39 | 2.28 | 2.93 / 2.83 |
| oracle | - | - | 1.98 (mean 2.82) | - |

DG served all 182 calls (0 fallbacks), and its outputs matched compile exactly. The step-time
distribution is skewed (most steps are small; a few have B=16/32 and long L), so compare medians
with medians and means with means.

Commands:

```bash
AMP=1 python e2e/zoo.py --model bert --bs 1 --modes compile,trees,pad,bucket,oracle,dg
AMP=1 python e2e/zoo.py --model bert --bs 1,2,4,8,16,32 --modes compile,bucket,oracle,dg
GEMM=triton ATTN=eager AMP=1 python e2e/zoo.py --model bert --bs 1,2,4,8,16,32 --modes compile,dg,bucket,oracle
GEMM=triton ATTN=eager AMP=1 NOSYNC=1 python e2e/zoo.py --model bert --bs 1,2,4,8,16,32 --modes compile,dg,bucket
GEMM=triton ATTN=eager DGPROF=replay AMP=1 python e2e/zoo.py --model bert --bs 1 --modes dg    # cProfile of the replay segment
```

## Training and scientific workloads (`e2e/`)

Per-step median in ms over the new-shape segment, then the replay segment where two numbers are
given. These workloads were measured at different stages of the code and with different GEMM
backends, so compare within a row, not across rows. Two numbers separated by "|" are two rounds
with the mode order reversed. Full tables are in `notes/E2E.md` section 6 and
`notes/WORKLOAD2.md`.

| workload | data | when, code, GEMM | eager | compile | trees (new) | DG new / replay | pad |
|---|---|---|---|---|---|---|---|
| GraphSAGE (`sage.py`) | ogbn-arxiv neighbor sampling, 88 distinct shapes | 2026-09-23, Python host path, DeepGEMM, GPU 0 | 3.01 \| 2.95 | 1.63 \| 1.34 | 5.41 \| 4.16 | 2.09 / 1.14 \| 2.54 / 1.22 | 0.94 \| 0.83 |
| GraphSAGE | same | 2026-09-24, C++ runtime (`82f1e4bb3e`), DeepGEMM | - | 2.17 | - | 1.61 / - | - |
| SchNet (`schnet.py`) | QM9, 80 distinct shapes | 2026-09-23, Python host path, DeepGEMM, GPU 0 | 10.47 \| 10.12 | 7.22 \| 11.56 | 10.42 \| 11.68 | 8.38 / 5.24 \| 8.33 / 5.16 | 2.89 \| 2.92 |
| point cloud (`pointcloud.py`) | KITTI drive 0093, 80 frames | 2026-09-23, Python host path, DeepGEMM, GPU 2 (shared with an idle simulation job) | 2.80 | 0.60 | 2.35 | 1.21 / 0.68 | 1.28 |
| MACE (`mace_md.py`) | MD22 double-walled nanotube | 2026-09-24/25, Triton GEMM, GPU 4 | - | 4.46 | - | 4.41 / 4.35 | 3.94 |
| ESM-2 35M (`esm.py`) | UniProt human proteome, token-budget batches, (B, L) both vary, 87 distinct | 2026-09-24, C++ runtime (`f0e7639038`), Triton GEMM, GPU 7 | - | 17.4 / 17.2 (seen) | - | 20.4 / 18.9 | - |
| ESM-2 35M | same | 2026-09-23/24, DeepGEMM | 41-50 | 19.9-20.4 | 51.8 (139.9 while recording) | - | 146 |

Notes:

- GraphSAGE and SchNet: graph-size max/mean is only ~1.07, so padding wastes little, and SchNet's
  static-shape kernels are faster than its dynamic-shape kernels at the same shape (pad 2.9 ms vs
  DG replay 5.2 ms).
- Point cloud: after compilation it runs 49 kernels with 0.41 ms of GPU time per step.
- MACE: GPU time is 3.39 ms per step.
- ESM-2: 14.6 ms of GPU time and 864 kernels per step. About 10 ms of host time per step is outside
  the compiled regions (HF eager code between graph breaks, autograd, the optimizer), where DG has no
  effect.
- ESM-2 with DeepGEMM on the C++ runtime (`82f1e4bb3e`): DG 23.0 / 18.1 ms against compile
  26.7 / 26.0 ms. Most of that difference was the custom op's per-call Python dispatch (216 GEMM
  calls per step), which compile pays and DG skips, rather than graph patching. With Triton GEMMs
  (the row above) that dispatch does not exist.

Commands (`e2e/timing.sh` and `e2e/timing2.sh` run the first four):

```bash
cd /workspace/pytorch-main/dynagraph/e2e
export PYTHONPATH=$DG_DEPS/deepgemm-src:$DG_DEPS/mace_site:$PYTHONPATH
AMP=1 GEMM=deepgemm python sage.py --batches 100 --warm 20 --modes eager,compile,trees,dg,pad
AMP=1 GEMM=deepgemm python schnet.py --batches 80 --warm 20 --modes eager,compile,trees,dg,pad
GEMM=triton python mace_md.py --npz $DG_DATA/MD22/md22_double-walled_nanotube.npz --frames 100 --warm 20 --modes compile,dg
AMP=1 GEMM=triton python esm.py --modes compile,dg
```

## LLM serving

**vLLM mixed prefill+decode batches** (`serving/probe_vllm_mixed.py`): Qwen3, max_num_batched_tokens
2048, 256 ShareGPT requests with chunked prefill, so the token count changes every step. Wall time
for the whole run, two rounds:

| cudagraph mode | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| NONE | 6.95 / 6.56 s | - |
| PIECEWISE (attention eager between captured pieces) | 4.92 / 4.37 s | - |
| FULL_AND_PIECEWISE (vLLM default) | 2.48 / 2.10 s | 5.74 / 5.84 s |
| FULL (mixed batches also in one graph, padded to capture sizes) | 1.89 / 1.88 s | 5.77 / 5.75 s |

On 8B, FULL used 4.44 GiB of graph memory, against 1.30 GiB for the default mode.

Pure decode: padding is at most 7 rows below bs 256 (15 above), and decode is bound by weight reads.
This was estimated, not measured. Tree verification in speculative decoding (SGLang EAGLE, TRT-LLM)
uses a fixed per-request token budget, so shapes stay batch x N even when tree shapes differ. See
`notes/WORKLOAD2.md`.

## Diffusion serving in SGLang (`serving/bcg_bench.py`)

These runs use random ("fake") weights generated from the real checkpoints' safetensors headers by
`serving/fake_hf_repo.py`. SGLang main `2a1c477`; local knobs from
`serving/sglang_experiment_knobs.patch`. End-to-end request time (text encoder + DiT + VAE), H100.
Details are in `notes/DIFFUSION.md`.

**SANA 1.5 1.6B, 20 steps, CFG 4.5:**

| config | 1024^2 (seen) | 512^2 (seen) | 512^2 first time | 768x1344 (never warmed up) |
|---|---|---|---|---|
| eager | 0.89-0.97 s | 0.79 s | - | 1.64 s |
| breakable CUDA graph (BCG) | 0.51-0.53 s | 0.26 s | 0.26 s (pre-warmed) | 1.68 s (falls back to eager) |
| SGLang default compile (max-autotune, shape-specialized) | 0.50-0.53 s | 0.41 s | 14.9 s (recompile) | 6.0 s |
| compile, dynamic=True | 0.48-0.52 s | 0.45 s | 22.5 s | 19.3 s |
| PT2 reduce-overhead + 5 small fixes (below) | 0.50-0.56 s | 0.28 s | 0.91 s | 1.31 s |
| same fixes, no graphs | 0.52-0.55 s | 0.42 s | 1.04 s | 1.32 s |

Plain `reduce-overhead` crashes on this code. Five fixes, about 25 lines in total, were needed to
run SGLang's SANA through the automatic PT2 path:

1. Register a tvm_ffi JIT kernel (`timestep_embedding`) as a custom op; it was the only graph break.
2. Call `cudagraph_mark_step_begin()` before each DiT call.
3. Clone the DiT output, because CFG replays the same graph twice and keeps both outputs.
4. Disable duck sizing. Channels, H and W all equal 32 at 1024^2, so they share one symbol, and a
   conv guard pins it to a constant.
5. Do not compile the VAE: it compiles shape-specialized, 18-23 s per new resolution.

With the prompt length differing on every request (16 or 40 requests, 4-160 words), measured
end-to-end at 512^2:

| | median | max | memory growth over 40 requests |
|---|---|---|---|
| BCG (prompts padded to 5 length buckets) | 0.26 s | 0.31 s | +8 MiB |
| PT2 reduce-overhead + fixes (one graph per prompt length) | 0.28 s | 0.37 s | +58 MiB (~1.5 MB per graph) |
| compile, no graphs | 0.42 s | 0.45 s | - |

**Z-Image-Turbo 6B, 9 steps, no CFG:** 512^2 runs 0.39 s eager and 0.38 s with BCG. 1024^2 runs
1.11 s eager and 1.08 s with BCG. Host submission (~25 ms/step) and GPU time (~24 ms/step) are about
equal on H100.

**SANA DiT alone in this harness** (`e2e/sana.py`: diffusers `SanaTransformer2DModel`, random
weights, 24 requests x 20 steps, CFG batch 2, one of SANA's aspect-ratio bins per request). Per-request
time in ms; the diffusers model has no graph breaks:

| mode | 512 bins: new | 512 bins: seen | 1024 bins: new | 1024 bins: seen |
|---|---|---|---|---|
| eager | 648 | 540 | 709 | 608 |
| compile(dynamic) | 197 | 197 | 363 | 365 |
| trees | 205 | 184 | 370 | 357 |
| DG | 232 | 185 | 427 | 362 |
| pad | 295 | 296 | 1053 | 1064 |
| oracle | 168 | - | 338 | - |

DG's new-shape cost here comes from harvesting cuBLAS GEMMs and cuDNN depthwise convs per shape:
449 and 470 new topologies were logged. DG served 901 of 901 calls. Peak memory: compile and trees
6.2 GB, DG 11.1 GB (512 bins) and 35.6 GB (1024 bins), pad 15.6 / 39.9 GB. 2026-09-28, GPUs 6/7
exclusive.

## Elastic parallelism (source reading, not measured)

| system | CUDA graphs | what changes per step per GPU | how graphs cope |
|---|---|---|---|
| LoongServe (SOSP'24) | none: `synchronize` in every prefill layer, a Ray RPC per step, baselines run with `enforce_eager` | role, slice size, peer set, number of NCCL send/recv | n/a; the graph structure itself would change |
| NanoCP (arXiv 2605.21100, no code) | 2-D bucketed table, routing in device memory | query and result counts, routing | 48 graphs, 5.32 GiB per GPU (vLLM 32DP: 1.02 GiB) |
| Shift Parallelism (ArcticInference) | two FULL-graph families (TP and SP), switched at 512 tokens | mode | pre-captured, never re-captured |

See `notes/ELASTIC.md` and `notes/WORKLOAD2.md`.

## Library behaviour that matters for any dynamic-shape scheme

- **cuDNN SDPA** (PyTorch main's default attention on H100) builds an execution plan for every new
  shape, ~5.4 ms per layer. BERT spends 65 ms on its 12 layers at a new shape, while the whole step
  is 1.66 ms of GPU time.
- **cuBLAS** picks different kernels (and node topologies, e.g. split-K) as M changes. Padding M is
  cheap. Summed over 17 values of M from 1 to 16000 on BERT-base GEMM shapes (bf16, GPU time only;
  `e2e/probes/probe_gemm_pad.py`):

| GEMM | pad M to a multiple of 128 | pad M to a power of two |
|---|---|---|
| qkv/o 768x768 | 1.02x | 1.23x |
| ffn1 768x3072 | 1.05x | 1.34x |
| ffn2 3072x768 | 1.01x | 1.24x |
| lm head 768x30522 | 0.99x | 1.25x |

  The exception is very small M: padding 7 rows to 128 costs 1.54x on the LM head.
- **VAE / conv first-time cost:** cuDNN chooses a convolution algorithm the first time it sees a
  shape, ~0.5-0.7 s for SANA's VAE decoder at a new resolution.

## Not measured yet

- **Multi-GPU workloads.** TP shrinks every kernel; GraCE reports its largest gains at TP4. Only
  single-feature probes exist here (NCCL inside a region, a DDP step).
- **The 2-D DG comparison for other models.** It was run only for BERT; T5 and distilgpt2 are not
  done.
- **Graph memory.** DG versus buckets versus per-shape graphs at scale.
- **DG on code with graph breaks between custom kernels** (SGLang-style serving code).

## Per-GEMM cuBLAS buckets (2026-10-15)

**Capture reduction works; a latency improvement over existing signature cloning is not
reproducible on BERT. Leave the feature opt-in.** Bucketing improves new-shape median latency
relative to DG without cloning, adds GPU work, and remains much slower than compile without
CUDA graphs on new shapes. Whole-model power-of-two buckets also remain a stronger baseline
for this bounded, one-dimensional shape space.

Setup: otherwise idle GPU 3, H100 80GB HBM3, **700 W**, self-built torch
`2.15.0a0+git2a6bdba` (HEAD `2a6bdbab528`, with the local bucket implementation), CUDA 13.1,
Triton 3.6.0. BERT uses Transformers 5.10.4. No pretrained weights were downloaded. Other GPUs
and the host were shared. Every A/B comparison was on this one card, with modes rotated and
order reversed across shapes, synchronization before/after each timed call, and compilation
excluded. Runtime new-shape verification was disabled for timing; every new output was checked
against eager outside the timer. No DG fallback or timed-segment recompile occurred. Maximum
relative L2 against eager was 0.0072 for the MLP and 0.0142 for BERT, including all compiled modes.

Workloads:

- MLP: 12 residual blocks, each LayerNorm(768), Linear(768,3072), GELU,
  Linear(3072,768), residual. 24 cuBLAS sites. BF16, varying row count M.
- BERT: HuggingFace `BertModel`, 12 layers, hidden size 768, intermediate size 3072,
  no pooler or MLM head, random BF16 weights, batch 1. 72 cuBLAS sites and 12
  `aten._scaled_dot_product_efficient_attention` sites. cuDNN SDPA disabled.
- Warm at M/L=512, then 40 distinct integers sampled without replacement from [33,511],
  seed 19 (actual range 44..504), followed by five passes over those shapes. The focused
  BERT repeat uses the first 24 shapes, and finishes trees' second-visit recordings before
  timing replay. Inputs already reside on the GPU.
- `DG + clone` enables existing `dynagraph_clone_sites` without padding. Bucket modes enable
  only the new bucket-site cloning. All DG variants use host updates and disable pointwise
  autotuning. `whole-model pow2` uses one dynamic compile and upstream trees with
  pre-padded inputs: **input-padding preparation is excluded**, an optimistic baseline.

### MLP, 40 new shapes

Median milliseconds; GPU time is direct replay of the prepared DG graph, excluding host updates
and input staging. It includes the new feature's padding/output kernels. GPU samples cover the
same eight shapes in every DG mode, so their medians are comparable to each other, not directly
to the wall-clock medians over all 40 shapes.

| Mode | New-shape wall | Replay wall | DG graph GPU |
|---|---:|---:|---:|
| compile, no graphs | 0.926 | 0.616 | - |
| DG exact | 4.886 | 0.475 | 0.252 |
| DG exact + clone | 3.814 | 0.468 | 0.254 |
| DG bucket 32 | 3.450 | 0.536 | 0.303 |
| DG bucket 64 | 3.288 | 0.518 | 0.301 |
| DG bucket 128 | 3.148 | 0.513 | 0.295 |
| DG bucket 256 | 3.323 | 0.514 | 0.332 |
| whole-model pow2 | 0.691 | 0.451 | - |

At width 128, six new cuBLAS captures serve the three previously unwarmed buckets (two GEMM
signatures per bucket), versus 960 captures for exact DG. New-shape median is 36% lower than
exact DG and 17% lower than DG + clone, but replay is 8% slower than exact DG and graph GPU
execution is 17% slower. The paired per-shape median speedups are 1.49x and 1.16x, respectively.
New-shape means: exact DG 6.476 ms, DG + clone 5.504 ms, bucket 128 4.051 ms.

### BERT, focused repeat, 24 new shapes

Wall-clock milliseconds. Capture counts cover only the new segment and exclude warmup.

| Mode | New median | New mean | Warm replay median | cuBLAS captures |
|---|---:|---:|---:|---:|
| compile, no graphs | 3.209 | 3.169 | 2.553 | - |
| DG exact | 21.523 | 31.592 | 1.548 | 1728 |
| DG exact + clone | 16.502 | 52.504 | 1.423 | 72 |
| DG bucket 128 | 18.095 | 20.232 | 1.575 | 9 |
| whole-model pow2 | 1.925 | 4.734 | 1.288 | - |

Bucket 128 makes 1,719 cuBLAS clones and nine cuBLAS captures (three signatures across three
new buckets), a **99.5% capture reduction** versus exact DG. All three DG modes still capture
**288 attention child graphs**, once per site per shape. Parent recaptures are 8 / 8 / 3 for
exact / clone / bucket. The region is still harvested 24 times: this implementation removes
cuBLAS capture on a hit, not the per-shape wrapper walk, cloning, topology queries or layout work.

The bucket's new median is 16% lower than exact DG but **10% higher than DG + clone**. Paired
per-shape median speedups are 1.26x versus exact DG and 0.96x versus DG + clone; it wins 18/24
and 9/24 pairs, respectively. GPU graph time rises from 0.839 to 0.967 ms (**15%**), while replay
rises 2% versus exact DG and 11% versus DG + clone.

The initial 40-shape BERT sweep gave different latency ratios:

| Mode | New median ms | New mean ms | Replay median ms |
|---|---:|---:|---:|
| eager | 4.391 | 5.557 | 4.511 |
| compile | 2.971 | 4.211 | 2.966 |
| trees, exact shapes | 9.464 | 9.511 | 1.844 |
| DG exact | 22.547 | 27.592 | 2.132 |
| DG exact + clone | 19.991 | 46.982 | 1.799 |
| DG bucket 32 | 21.684 | 39.060 | 2.199 |
| DG bucket 64 | 16.892 | 32.574 | 2.285 |
| DG bucket 128 | 15.741 | 30.520 | 2.258 |
| DG bucket 256 | 18.429 | 29.503 | 2.108 |
| whole-model pow2 | 1.966 | 5.996 | 1.690 |

Here bucket 128 improved the median over DG + clone, but made the mean worse than exact DG.
The first replay pass for exact trees includes recording; its five-pass median mostly reflects
warm hits, but its mean is not a steady-state number. The focused repeat removes that ambiguity.
Large new-shape outliers persist: in the repeat a DG + clone call took 834 ms of wall and 835 ms
of process CPU time; an exact-DG call took 144 ms wall and 28 ms CPU. These are not removed from
the means, and their causes are not established. **Do not interpret the median improvement as
an established throughput or tail-latency win.**

### What remains expensive

Unfiltered top-10 cProfile reports are saved beside the raw samples. On three new BERT shapes,
bucket hits still perform 216 `_clone_child` calls, 216 `_call_signature` calls, 252 `_node_sig`
calls, and 36 attention captures. Those operations, plus executing the intercepted wrapper,
remain on the new-shape path. Extra padding/materialization raises GPU time. Reducing captures
alone has not made the path competitive with ordinary compile or whole-model buckets here.
These experiments cover inference and one dynamic axis; they do not establish training, SANA,
unbounded-shape or multidimensional results.

Reproduce with the built fork and matching C++ headers/libraries:

```bash
CUDA_VISIBLE_DEVICES=3 python dynagraph/microbench/cublas_buckets.py \
  --workload mlp --output /tmp/cublas_buckets_mlp.json
CUDA_VISIBLE_DEVICES=3 python dynagraph/microbench/cublas_buckets.py \
  --workload bert --modes eager,compile,trees,dg,dg-clone,bucket32,bucket64,bucket128,bucket256,trees-pow2 \
  --output /tmp/cublas_buckets_bert.json
CUDA_VISIBLE_DEVICES=3 python dynagraph/microbench/cublas_buckets.py \
  --workload bert --shapes 24 --modes compile,dg,dg-clone,bucket128,trees-pow2 \
  --output /tmp/cublas_buckets_bert_repeat.json
```

[Raw samples, logs and profiles](../microbench/results/cublas_buckets_20261015/).
The initial sweeps' generic `captures` counter includes parent recaptures made during harvest;
the focused repeat separates `captures` (child), `cublas_captures` and `parent_captures`.
`bucket_captures` is always cuBLAS-only. The final harness also warms replay twice for every
mode; that clarification was added after the initial sweeps.
