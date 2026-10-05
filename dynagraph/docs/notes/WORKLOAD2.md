# Second-round workload survey (2026-09-24, researched by a background agent, not verified locally)

Question (Xinwei): is there a real workload that is severely launch-bound, with large shape variation, where padding/bucketing/re-capture is expensive? Such a workload is where DG's per-call overhead could be offset.
Measured so far: SAGE and SchNet are launch-bound, but pad-to-max is cheap and beats DG; ESM-2 35M is expensive to pad but GPU-bound;
on ESM, compile with the default cuBLAS is 17.2 ms vs 14.6 ms of GPU time, headroom <=1.2x.

## Ranking (by strength of the evidence the agent gave)

1. **LLM prefill/extend and mixed prefill+decode batches**: the token count changes every step. vLLM defaults to FULL_AND_PIECEWISE:
   uniform decode runs as a full graph, everything else piecewise (attention runs eager between the pieces); capture sizes [1,2,4]+range(8,256,8)+range(256,max,16), capped at 512.
   SGLang: prefill token buckets 4..4096+; on GLM, 42 shapes take 2.4 GB of graph memory, and building graphs for a 235B MoE takes 90 s; "token padding has a real compute cost".
   gpt-oss-120b TP4 prefill: full graph 1.93x vs eager, breakable graph 1.70x, torch.compile piecewise 1.45x (GB300; H100 not measured).
   -> If DG could bring attention into the graph and use one graph patched for each token count, it could close the piecewise gap and save the memory of multiple buckets. Needs measurement on H100.
   Sources: https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph/ , https://docs.vllm.ai/en/latest/design/cuda_graphs/
2. **Speculative decoding: verify length differs per request** (DSpark ragged verify, D-cut): chosen each step by confidence. SGLang rounds the total up to a captured bucket;
   D-cut allows only 4 budget ratios "to avoid the memory blow-up of one graph per K", going 1.26x -> 1.65x (vs DFlash-16) at high concurrency on H20.
   dflash is available locally. https://www.lmsys.org/blog/2026-07-06-dspark-sglang/ , https://arxiv.org/pdf/2607.14647
3. **MLIP inference (MACE/UMA/SevenNet)**: in MD the edge count changes every step; batched screening has different molecules. On H100 the minimum latency for a 2-atom graph is 18-25 ms (MACE-MP),
   source unknown (launch? Python? autograd for the forces? neighbor list?) -- cProfile first. TorchMD-Net pads neighbors to the maximum pair count.
   https://arxiv.org/abs/2606.02455 , https://arxiv.org/pdf/2402.17660
4. EAGLE/MTP draft chain: 1.4 ms/forward (concurrency 1, about 10x the weight-read lower bound), 15-18% of decode; vLLM supports only piecewise for the draft. Shapes are essentially fixed; it is a "full graph" problem, not a changing-shape problem.
5. Multi-LoRA: vLLM captures one graph per number of active LoRAs: 94 s/1.21 GiB (before the fix) -> 29 s/1.06 GiB. Device-side metadata may be a better fit than DG.
6. Elastic EP: on scale-up/down vLLM re-captures, about 38 s of downtime; padding to the max EP cuts that to 1 s, but some MoE kernels lose 13-77% of their throughput.
7. RL rollout (verl): the batch shrinks as requests finish; graphs cut generation from 85 s to 62 s (Qwen2-7B).
8-10. DeepEP normal-mode prefill, streaming ASR (NeMo has largely solved it with conditional nodes), point clouds/recommendation: weak evidence.

Not candidates: grammar-constrained decoding (fixed-shape bitmask), beam search (vLLM V1 splits it into independent requests in the front end).

## Tree verification (each request has a different tree shape)

SGLang EAGLE-2/3, TRT-LLM `use_dynamic_tree`: tree shapes differ but the **token budget is fixed** (N per request), so the shape is always batch x N;
the tree structure lives in the data (mask/positions/retrieve), and **graphs work as usual**. Shapes change only when the per-request **size** changes (DSpark ragged, D-cut, adaptive step count).
vLLM supports only static trees (the issue for dynamic pruning, #41823, is "not planned"). Nobody has reported how much acceptance length is lost by using a uniform tree shape.
The draft is launch-bound; verification at scale is not (42-95% of step time; the EAGLE speedup drops from 1.73x to 1.21x going from bs1 to 128).

## Elasticity: look at the frequency

| Mechanism | How often it changes | How graphs are handled | Cost |
|---|---|---|---|
| vLLM elastic EP | scale-up/down events | re-capture / pad to max EP | ~38 s vs 13-77% throughput |
| SGLang/Mooncake EP | on failure | fixed graph + device-side peer table | <10-21 s recovery |
| EPLB rebalancing | every 3000 / 1000 steps | shape unchanged | none |
| LoongServe elastic SP | every iteration | paper does not mention graphs | unknown |
| GF-DiT | every denoising step | does not mention graphs | group creation 778 ms -> 60 us |
| Dynamo planner etc. | 180 s / 1-2 min | re-capture at startup | does not matter |

In production the width changes rarely and re-capture is enough; only research systems change width every step, and none of them says what they do about graphs.

## LLM pure decode

Padding adds at most 7 rows (bs<256) / 15 rows; decode is bound by weight reads, so padding is nearly free (estimated <2%, not measured). All graphs are captured once at startup.
5 ms of host patching overlaps with the previous step -> steady-state per-token time changes by about 0-2%. The real gains are in startup time (10-65+ s) and graph memory (hundreds of MB to several GB).
A stronger LLM argument: for mixed batches and variable-length verification, use one full graph patched per token count, replacing piecewise capture and the per-bucket memory.

## Measured additions (2026-09-24/25)

- MACE MD22 (GPU 4, clean): compile 4.46 ms, DG 4.41 / 4.35 ms, pad to the max edge count 3.94 ms, GPU time 3.39 ms. Padding is the fastest of these.
- vLLM mixed batch (MBT 2048, 256 ShareGPT requests, two rounds):
  - 0.6B: NONE 6.95/6.56 s, PIECEWISE 4.92/4.37, FULL_AND_PIECEWISE (default) 2.48/2.10, FULL 1.89/1.88. The full graph is about 1.2x faster than the default, but vLLM's own FULL already achieves this (by padding to the capture sizes).
  - 8B: FULL_AND_PIECEWISE 5.74/5.84, FULL 5.77/5.75, a tie; FULL graph memory 4.44 GiB, default 1.30 GiB.

## Elastic parallelism source-code survey (2026-09-25)

Sources cloned locally for reading (not included): LoongServe f6e8fc15, ArcticInference aca5d9a8. NanoCP has no public code; only the paper was read.

| System | Uses graphs? | What changes per step per GPU | How graphs cope | DG opportunity |
|---|---|---|---|---|
| LoongServe (SOSP'24) | Not at all. prefill calls `torch.cuda.synchronize()` in every layer, there is a `.item()` before decode, and one Ray RPC per rank per step; the baseline vLLM is also enforce_eager | role (master/peer), slice size, peer set, number of send/recv | n/a | the graph structure changes with the NCCL peers; the host syncs must be removed first, which is a large change |
| NanoCP (arXiv 2605.21100) | 2-D bucketed graph table (M local queries, N remote results), routing table on device | number of queries, number of results, routing | 48 graphs, 5.32 GiB/GPU (vLLM 32DP: 1.02 GiB) | only graph memory is left, and there is no code |
| Shift Parallelism (ArcticInference) | two families of FULL graphs (TP / SP), switching at a 512-token threshold | mode chosen by token count | captured ahead of time, no re-capture, fixed communicator | small (the second copy of the weights costs far more than the graphs) |

- Nobody has measured how often LoongServe switches. Under ShareGPT it probably stays on a single instance the whole time; under LEval the master split varies with the batch, and scale-up happens only when the KV cache is full.
- A scheduler-only simulation is feasible (stub out `model_rpcs`, write our own A/B/C cost CSV), about one day of work. But since LoongServe itself does not use graphs, the simulation's results cannot answer the DG question.
- Observed: the systems that use graphs rely on device-side routing plus bucketing, or on one family of graphs per mode; none of them leaves a cost that is "paid every step and cannot be removed by bucketing". Whether elastic parallelism offers a case for DG is an open question.

## GraCE-style HF models, shape changes every step (2026-09-25, e2e/zoo.py)

GraCE (OSDI'26, arXiv name PyGraph) uses static-shape models from TorchBench/HF/TIMM and compares PT2 with and without CUDA Graphs.
Here the same set of HF models is used (random init, bf16 inference), with a random (B, L) every step, L log-uniform over 16..512.
cuDNN SDPA is disabled: it builds a plan each time it meets a new shape, about 5 ms per layer, which would swamp everything else on new shapes.
GPUs 5/4/3; the first part was clean; the last part was aborted when someone else's vLLM saturated all 8 GPUs, and that part's data is unusable.

bs=1, per-step median ms (cuBLAS GEMM + SDPA):

| Model | compile | trees | pad->490 | bucket (powers of 2, 6 buckets) | oracle |
|---|---|---|---|---|---|
| bert | 4.32 | 7.17 | 1.96 | 1.49 | 1.05 |
| distilgpt2 | 1.53 | 4.04 | 1.41 | 0.92 | 0.63 |
| xlnet | 2.88 | 6.74 | 1.86 | 1.36 | 1.15 |
| t5 | 3.60 | 7.61 | 1.99 | 1.49 | 1.05 |
| mobilebert | 11.27 | 29.7 | 4.80 | 5.55* | 3.08 |
| debertav2 | 4.98 | 10.2 | 2.68 | - | 2.33 |
| blenderbot | 5.33 | 10.8 | 2.97 | 2.32 | 1.62 |
| albert | 2.94 | 6.41 | 1.45 | 1.04 | 0.82 |
| electra | 2.67 | 7.21 | 1.42 | 1.19 | 0.77 |

*Ran concurrently with other jobs; suspect. oracle = the same compile(dynamic) code with one graph captured per shape, replay-only, excluding the dynamo/AOT front end; no torch.compile approach can reach it.

B in {1..32}, per-step mean ms on new shapes (padding to (32,511) is 10.4x the average token count):

| Model | compile | trees | pad | oracle mean/median | bucket steady-state median (29 buckets, 30 compilations) |
|---|---|---|---|---|---|
| bert | 4.60 | 11.41 | 14.87 | 2.17 / 1.22 | 2.00 (warm 97 s) |
| distilgpt2 | 3.05 | 4.79 | 10.38 | 1.51 / 0.91 | 1.46 (warm 54 s) |
| xlnet | 4.73 | 8.71 | 27.58 | 3.06 / 1.40 | 1.87 (warm 170 s) |
| t5 | 4.42 | 8.92 | 19.42 | 2.34 / 1.21 | 1.98 (warm 157 s) |
| mobilebert | 14.87 | 36.84 | 11.86 | 3.65 / 3.21 | - |
| blenderbot | 5.66 | 10.45 | 19.98 | 3.04 / 1.86 | - |
| albert | 2.75 | 5.42 | 11.67 | 1.61 / 0.93 | - |
| electra | 3.61 | 7.14 | 5.44 | 1.14 / 0.84 | - |

In this set, debertav2's compile warm-up took 813 s (anomalous); deberta overflows in bf16 and was skipped.

Findings:
- PT2's built-in trees is slower than no graphs for every model under dynamic shapes (it records one graph per shape, and is slower even on seen shapes).
- Padding to the global max is not viable; the real competitor is power-of-2 bucketing. At bs=1, bucketing is 1.2-1.5x slower than the oracle, with 6 buckets.
  With 2-D BxL it needs 29 buckets and 30 compilations (warm-up 1-3 minutes, and new buckets keep being compiled during the timed section); at steady state it is still about 0.6-0.8 ms slower than the oracle median.
- DG measured (BERT bs=1):
  - cuBLAS: new shapes 14.7 ms (the cuBLAS child graph is re-harvested for every shape), seen shapes 1.78 ms.
  - Triton GEMM + eager attention: 1.50 ms on both new and seen shapes, no harvest; compile 1.58, bucket 1.39.
  cProfile: GPU 1.04 ms (same as the oracle), host path 0.52 ms, of which DG's own `__call__` is about 0.15 ms; the rest is the dynamo/AOT front end shared by all compiled modes.
  At bs=1 bucketing costs almost nothing on the GPU, so DG's 0.15 ms has nothing to offset it in this setting; with 2-D BxL bucketing costs about 0.8 ms more, so DG could come out ahead there and would also save the 30 compilations. This comparison was not finished here (the GPUs were fully occupied); it was run on 2026-09-25, see the BERT 2-D table in `docs/MEASUREMENTS.md`.
- Fixed a DG bug (not committed): after LRU evicted the main graph for some topology combination, a previously harvested hkey arriving again raised a KeyError on `self.execs[combo]`; now a missed lookup triggers a re-capture (`_drop_lru_exec` was factored out into a method).
