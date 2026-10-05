# SGLang diffusion: BCG measurements (fake weights, H100, 2026-09-28)

Purpose: find a workload first, without presupposing a solution. The questions: in modern diffusion serving, how serious are launch overhead and "new shapes"?
How does SGLang's Breakable CUDA Graph (BCG) handle them? What does it cost?

## Environment

- Container `xinwei_bcg`: image `sglang-ab:base` (torch 2.11 cu130). The code is SGLang main `2a1c477` (2026-09-29),
  placed in `_deps/sglang-main` and used via PYTHONPATH; the diffusion dependencies were installed on top with `--no-deps`.
- Fake weights: `dynagraph/serving/fake_hf_repo.py` downloads only the small files, reads the header of each safetensors file with Range requests,
  and generates random weights (0.02*randn) in `/dev/shm/xinwei_fake`. The compute is the same as with real weights, without downloading 30 GB of weights.
- Load test: `dynagraph/serving/bcg_bench.py`. In a single process, warm up first, then send 8 requests in order:
  - at resolution A, a short and a long prompt, twice each;
  - at resolution B, a short and a long prompt, once each;
  - one resolution that was not warmed up;
  - finally, one more request at resolution A.
- GPU 7, exclusive.

Note: SGLang's per-stage timings are **not synchronized**. With BCG on, the "per-step time" is only the host dispatch time,
and the GPU time gets counted in the decoding stage that follows. So below we only look at end-to-end time, which includes synchronization.

## Z-Image-Turbo (6B DiT + Qwen3-4B text encoder, 9 steps, no CFG)

| Request | eager end-to-end | BCG end-to-end |
|---|---|---|
| 512x512, short prompt | 0.39 s | 0.38 s |
| 512x512, long prompt (224 tokens) | 0.42 s | 0.41 s (signature miss, runs eager) |
| 1024x1024 | 1.11 s | 1.08 s |
| 768x1344 (not warmed up) | 1.12 s | 1.12 s (runs eager) |

- **Not launch-bound**. At 512x512 the host takes about 25 ms per step to dispatch and the GPU about 24 ms per step (one step is about 12.7 TFLOP); the two are roughly even.
- BCG captured 35 segments (all attention is placed outside the graph).
- The 1.23 -> 0.66 s in the SGLang blog post was measured on B200. That GPU is more than twice as fast, so the host becomes the bottleneck.

## SANA 1.5 1.6B (Gemma-2 2B text encoder, 20 steps, CFG 4.5)

DC-AE compresses 32x, so 1024x1024 is only 1024 tokens, and the DiT is small too, so the GPU workload is very light.

| Request | eager | BCG | torch.compile (SGLang default: max-autotune-no-cudagraphs) |
|---|---|---|---|
| 1024x1024, short prompt | 0.89-0.97 s | 0.50-0.59 s | 0.50 s |
| 1024x1024, long prompt | 0.93-0.95 s | 0.52-0.58 s | 0.61 s, second time 1.00 s (recompile) |
| 512x512 | 0.79 s | 0.25-0.30 s | **first time 99 s (recompile plus autotune)**, afterwards 0.39 s |
| 768x1344 (not warmed up) | 1.64 s | 1.65 s (miss, runs eager) | **20.9 s (recompile)** |

- **Severely launch-bound**: under eager, the denoising time at 512x512 and at 1024x1024 is almost the same (0.69 vs 0.72 s); the GPU is waiting on the host the whole time.
- BCG has only 1 segment for SANA: linear attention does not go through the attention classes that are wrapped as break points, so it is one whole graph.
  Each signature takes about 0.2 s to capture, and memory barely grows (peak 10.1 GB vs 10.3 GB for eager; all graphs share one memory pool).
- Signature = resolution x text-length bucket (64/128/256/512/1024). This run warmed up 2 resolutions, 10 graphs in total.
  SANA has dozens of aspect-ratio buckets; warming up all of them means hundreds of graphs.
- **For a signature it has not seen, BCG falls back to eager and is 3x slower**. The VAE decode also spends an extra 0.5-0.7 s the first time it meets a new resolution;
  this is cuDNN convolution selecting algorithms for the new shape.
- On seen shapes torch.compile is about the same as BCG (kernel fusion brings the launch count down),
  but every new shape needs a recompile plus autotune, 20-99 s each time. The log has 30 recompile records.

## How to read this

1. **Small-token-count DiTs like SANA are a real, modern, severely launch-bound workload**:
   eager is 1.7-3x slower than with graphs. Models of 6B and up, like Z-Image, are not on H100; on faster GPUs such as B200 or GB300 they might be.
2. The three existing approaches each miss one piece:
   - eager: slow;
   - BCG: shapes must be declared in advance, and unseen ones fall back to eager; every model needs manual integration (whitelist, padder);
   - compile: a new shape needs a recompile of tens of seconds.
   None of them achieves "no manual work + arbitrary shapes + fast".
3. Shapes change per request, not per step: one request runs 20 steps x 2 (CFG) forwards.
   So "capture once whenever a new signature appears" (about 0.2 s) is already quite cost-effective. That SGLang does not do this is a policy choice; the source does not say why.
   This simple scheme is also a baseline DG must be compared against.
4. For DG, the value is not in steady-state latency but in this: one `compile(dynamic=True)`, and each new shape only pays the cost of patching parameters;
   no recompile, no per-model integration, and no shapes declared in advance.
   To verify this, we need to take SANA's transformer out on its own, put it into our harness,
   and compare it side by side with eager, compile(dynamic), one graph captured per shape, DG, and the oracle.

## Manual effort (SGLang, as of 2026-09-28)

- There are 110 PRs with breakable or BCG in the title (since 2026-05): 62 merged, 33 still open, of which about 31 are for diffusion.
- Typical PRs:
  - enable BCG for a model (LTX-2, SANA, SANA-Video, LongCat, JoyEcho, FLUX.1, FLUX.2 Klein, Cosmos3, ...);
  - fix crashes during replay, stale inputs, out-of-bounds accesses, padding positions;
  - make results bitwise identical to eager.
- The BCG core code is about 2400 lines, plus 7 model-specific padders and a whitelist of 44 models;
  it is mentioned in 639 places across 131 files.

## Next step

First use diffusers' `SanaTransformer2DModel` (random weights) to run SANA's DiT on its own, following SANA's real aspect-ratio buckets
and text-length distribution, and compare: eager, compile(dynamic), trees, bucketing, one graph per shape (oracle), DG.

## SANA DiT on its own in our harness (2026-09-28, `e2e/sana.py`, GPU 6/7 exclusive)

Using diffusers' `SanaTransformer2DModel` (SANA 1.5 1.6B config, random weights, bf16) to simulate how SGLang serves:
each request randomly picks one of 33 aspect-ratio buckets, with CFG on (batch 2), and runs 20 steps on that same shape; the text is fixed at 300 tokens (same as the diffusers pipeline).
24 requests in total, 16 shapes, 13 of which appear for the first time inside the timed section. The table below is the sum of the 20 steps of each request, median, in ms:

| Mode | 512 bucket: new shapes | 512: seen shapes | 1024 bucket: new shapes | 1024: seen shapes |
|---|---|---|---|---|
| eager | 648 | 540 | 709 | 608 |
| compile(dynamic=True), no graphs | **197** | 197 | **363** | 365 |
| trees (record one graph per shape, i.e. "capture whenever a new shape appears") | 205 | 184 | 370 | 357 |
| DG (cuBLAS and cuDNN through tier 2) | 232 | 185 | 427 | 362 |
| pad to the largest shape | 295 | 296 | 1053 | 1064 |
| oracle (one graph per shape, replay-only) | 168 | - | 338 | - |

- eager really is severely launch-bound, 2 to 3.9x slower than the oracle.
  **But `compile(dynamic=True)` recovers most of that in one step**: it is only 1.17x (512) and 1.07x (1024) off the oracle,
  and it does not recompile on new shapes.
- When SGLang uses compile, every new shape stalls for 20 to 99 seconds. The cause is that it uses max-autotune and specializes per shape; it is not a problem with compile itself.
  In other words, the "eager plus BCG" route SGLang chose is itself an engineering choice that could be replaced.
- On top of compile, CUDA Graph saves at most another 7% to 15%. trees (capture whenever a new shape appears) already gets most of that: 184 vs 168 on seen shapes.
- DG is the slowest on new shapes. Every new shape requires re-harvesting the cuBLAS and cuDNN (depthwise convolution) child graphs; the log recorded 449 and 470 new topologies.
  After pinning cuBLAS to a few variants by M bucket, this part of the new-shape cost can be removed; the measured headroom over compile in this setup is the 7% to 15% above.

Summary: SANA is a modern workload that is severely launch-bound under eager; on this implementation the slowness comes from the "eager plus hand-made graphs" route.
A correctly configured `torch.compile(dynamic=True)` handles it with no manual work, and the remaining gap for "one graph covering all shapes" measured here is 7% to 15%. Whether that gap, or the new-shape cost, matters on some workload is an open question.

## Direct comparison inside SGLang: BCG and several compile configurations (2026-09-28, GPU 6/7, `bcg/sgl_*.log`)

**This section supersedes the reading of the previous section**: the previous section used diffusers' clean implementation and found that "compile(dynamic) is enough";
but SGLang's own model code behaves differently, and this section takes precedence.

SANA 1.5 1.6B, 20 steps, CFG 4.5; warm up 1024x1024 and 512x512; two switches were added locally for the experiment:
`SGLANG_TORCH_COMPILE_DYNAMIC` and `SGLANG_MARK_STEP`. The table below is end-to-end time (in seconds):

| Config | 1024x1024 (seen) | 512x512 (seen) | 512x512 first time | 768x1344 (unseen) | graph breaks / recompiles |
|---|---|---|---|---|---|
| BCG | 0.51-0.53 | **0.26** | 0.26 (warmed up) | 1.68 (falls back to eager) | - |
| compile, SGLang default (max-autotune-no-cudagraphs, dynamic=None) | 0.50-0.53 | 0.41 | 14.9 | 6.0 | 9 / 6 |
| compile, dynamic=True, default mode | 0.48-0.52 | 0.45 | 22.5 | 19.3 | 8 / 4 |
| compile, dynamic=True, max-autotune | 0.47-0.50 | 0.40 | 6.5 | 5.6 | 8 / 12 |
| compile, reduce-overhead (dynamic or static, with or without mark_step) | **crash** | | | | 7-8 / 1-2 |

- At 1024x1024 all configurations are even, at about 0.5 s; it is already GPU-bound. **At 512x512 BCG is 1.5-1.7x faster than every compile configuration** (0.26 vs 0.40-0.45 s).
  compile does kernel fusion, but SGLang's own JIT kernels (tvm_ffi calls, `timestep_embedding_jit` wrapped in `torch.compiler.disable`, etc.)
  cause 7-9 graph breaks; add the absence of CUDA Graph, and small resolutions are still launch-bound.
- The first time on a new shape stalls for 6-22 s, **mostly because of the VAE**: SGLang compiles the VAE as well, and compiles it statically per shape (`torch_compile.py:158`,
  reporting `size mismatch ... expected 32, actual 16`); a single decode stage takes 18-21 s. With dynamic=True, the DiT only adds about 1 s.
- **PT2 automatic graphs (reduce-overhead) crash outright on SGLang's SANA**: `accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`,
  at `sana.py:652` (the continuation function after the `time_embed` graph break). Adding `cudagraph_mark_step_begin()` before each DiT call does not help either:
  one forward is split into several segments, and an earlier segment's output is overwritten by a later segment's replay. To make it work, the JIT kernels have to be registered as custom ops to remove the breaks, or the outputs cloned by hand. All of this is manual work.

Takeaways:
- In real serving code, neither of the two "no manual work" routes works: compile is 1.5-1.7x slower at small resolutions and stalls for seconds to tens of seconds on new shapes; PT2 automatic graphs crash outright.
  BCG pays with per-model manual integration to get the 1.7x at 512x512.
- This is where DG fits: on real code with custom kernels and graph breaks, achieve BCG's effect automatically and cover all shapes.
  The prerequisite is that DG can handle the eager segments between graph breaks (which is exactly what BCG does), not only the segment Inductor generates.

## Getting PT2 automatic graphs to work in SGLang (2026-09-28, GPU 1/2, `bcg/sgl_te_*`, `bcg/sgl_auto_min*`)

Several experimental switches were added in the local clone to remove the problems on the PT2 automatic path one by one:

1. **The only graph break**: the JIT kernel of `timestep_embedding` is called through tvm_ffi and is also wrapped in `torch.compiler.disable`.
   Its input t is (B,) and its output is (B, dim), independent of resolution. Wrapping it with `torch.library.custom_op` plus a fake implementation (about 20 lines, `SGLANG_TE_CUSTOM_OP`)
   brings graph breaks down from 7-9 to 0. But **without graphs, 512x512 is still 0.46 s**, which shows that 512x512 is slow because it is launch-bound, not because of this break.
2. **The reduce-overhead crash**: CFG calls the DiT twice in a row (conditional first, then unconditional); both calls replay the same graph, and the second overwrites the output of the first.
   The fix is to add `cudagraph_mark_step_begin()` before each call (`SGLANG_MARK_STEP`) and clone the output once after the call (`SGLANG_CLONE_OUT`).
   BCG also copies its outputs out; DG avoids the same problem by rotating through several sets of output buffers.
3. **dynamic=True still recompiles**: the cause is duck sizing. At 1024x1024 the latent is (2,32,32,32); the channel count, H and W are all 32,
   so they share one symbol; the first convolution's input channel count is fixed at 32, so the compiler adds an `Eq(s, 32)` guard and specializes H and W to 32 along with it.
   Turning off `torch.fx.experimental._config.use_duck_shape` (`SGLANG_NO_DUCK_SHAPE`) fixes it.
4. **The VAE is compiled statically per shape**, 18-23 s for every new shape. Here the VAE is simply not compiled (`SGLANG_NO_VAE_COMPILE`);
   it is not the bottleneck here, about 0.02 s per decode.

Results (end-to-end, in s; "first time" means this shape was not warmed up):

| Request | BCG | automatic graphs (the 5 changes above + reduce-overhead + dynamic) | same changes, no graphs |
|---|---|---|---|
| 1024x1024 (seen) | 0.51-0.53 | 0.50-0.56 | 0.52-0.55 |
| 512x512 first time | 0.26 (warmed up) | 0.91 | 1.04 |
| 512x512 (seen) | **0.26** | **0.28** | 0.42 |
| 768x1344 first time | 1.68 (falls back to eager) | **1.31** | 1.32 |

Both groups compiled only once, on the first call after startup (about 35 s), and then had 0 recompiles and 0 graph breaks.

- In steady state, automatic graphs essentially match BCG (512x512: 0.28 vs 0.26), and on unseen shapes they beat BCG (they record a graph on the spot, while BCG falls back to eager).
- The extra 0.6-0.8 s the first time on a new shape is mostly cuDNN selecting algorithms the first time the VAE meets that shape (about 0.5-0.7 s), which BCG and eager pay as well.
  The recording itself costs only 100-200 ms: the first 512x512 request is 0.91 s with graphs and 1.04 s without, so with graphs it is actually faster.
- So on SANA, **the 1.7x BCG gets at 512x512 can be obtained with PT2's automatic path plus 5 small changes (about 25 lines)**. Compare BCG's investment: 110 PRs, about 2400 lines of core code, a whitelist of 44 models.
- What DG could still remove here, as measured: the 100-200 ms of recording per new shape, and about 0.02 s in steady state. Whether a workload exists where this matters is an open question.

Observations for the project: the gaps found on this path are a few pitfalls on the PT2 automatic-graph path that nobody had worked through inside a serving framework:
- custom JIT kernels break compilation, and a way to wrap them as custom ops automatically is needed;
- with CFG, where the same graph is called back to back and both outputs must be kept, trees' output-lifetime rules go wrong;
- duck sizing wrongly specializes the resolution to a constant;
- the VAE is compiled statically per shape.
Each of these could be turned into a concrete improvement to PT2 or SGLang; how that line of work relates to DG is left open.

## Every request has a different prompt length (2026-09-29, GPU 1/2/3, `bcg/sgl_vp_*`, `bcg/sgl_mem_*`)

SGLang's text-encoding setting for SANA is `padding=True` (pad only to the longest prompt in the same batch),
so the text dimension is the real prompt length, and it differs for every request. The earlier tests used only two fixed prompts and did not cover this.
This load test: 16 requests (12 at 512x512, 4 at 1024x1024), with prompt lengths all different, between 4 and 160 words; plus a separate run of 40 512x512 requests to sample memory.

| End-to-end | BCG (text length padded to 5 buckets) | PT2 automatic graphs (5 changes) | compile, no graphs |
|---|---|---|---|
| 512x512, every request | 0.25-0.27 s | 0.28-0.31 s | 0.41-0.45 s |
| 1024x1024 (excluding the first request at that resolution) | 0.51 s | 0.53 s | 0.52 s |
| 40 requests: median / max | 0.26 / 0.31 s | 0.28 / 0.37 s | - |
| memory growth over the 40 requests | +8 MiB | +58 MiB (about 1.5 MB per graph) | - |

- BCG never fell back to eager in the 16 requests: once the text length is padded to a bucket, every request hit a graph captured during warmup.
- PT2's trees records a new graph for every new text length (it also printed a warning about "seeing many distinct sizes").
  But one request has 40 forwards: on a new length, the first two are used for warmup and recording, and the rest replay directly.
  So each request only costs 20-50 ms more, about 8-15%; each graph takes about 1.5 MB of memory, which is negligible.
- The first 1024x1024 request takes about 1.04-1.09 s; it is the first time this resolution appears and cuDNN in the VAE is selecting algorithms, unrelated to prompt length.

Summary: even when every request has a different shape, diffusion runs "one shape 40 times in a row", which spreads the per-new-shape recording cost thin.
Measured: the most DG could save here is those 20-50 ms per request (about 10% at 512x512, nearly 0 at 1024x1024).

## The full BCG usage surface in SGLang (source `2a1c477`, 2026-09-29)

**On the LLM side (`srt/`): BCG is the default prefill backend on CUDA** (`cuda_graph_config.default_prefill_backend`; HIP, NPU and other platforms still use torch.compile's piecewise).
decode can also choose BCG, but the default is the full graph (FULL).
prefill captures graphs bucketed by total token count; at replay it pads up to the nearest bucket, and each segment of each bucket is captured once.
The places broken out with `@eager_on_graph` in between are all operations that "depend on each batch's real metadata, or need to synchronize with the host, or need cross-rank communication":

| Break point | Location | Reason |
|---|---|---|
| Attention (general) | `layers/radix_attention.py` (unified attention, including the variant with lse and the variant with extra parameters) | per-request lengths and KV cache indices differ every batch (varlen, paged KV) |
| Linear attention | `layers/radix_linear_attention.py` | same: per-sequence state and metadata |
| MLA | `forward_mla.py` (bmm plus unified attention) | attention of the DeepSeek family |
| DSA sparse-attention indexer | `dsa/dsa_prefill_cuda_graph.py`, `dsa/kpool_prefill_cuda_graph.py` | choosing the top-k positions depends on real lengths |
| DeepSeek V4 | `models/deepseek_v4.py` (attention, the low-ratio compressor and indexer, engram hash ids), `deepseek_v4_backend.py` | source comment: "the compressor and the prefill indexer need to sync with the host, same as attention" |
| fp8 rope plus KV write | `hpc_ops_backend.py` | the KV cache write positions differ every batch |
| MoE EP all-to-all | `moe/ep_moe/layer.py` (DeepEP, flashinfer's a2a) | cross-rank communication; capture only records the buffer addresses, replay actually runs it |
| Mamba2 | `models/nemotron_h.py` | per-sequence state of the state-space model |
| Inkling short convolution | `models/inkling.py` | source comment: the short-convolution metadata (cu_seqlens, seq_idx) gets frozen to bs=1 values at capture time, which breaks multi-sequence prefill, so the whole attention plus short-convolution group is put outside the graph |

In addition, a set of models check `is_in_breakable_cuda_graph()` in their code to take a different branch: deepseek_v2, kimi_k3, qwen2_moe (the side stream must be joined before DeepEP), gemma3, muse_glimmer (puts the embedding inside the graph).
Speculative decoding is adapted too: eagle_worker_v2 switches to a different path when prefill goes through BCG; a comment in dspark says "BCG replay skips the Python logic in the model, so the same check has to be recomputed".

**On the diffusion side (`multimodal_gen/`)**:
- the four DiT attention classes (Ulysses, Ulysses_VSA, Local, USP) are uniformly wrapped as break points: sequence-parallel communication, varlen packing, and sparse or dynamic attention all live inside them;
- MiniMax-H3's attention core and hybrid attention get their own separate break points;
- plus a whitelist of 44 models, 7 model-specific padders (prompts padded to buckets), and warmup per resolution.

**How to read this**: in SGLang, BCG is essentially a **general "prefill/DiT piecewise capture" framework** with a fixed pattern:
within a batch, the only thing that really varies per request is attention (and the KV writes, indexing and communication tied to it); the rest (GEMM, MLP, norm, MoE expert compute)
depends only on the total token count and is static once bucketed by token count. So it breaks at attention, runs attention eagerly with each batch's real metadata, and records the rest as graphs per token bucket.
This is the same idea as vLLM's piecewise (torch.compile splitting at attention), only without needing a compiler.

Two implications for DG:
1. The "dynamism" BCG handles is mainly attention metadata (varlen, paged KV, sparse indices), not GEMM shapes. GEMM shapes are handled by padding to token buckets,
   and as measured earlier, padding to a multiple of 128 is nearly free on the GPU.
2. If DG were to replace BCG, the key is to keep attention inside the graph: patch the attention kernel's parameters per batch (grid, the cu_seqlens pointer or length),
   like the describe we wrote for FA3 in tier 3. Then one forward is one full graph, with no break at every layer.
   The cost of one break per layer is one cudaGraphLaunch plus one Python call, which is small relative to the GPU work in a GPU-busy setting like prefill.
   So the benefit of this replacement would only show up in small-batch, launch-heavy settings like decode, where the default is already a full graph. Whether some workload makes this replacement pay off is an open question.
