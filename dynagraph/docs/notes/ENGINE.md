# The real engine: handing vLLM's decode to DynaGraph (2026-09-21, late night)

The approach Xinwei set: **do not write a decode loop ourselves** -- "Personally I feel that writing our own decode that doesn't line up with the engine is pointless."
Install vLLM directly in `xinwei_autocudagraph`, use our own torch, turn compile on, and see what happens.

Probe: `dynagraph/serving/probe_vllm_decode.py`. Model `Qwen/Qwen3-0.6B` (already in the cache), single GPU,
`max_model_len=1024`, greedy, `max_tokens=16`, batch stream described below (the first version only swept `1,2,3,2,1`, which was not wide enough; see "The first version's conclusion was wrong").
For how the environment was set up, see `SETUP.md` pitfall 11 (three traps: not sourcing use_main.sh, `cmake_prefix_path` being broken,
and the flash_attn 2.7.4 in the container having been built for torch 2.11).

## Four configurations compared

The number of times `torch.cuda.CUDAGraph.__init__` is called is the only ruler that can measure both paths at once.

| Config | Compile | Who captures | Captured at startup | Total captured | inductor recordings | extern sites | Regions served by one graph | Fallback tag |
|---|---|---|---|---|---|---|---|---|
| `vllm` | vLLM piecewise | vLLM, per its ladder | **178** | 178 | 0 | - | - | - |
| `stock` | bare torch.compile | inductor | 1 | 6 | 5 | 0 | 0 | - |
| `dynagraph` | bare torch.compile | inductor + DynaGraph | 171 | **1522** | 1 | **168** | **2** | - |
| `dynagraph` + `EXTERN_CHILD=0` | same as above | same as above | 1 | 6 | 5 | 0 | 0 | **`extern-launch`** |
| `piecewise` | vLLM piecewise | inductor + DynaGraph | 63 | 445 | 167 | 4 | - | **`runtime-mismatch`** |

All configurations produce the same output text (`" 0, 1, 2, 3, 4,"`); there are no numerical problems.

## The first version's conclusion was wrong; setting that straight first

In the first version I saw "1522 graphs captured in total, still growing during serving" and drew a conclusion from it. Xinwei knocked it down with one sentence:
**"Is this decode? Doesn't normal decode just capture one graph per batch size?"**

Two mistakes:

1. **What was measured was not decode.** In the log, `s72/s80` take the values 7, 14, 21 -- those are **prefill** token counts
   (prompt ~7 tokens x batch); for decode, num_tokens equals bs. I counted the two phases together.
2. **"Still growing" was an illusion.** In that round I swept `1,2,3,2,1`, and every new bs brought in new prefill and
   decode shapes. Fixing the batch at `2,2,2` and looking again:

   ```
   bs=2  new captures this round 340   <- cold start
   bs=2  new captures this round 0
   bs=2  new captures this round 0
   harvest 3 times, 2 distinct shapes:
     {'s72': 14}  prefill  1 time   168 child graphs
     {'s72':  2}  decode   2 times  338 child graphs
   ```

   **Steady state is 0.** What grows is cold start, not serving.

## The real comparison: sweeping bs wide

`BS=1..12`, both configurations measured with the same ruler:

| | Total captures | Composition |
|---|---|---|
| vLLM | **354** | all at startup, 0 during serving |
| DynaGraph | **4048** | 22 distinct shapes x ~168 child graphs |

22 shapes = 12 decode (bs 1..12) + ~10 prefill (7, 14, 21, 28, 35, 42, ...).

**Measured: on a real transformer, DG currently captures 11x as many graphs as vLLM's bucketed capture (4048 vs 354).**

The claim "one graph serves all shapes" holds only for the **main graph** -- the two log lines
`DynaGraph served: one graph now covers all shapes for this region` are genuine,
and the main graph was re-captured only 3 times, because the arena had to grow. But **the extern child graphs still grow linearly with the number of shapes,
and each growth step is 168 graphs**.

## Why 168

Following the stack down, there is only one source:

```
dynagraph.py:5546 in on_extern | g = torch.cuda.CUDAGraph(keep_graph=True)
  <- partition_0 | torch.ops.vllm.unified_attention(...)
  <- partition_0 | extern_kernels.mm(buf375, ...)
```

Qwen3-0.6B has 28 layers; in every layer, each linear-layer `extern_kernels.mm` and the `torch.ops.vllm.unified_attention`
counts as one site, 168 in total.

**In a transformer, extern calls are almost all of the operators.** The ones DG can patch
(pointwise, reduction and similar Triton kernels) are actually the minority. DG's earlier probes were all small regions with a few sites,
so this ratio had never been exposed.

## Turning the child route off: outright refusal

With `EXTERN_CHILD=0`, the region is tagged `extern-launch` and **refused as a whole**, falling back to plain cudagraph_trees
(6 graphs, exactly the same as `stock`).

So today there are only two options, and neither works as is:

- **child on**: one main graph, but 168 child graphs per shape -- an order of magnitude more than bucketing;
- **child off**: the whole region is not served.

## So the next step comes down to one thing

**Extern calls have to stay in the main graph and be patched by parameters**, neither captured as child graphs nor given up on.
If that works, 4048 becomes 1, and only then is there any basis for claiming it beats 354; whether it can be done, and what it is worth to an inference engine, is an open question.

The two kinds are handled separately:

- **`extern_kernels.mm`**: cuBLAS picks the kernel by shape (dense sweep over M=1..4096: 94 kernels,
  **3 cluster shapes**, the choice changes 357 times between adjacent M). On an already-instantiated exec:
  `cuGraphExecKernelNodeSetParams` **can only swap func**, not the cluster (that is a node attribute,
  not part of the kernel node params) -- crossing clusters always fails, measured CUDA 912 / 715
  (`dynagraph/_wf_swap_bf16.log`, `_wf_swap2/3.log`, 2026-09-19 06:19).
  What can swap the cluster along with everything else is **full-graph `cudaGraphExecUpdate`** (verified bit-exact the same day, but not timed).
  So the right decomposition for mm is: **one SWITCH body per cluster shape (3 of them), with the func inside each body swapped via setparams**.
- **`torch.ops.vllm.unified_attention`**: a vLLM custom op; it has to go through the declaration route.
  `_capture_deps` / `_capture_tma` already have this shape -- let the operator declare "what my grid / my
  descriptor is computed from", instead of having DG guess.

## Latency: how long does patching nodes take per forward (2026-09-22)

Xinwei asked the question that really decides this line: **"The key is how much will latency increase? I mean when nodes are replaced during a single forward"**,
and pointed out that 0.6B is too small -- **"otherwise, with the CPU, our patching every round would absolutely not work"**.

Probe `dynagraph/serving/probe_vllm_latency.py`, driven by `_latency_sweep.sh`.
Method: run the same batch with two `max_tokens` values (T1=8, T2=72) and take the slope

    per-step latency = (t(T2) - t(T1)) / (T2 - T1)

Overheads that depend only on the number of requests -- prefill, scheduling, detokenize -- all cancel out. The three configurations are **run in separate processes, interleaved**
(host load is the operating condition, not noise), and within each process both the host side (`perf_counter`) and the device side (CUDA event) are measured.

### Not measurable on 0.6B

Qwen3-0.6B / bs=8, median over four interleaved rounds: vllm 1640, stock 2550, dynagraph 2377 us.
**stock->dynagraph is negative (-174 us)**, while stock alone can jump from 2149 to 3608 between single rounds --
the effect is completely buried in host-load drift. At 0.6B a step takes only 2.4 ms; it is host-bound, and a device-side difference cannot be measured.
Only the in-process self-reported numbers are stable: `_host_step` 98 us, `runner.__call__` 254 us, plain replay 173 us.

### On 14B it is clean

Qwen3-14B / bs=8, 11-13 ms per step, GPU-bound, very low variance (min/max differ by < 0.5% across three rounds).

| | Wall clock per step | Region GPU time (CUDA event) | Host side |
|---|---|---|---|
| `vllm` | 11201 us | - | - |
| `stock` | 12079 us | **11348 us** | cudagraph replay 224 us |
| `dynagraph` | 13317 us | **12332 us** | `_host_step` 173 us, `__call__` 468 us |
| **dynagraph - stock** | **+1238 us (+10.3%)** | **+984 us (+8.7%)** | **+244 us (+2.0%)** |

The GPU times of two independent runs were 11346/11350 and 12332/12332 us respectively, matching to the microsecond.

### Conclusion: patching nodes is cheap, child graphs are expensive

- **The host overhead of patching nodes is ~173 us, the net increase ~244 us, which is 2% of a step on 14B.**
  The worry that "patching every round on the CPU absolutely won't work" does not hold at this size -- but it does hold on 0.6B
  (254 us is 10% of 2.4 ms), so **the lower bound on where this line applies is model size, not the algorithm**.
- **79% of the extra 1238 us is on the GPU side.** Both sides run the same compiled artifact and the same kernels;
  the only structural difference is that those 168 extern calls were turned into **child graph nodes**.
  A graph carrying 168 child-graph nodes simply executes ~1 ms slower than one with 168 ordinary kernel nodes.

This and the "4048 vs 354 graphs" above are **two symptoms of the same cause**. Moving the extern calls back into the main graph as ordinary nodes
that get patched fixes three things at once: the 984 us on the GPU, the graph count going from 4048 to 1, and leaving only ~173 us of host patching.

### An aside: a number unrelated to DG that is worth remembering

`vllm` -> `stock` is already +867 us (+7.7%): **just handing graph capture over from vLLM to inductor's
cudagraph_trees costs this much**. A real engine integration has to account for this as well, or simply have DG
sit underneath vLLM's own capture layer.

## That +8.7% GPU overhead is a lifetime bug, not a cost of DG (2026-09-22)

The previous section said DG adds 1238 us per step, 984 us of it on the GPU side. This is now resolved: **those 984 us have nothing to do with
DG; it is an ownership bug, and bare inductor cudagraph_trees is hit by it just the same.**

### Six hypotheses ruled out

Each was refuted by measurement along the way (each one was guessed first and then measured, and dropped once measured):

| Hypothesis | Measurement that ruled it out |
|---|---|
| child graph node scheduling is expensive | kernel count 51320, in-graph 49976, graphId count 2 -- identical in both configurations |
| different kernels are selected | all GEMM names / counts / mean durations agree to within 0.1% |
| grid is frozen | both `(132,1,1)`, same launch count, only one grid |
| scheduling table / semaphore is stale | the prepare kernel recomputes on every replay and writes the same address |
| stock is actually eager | stock's attention kernels also carry a graphId |
| host scalar `max_seqlen_k` is frozen | stock is frozen just as hard (frozen at 15, all 21 later steps are replays), yet stays flat as a line |

The remaining hard facts: **same kernel, same grid, same parameters (alignment 256, `splits=0`,
`sched=(17,)`, `bt=(8,128)`), same graph structure; under dynagraph attention averages 52.8 us,
under stock 26.6 us; the medians are the same but the long tails differ by 6x, and it gets worse as generation proceeds.**

### Mechanism: the pointer was captured, the storage was not held

Pinned down after consulting codex. It pointed out that my "stale scheduling table ruled out" item does not hold, for a very specific reason:

> In vLLM's FA3 source, once `scheduler_metadata` is passed in explicitly, `tile_count_semaphore`
> **aliases that tensor**, and FA3 skips its internal preparation; the non-split path only zeroes it **after** the forward.
> So looking only at which address the prepare kernel writes to does not verify the real consumer, the reset timing, or the **lifetime**.

And the key one:

> a child graph can retain a pointer to a tensor allocated outside its capture
> **without retaining that tensor's storage**.

`scheduler_metadata` is not among the operator's arguments -- vLLM allocates it fresh every step in `build()`, puts it into the
forward context, and the op reads it internally. At capture time its pointer is baked into the kernel parameters; **after capture that storage is
freed and the allocator reuses it for something else**. On every subsequent replay, FA3 reads and writes its own
scheduling table and semaphore in memory that now belongs to someone else. The output is still correct (scheduling only decides how the 132 resident CTAs split the work, not the result),
but the division of work gets more and more chaotic -- exactly "median unchanged, long tail exploding, worse the longer it runs, with spikes".

### Five-line verification

Hook `flash_attn_varlen_func` and store each call's `scheduler_metadata` in a global list
so it is never freed (`PIN_SCRATCH=1`). Reproduced in two independent runs, error +-10 us:

| | Per decode step |
|---|---|
| vllm captures on its own | 11201 us |
| stock | 12084 / 12067 us |
| **stock + pinned** | **11266 us** (-6.7%) |
| dynagraph | 13323 / 13316 us |
| **dynagraph + pinned** | **11269 / 11265 us** (-15.4%) |

**The three paths converge to within 0.6%.** The +8.7% GPU overhead disappears entirely.

### The conclusion has to be rewritten

- **DG's real cost on 14B decode is ~180-240 us on the host side, about 2% per step, and zero on the GPU side.**
  Xinwei's worry that "patching every round on the CPU absolutely won't work" does not hold at 14B; it does hold at 0.6B (10% of a 2.4 ms step),
  so **the lower bound on where it applies is model size**.
- This bug is **not DG's**: when vLLM's FA3 attention is captured under `cudagraph_mode=NONE`,
  whoever captures it gets hit; bare inductor cudagraph_trees also lost 6.7%. vLLM's own `use_full_cuda_graph=True`
  path is fine precisely because it replaces `scheduler_metadata` with a **persistent, zero-initialized, fixed-address**
  buffer (`flash_attn.py`: `self.scheduler_metadata = torch.zeros(1 + round_up(bs,4)*4)`,
  refreshed in place every step with `[:n] = ...`, with the tail cleared by `[n:] = 0`), and pins `max_num_splits`.
- **The general lesson for DG**: for any extern operator whose every call reads or writes a temporary buffer "allocated outside the operator
  and not in its argument list", capturing it is unsafe -- and **it does not error, it just silently gets slower** (this time)
  or silently goes wrong (with a different operator). The **declaration** route of `_capture_deps` / `_capture_tma` is exactly what should be extended
  to cover this: the operator declares "I have per-call scratch, where it is, and whether it needs zeroing", and DG uses that to
  take over the lifetime of that buffer, instead of silently capturing a dangling pointer into the graph.

### Methodology lesson: nsys does not expand graphs by default

The first profile looked completely normal -- kernel names, instance counts and durations were identical in both configurations. In fact
`nsys` defaults to `--cuda-graph-trace=graph`, **kernels inside graphs are not recorded at all**, and those statistics covered only
4% of the GPU work (the lm_head and sampling outside the graph). Any profile on this line must use
`--cuda-graph-trace=node`.

## Position: don't move the data, move the contract (2026-09-22, settled with Xinwei)

After chasing down the bug above, the real thing to answer is a design question: **when an engine hides its per-step metadata in a side
channel, what should our stance be?**

### First, acknowledge that the side channel is right

`unified_attention_with_output` takes only `(query, key, value, output, layer_name)`, and
`layer_name` is a **string**; the real `attn_metadata` is looked up at run time in a global registry by that string,
and `fake_impl` is an empty `return`. So during Dynamo tracing **nothing at all is known** about what this operator reads;
these tensors never cross the graph boundary, so naturally they never become placeholders and never appear on the `... = args` line,
and DG's text analysis has no name to grab.

**This is deliberate, and it is right.** `FlashAttentionMetadata` has 27 fields; among them `max_seq_len`
grows by 1 every decode step (measured 14, 15, 16, ...), `num_actual_tokens` follows the batch, and `causal` even changes
**type** (`bool` or `Tensor`). If these entered the graph signature, Dynamo would generate value guards,
which amounts to recompiling every step. Only the side channel keeps the signature stable at `(input_ids, positions, ...)`, so one compile serves every step.
SGLang uses the same scheme. Fighting this means fighting the whole ecosystem.

### The contract is narrower than it looks: content changes are harmless, identity changes are fatal

**For a captured graph, a change in the *content* of a side-channel tensor is harmless; only a change in its *identity* (address) is fatal.**

Of those 14 metadata tensors, `block_table`, `slot_mapping` and `seq_lens` are all persistent buffers
whose content alone changes, so capturing them into the graph is completely fine; the only one that bites is `scheduler_metadata`, freshly allocated every step.
So the requirement on the producer is not "don't use a side channel" but "**buffers in the side channel must have stable addresses**" -- allocate once,
refresh in place. The requirement is so small it barely changes the design, and it is checkable.

### Three things, no extra mechanism

Xinwei's judgment: **do not add a separate declaration mechanism for "scratch lifetime".**
That would be recording the problem; the right move is to make it go away -- **once FA becomes an operator that can be patched by parameters,
it no longer matters whether the scratch address changes; it is just one more pointer to patch**,
no different from a buffer in the arena.

1. **Tell the engine we are capturing.** If the engine has a replay-safe mode, turn it on, instead of setting
   `cudagraph_mode=NONE` to turn it off and then capturing ourselves (that is exactly how this bug came about).
   vLLM already implements it: `use_full_cuda_graph`.
2. **Patch extern calls by parameters instead of capturing them as child graphs.** The good news is that the FA side is simpler than expected:
   **the grid is constant at `(132,1,1)`** (persistent kernel; the grid is the SM count), frozen host scalars are fine too
   (stock frozen at `max_seqlen_k=15` stays flat anyway), and **the only thing that really has to change per call is the pointers**.
   And the node traversal in `_foreign_pointers` (`cudaGraphGetNodes` + `cuFuncGetParamInfo`
   to read kernel parameters) **is already the first half of this path** -- it was written for detection, but it is really infrastructure.
   Getting this done solves three things at once: the 984 us on the GPU, the graph count going from 4048 to 1, and leaving only host patching.
3. **Dangling-pointer detection as a backstop.** The protocol only covers cooperating producers; the fourth axis ("badly written code") requires that the uncooperative ones
   **fail loudly, not silently slow down**. This one is purely DG's responsibility, because DG is the party that
   **shows up uninvited to capture other people's code**.

At patch time there are three ways to know "where that scratch currently is": (a) the operator declares a way to fetch the current value --
this is where a registry really belongs, the same shape as `_capture_deps` ("who this name currently stands for");
(b) re-run the host-side first half of the operator on every call, which is too expensive; (c) **tell the engine we are capturing, it makes the scratch persistent,
and then nothing needs patching at all**. Engines with a contract take (c), third-party operators without one fall back to (a), and if neither holds, fall back to
opaque + a warning.

### How detection works

Instead of asking "what is written in the text", ask "**which pointers does the captured graph actually hold**" -- this is ground truth,
independent of whether there is a name. `_capture_ranges` + `_foreign_pointers`
(`torch/_inductor/dynagraph.py`):

- Use `torch.cuda.memory_snapshot()` to get all segments. Segments with `segment_pool_id != (0,0)` are
  **graph pools**: allocations made during capture stay there until the graph is destroyed, so they are safe; `(0,0)` is the default pool and gets reclaimed.
- Safe ranges = the graph pools + every lane of the arena + `input_store` + `static_inputs`.
- Walk every kernel node with `cudaGraphGetNodes`, enumerate its parameters one by one with `cuFuncGetParamInfo` to get
  (offset, size), and read the 8-byte ones as candidate pointers. **A value counts as a pointer only if it falls inside some live segment**
  (otherwise it is an int64 scalar); one that falls inside a live segment but outside every safe range -> **dangling pointer**.
- Switch `TORCHINDUCTOR_DYNAGRAPH_CHECK_CAPTURE` = `warn` (default) / `refuse` / `off`.
  One snapshot is taken per harvest, not one per site.
- False-positive check: on `probe_sdpa`, `probe_extern_child` and `probe_conv_child`, three probes known to be benign
  that all have extern child graphs, **zero warnings**.

## Is the annotation framework workable? Reconciling every pointer of real serving ops, one by one (2026-09-22)

What Xinwei asked for is a criterion: **"let the compiler directly know how to generate the CPU update code"**.
Translated into something measurable: **can every captured pointer be mapped to a named source**.
If yes, we can generate "`(node k, param offset o)` <- the address of source X for this call";
if even one does not match, that op cannot be annotated.

### How it was measured

`_capture_ranges` / `_foreign_pointers` (`torch/_inductor/dynagraph.py`) were originally written for detection;
they were changed to classify by **name**:

- Walk the captured child graph with `cudaGraphGetNodes`, and get each parameter's (offset, size) with `cuFuncGetParamInfo`;
- Read both launch forms: `kernelParams` (an array of pointers) and `extra` (a packed blob;
  cutlass kernels with clusters take this path). **Reading only the former misses 57% of the nodes**; the measured ratio is a stable 2:3;
- **Regardless of the declared parameter size, scan word by word at 8-byte alignment.** FA3 passes a params struct of several hundred bytes by value,
  with all the pointers hidden inside it; looking only at parameters with `size == 8` finds nothing on exactly the op that most needs checking;
- A word counts as a pointer only if it falls inside a live allocation (sizes, strides and the like do not), and it is then matched against:
  each actual argument of the op, the declared per-call buffers (the newly added `torch/utils/_capture_scratch.py`),
  the arena, the graph's pool, and the engine's resident large blocks.

### Finding

**For the three classes of ops in real serving, 504 sites: every captured pointer has a name, 0 unaccounted for.**
Collecting enough names did not take an elaborate mechanism, just two lines of push: at the moment an op runs, it registers the
buffers it reads this time. Of the 17 pointers in attention, 8 could not be matched at all before declaration
(512B x7 + 2048B x1); after 25 lines of declarations, all of them became named sources such as `declared:query_start_loc`.

### Results (Qwen3-0.6B, 28 layers, three harvests, 504 sites)

**Scope: these are numbers for a single variant, plain varlen** (single GPU, so DCP is off; decode does not trigger cascade).
The registration statement in the experiment is "iterate over the tensor fields of `attn_metadata`", which does not cover other variants -- see
"A dozen-plus variants under one op name" below.

| Site | Pointers | **Unaccounted** | Count | Resolved to |
|---|---|---|---|---|
| `mm` | 17 / 19 / 24 | **0** | 336 | `kw:out`, `arg1`, cuBLAS workspace (`graph-pool`) |
| `unified_attention_with_output` | 1 / 17 / 18 | **0** | 84 | `kv_cache[0]` x6, `arg3` x3, `query_start_loc` x3, `scheduler_metadata` x3, `block_table` |
| `unified_kv_cache_update` | 7 | **0** | 84 | `arg1` x2, `kv_cache[0]` x2, `slot_mapping`, `layer._k_scale`, `layer._v_scale` |

Two things that were forced out along the way deserve their own note:

**One op cannot register on behalf of another.** In the first version I hooked only `FlashAttentionImpl.forward`
and had it register for both attention and kv_cache_update. As a result `unified_kv_cache_update` had 2
unmatched pointers, 56 sites got the record left over from the **previous step** (the addresses happened to still be right), and 28 sites got nothing at all and
honestly reported 5 unaccounted. That is exactly why `taken()` uses pop rather than read: a second take returns `None`,
instead of letting one op's buffers pass as another's. After changing it so that `do_kv_cache_update` registers for itself, this went to zero.

**The last two pointers are scalars hanging off the module.** The last two arguments of `reshape_and_cache_flash` are
`layer._k_scale` / `layer._v_scale` -- neither actual arguments nor metadata, but two `torch.Tensor`s the op reaches through `self`,
packed into the same 512B allocation. They are never reallocated, so they happen to be safe,
**but the consumer has no way of knowing that**. This is exactly why the declaration scheme exists: not "help find them", but
"let whoever knows say so".

### Why push and not pull

The first version had the consumer pull: the producer registers an accessor, and after capture the consumer asks "which buffers are you using right now".

```
accessor called 168 times: 112 times got attn_metadata, 56 times got None
```

**In one of the three harvests the engine's forward context was not available at all**, and a third of that harvest's pointers
silently lost their source. The reason is that DynaGraph's harvest **re-runs the wrapper itself**, with no guarantee that it is still inside the engine's
own calling context. So pull is the wrong shape -- it depends on "the context still being there when the consumer asks".

Push does not have this problem: the op registers at the moment it reads these buffers; it comes for free and cannot come up empty.

### Multiple backends: registration belongs to the backend, not the front end

vLLM's attention backends span 20 files and at least 9 `AttentionImpl` subclasses
(`flash_attn`, `flashinfer`, `triton_attn`, `flex_attention`, `gdn_attn`, `mamba1/2`,
`rocm_*`, `cpu_attn` ...), 13 of which implement their own `do_kv_cache_update`. So the question is:
where does that one `record()` line go?

Two candidates: the **front end** (the `unified_attention_with_output` op, which, once it has the context, iterates over
all tensor fields of `attn_metadata`), or the **backend** (each `AttentionImpl` registers what it reads this time).

**It has to be the backend.** The front end looks like "one place covers every backend", but in practice the registration is incomplete, and it misses silently:

1. **A buffer is not necessarily a tensor field of the metadata.** FlashInfer does
   `attn_metadata.prefill = FIPrefill(wrapper=prefill_wrapper)` -- the real paged_kv
   indices live **inside** that wrapper object. Iterating over the dataclass's tensor fields gets none of them.
2. **A buffer does not necessarily pass through the metadata.** `trtllm_workspace_buffer` is a
   **module-level global** in flashinfer.py, lazily allocated; `self._workspace_buffer` hangs off the impl. Nothing in the arguments
   or the metadata points to them. The front end has no way to see them.
3. **We just ran into this one ourselves.** `layer._k_scale` / `layer._v_scale` are reached through `self`
   and are not in the metadata. The front-end approach would leave these 2 pointers in the "unaccounted" column forever.
4. **The front end's "generality" is fake.** Even if the iteration succeeds, the names that come out are still
   **backend-specific field names** like `scheduler_metadata` / `paged_kv_indptr` -- not one bit more general,
   it just moves backend knowledge to a place that cannot see the backend. Even the attribute names do not line up:
   FA reads `layer._k_scale` (a tensor), FlashInfer reads `layer._k_scale_float` (a Python float).

Conversely, backend registration costs close to nothing: **the backend is holding these buffers at that very moment**, one extra `record()` line adds no work,
and whether anything was missed can be judged locally -- whoever writes this kernel call can see what they passed.
What the framework should provide is a convenience function (sweeping up the metadata's tensor fields + kv_cache + the passed-in scales together),
so that most backends need one line, and backends with extra state (FlashInfer's wrapper, workspace) add theirs on top.

The backend knows which op name to use as the key: each impl method is called by exactly one op
(`FlashAttentionImpl.forward` <- `unified_attention_with_output`,
`do_kv_cache_update` <- `unified_kv_cache_update`). A cleaner variant is **the front end opens the scope,
the backend fills in the contents** -- the op pushes a key on entry, and the backend's `record()` fills the current frame without a key.
That divides the responsibilities cleanly: **the front end knows "which capture site this is", the backend knows "what was read this time"**, and neither guesses for the other.

This experiment ran backend registration (hooking `FlashAttentionImpl.forward` and
`FlashAttentionImpl.do_kv_cache_update`); the table above (504 sites, all 0) is the result of that shape.

### A dozen-plus variants under one op name, with two kinds of consequences

`unified_attention_with_output` is not one kernel; it is a branch tree. FA's `forward`:
`use_cascade` -> `cascade_attention` (two varlen calls + `merge_attn_states`), otherwise a single
`flash_attn_varlen_func`; `dcp_world_size > 1` -> `_forward_with_dcp`, which branches further on
`max_dcp_context_kv_len == 0` and `split_dcp_context`, up to three varlen calls + merge.
FlashInfer has more: the cascade / prefill / decode paths can run two of them **within the same call**,
and prefill further splits into trtllm, `BatchDCPPrefillWrapper`, `BatchPrefillWithPagedKVCacheWrapper`,
`BatchAttentionWithAttentionSinkWrapper`.

**Pointer declaration does cover the variants.** `record()` happens inside the branch that actually runs, so it records what was really read this time.
The key is just a mailbox that is emptied right away (record -> capture -> taken); one op name is enough, no per-variant key is needed.
Over-declaring is harmless; under-declaring is fatal -- and "whether anything was under-declared" is exactly what the person writing that branch can judge locally.

**Variant selection itself has to take a different path: declared dependency + patch selector, not re-capture.**
Which path is taken is decided by data (`attn_metadata.use_cascade`, `num_prefill_tokens > 0`,
`max_dcp_context_kv_len == 0`, `prefill_use_trtllm`). The captured child graph freezes
**the set of kernels selected that time**; if the next step switches variant, replay runs the wrong kernels.

The mechanism is in fact already built, and it has exactly the shape of "declare the dependency, patch the graph by hand":

- A site whose extern topology can change is a **SWITCH (conditional) node**, with one body per topology
  (`site_topos` / `site_graphs`);
- `key_bodies: dict[hkey, list[int]]` stores "which body each site uses under this hkey" (`_combo`);
- The host writes the index into `ctx[BODY0 + s]`, and thread 0 of the device-side planner does
  `cudaGraphSetConditional(handle, body)`. **This is a patch, not a re-capture** --
  a new combination pays for one harvest, after which switching back and forth between variants at the same shape is just writing an index.
- And the fourth element of `_hkey` is `_deps_key()`. **The declaration channel is already wired through.**

So the gap is not "there is no SWITCH", nor a missing new registration table -- **the shape of `_capture_deps` is right**.
What is missing is someone actually declaring, plus two relaxations on the consumer side.

**Declaring does not require us to know any of vLLM's abstractions.** The interface of `_capture_deps.register` is
`(fully qualified op name, argument names, pure function)`, and **the body of the resolver is the producer's own business**.
The only existing registrant in torch looks like this:

```python
# torch/distributed/distributed_c10d.py:8244
_capture_deps.register("_c10d_functional::all_reduce", ("group_name",),
                       _capture_identity_of_group)
```

`group_name` is a string argument, and `_capture_identity_of_group` internally uses c10d's own
ProcessGroup abstraction -- dynagraph has no idea what a ProcessGroup is. The docstring of `_declared_dep`
is a single line: "Nothing here knows any operator."

The vLLM side is isomorphic: in `unified_attention_with_output(query, key, value, output, layer_name)`,
`layer_name` is an actual argument, and at every call site it is a **literal constant** (`_declared_dep` takes exactly
the literal at that position in the generated code). So it is

```python
_capture_deps.register("vllm::unified_attention_with_output", ("layer_name",), _branch_of)
```

Whether `_branch_of` looks up the forward context or its own registry is vLLM's internal affair.
The contract has only three clauses: **return a hashable value, be cheap, never raise** (the context may be absent during warmup / profile runs;
the producer handles that itself).

The evaluation timing is already right: `_deps_key()` is computed on the host, in the runner's `__call__`, before launch,
with no device->host sync needed.

(The placement here differs from `_capture_scratch`, which is worth noting: scratch is **push**, registered by the impl at
the moment of capture, because the pointers are baked in at capture time; deps is **pull**, the consumer calls the resolver on the arguments every step,
because it drives the per-step selector. Where each one sits is determined by "when the value is needed", not a matter of style.)

Today nobody declares, so `_deps_key()` is empty for attention; thus at the same shape the cascade switch does not change the hkey,
`key_bodies` returns the old body, and the device-side planner's comment states it plainly --
"Set only when the shape changed, so a skipped planner leaves the last selection standing"
-- **the previous selection stays in place**, and it silently runs the wrong thing. The self-check cannot save it either: once the comparison is done it does `ex.verified.add(key)`,
so each shape key is verified only once.

**Two ceilings that will be hit first:**

- The per-site body limit is `_max_graphs()`; beyond it, `_fallback("extern-topology")`.
  The code comment's reference point is "three for cuDNN conv, two for cuBLAS"; FA + FlashInfer have more variants than that.
  This limit should be set by the number of branches the op declares, not by a global constant.
- Harder still is `cond-topology`: **a site that is already inside a conditional body gives up outright as soon as a second topology appears**
  ("a second topology there would need a conditional of its own").
  Nested variants such as cascade further splitting into trtllm / wrapper will run straight into it.

### This is only a necessary condition

**All pointers accounted for != this op can be patched by parameters.** Two more things are needed:

1. **Whether func / grid / cluster change with shape.** These two ops are exact opposites:
   - `unified_attention`: **does not change**. FA3 is a persistent kernel, the grid is always `(132,1,1)` (the SM count),
     and freezing the host scalars does no harm (stock, frozen at `max_seqlen_k=15`, still stays flat).
   - `mm`: **does change**. It is right there in our own logs -- at the same site, as `s72` goes from 7 to 14,
     the topology tuple `(type, clusterX, clusterY, clusterZ, coop)` changes from `(0,4,1,1,0)` to
     `(0,8,1,1,0)`; cuBLAS switched kernels and even switched the cluster shape.
     On an already-instantiated exec, `cuGraphExecKernelNodeSetParams` **can only swap func**,
     not the cluster (a node attribute) -- across clusters it measured CUDA 912 / 715.
     So when the cluster changes we have to switch SWITCH body (fortunately there are only 3 kinds); swapping func inside a body costs
     0.73-0.81 us/node. In addition, every new shape still has to ask cuBLAS on the host once which kernel it wants.
2. **The final criterion is how many child graphs still have to be captured at runtime.** Everything above is static reconciliation; the real number is
   the call count of `torch.cuda.CUDAGraph.__init__`: today DynaGraph makes 4048 graphs, vLLM 354.
   Once the work is done this number should drop toward 1; if it does not, some sites are still falling back to opaque capture,
   and however clean the reconciliation is, it does not count.

## Another thing to check: `runtime-mismatch` in the `piecewise` configuration

The configuration that keeps vLLM's piecewise compilation and only hands graph capture to inductor hit `runtime-mismatch` --
the self-check found that the replay result disagreed with eager, and so it fell back. The final output was correct (because it fell back),
but this shows that on the child graphs cut out by piecewise, DynaGraph has a case where it really computes the wrong answer.
**This is more urgent than the performance issues above.**

## Reproduce

```bash
# run from the repository root (the directory that contains dynagraph/)
source dynagraph/setup/use_main.sh
# wide-bs comparison (the table above)
for m in vllm dynagraph; do
  MODE=$m BS=1,2,3,4,5,6,7,8,9,10,11,12 MAXTOK=8 CUDA_VISIBLE_DEVICES=6 \
    python -W ignore dynagraph/serving/probe_vllm_decode.py
done
# latency (source use_main.sh first)
GPU=3 MODEL=Qwen/Qwen3-14B GPU_UTIL=0.6 BS=8 N=3 ROUNDS=3 bash _latency_sweep.sh
GPUTIME=1 MODE=dynagraph MODEL=Qwen/Qwen3-14B GPU_UTIL=0.6 BS=8 python -W ignore dynagraph/serving/probe_vllm_latency.py
# fixed bs, look at steady state
MODE=dynagraph BS=2,2,2 CUDA_VISIBLE_DEVICES=6 python -W ignore dynagraph/serving/probe_vllm_decode.py
# child-graph route turned off
MODE=dynagraph EXTERN_CHILD=0 BS=1,2,3 CUDA_VISIBLE_DEVICES=6 python -W ignore dynagraph/serving/probe_vllm_decode.py
```

The probe prints harvest counts and child-graph counts broken down per shape; `TRACE_CAP=N` prints the call stacks of the 1st/Nth/(N+1)th
`CUDAGraph` construction -- that is how "168 extern sites" was located.
