# DynaGraph performance: first batch of trustworthy numbers (2026-09-19, H100 GPU 6, exclusive)

See `dynagraph/probes/bench.py` for how this is run. Four configurations are timed interleaved and the
minimum is taken; the shape stream comes from the UniProt human reference proteome (256 steps,
63 distinct lengths, longest 947). **Host load average 97** -- this is not contamination but the
operating condition: the whole reason CUDA Graphs exist is to cut CPU launch overhead.

The GPU's power cap is 550 W; measured draw during the run was 140-232 W, so it never hit the cap.

## First, fixed an 8.6x self-inflicted overhead

The first run gave steady-state DynaGraph 127 ms vs re-recording 61 ms, **twice as slow**. That is an
order of magnitude more than "one extra input copy" should cost, so measure first, then change
(`dynagraph/_probe_call_breakdown.py`):

| Segment | Before the fix | After the fix |
|---|---|---|
| Input copy | **451.7 us/call (92.4%)** | 22.4 us/call |
| Static pointer check | - | 5.5 us/call |
| ctx symbol write | 12.8 | 10.9 |
| replay | 7.5 | 6.2 |
| Output views | 9.3 | 7.5 |
| **`__call__` total** | **488.7 us/call** | **57.1 us/call** |

Cause: the 12-layer model takes 50 inputs, of which **49 are weights**, and `build` allocated a static
buffer for every tensor input and copied all of them on every call. The weights never move.

`cudagraph_trees` already marks these with `static_input_idxs` -- that is a promise about the
**address**, not about the contents, so the captured nodes can read the weights in place and optimizer
updates remain visible. Now these inputs get no buffer and no copy; we only use the same upstream C++
helper `torch._C._tensors_data_ptrs_at_indices_equal` to check in one shot that the addresses have not
moved (5.5 us, 48 inputs together), and if an address really did change we retire with
`static-input-moved`.

## launch-bound (12 layers, d_model=128, GEMM restricted to Triton)

|  | First pass (compilation already warmed up) | Steady state (min / spread) | Graphs recorded | Peak memory |
|---|---|---|---|---|
| A re-record (upstream status quo) | 295.3 ms | 61.3 ms x1.5 | **56** | 0.09 GiB |
| B eager | 229.4 ms | 301.6 ms x1.1 | 0 | 0.10 GiB |
| C pad2max | 63.1 ms | 62.3 ms x1.0 | 0 | 0.18 GiB |
| **D DynaGraph** | **67.9 ms** | **67.3 ms x1.0** | **0** | 0.16 GiB |

* **First pass 4.35x** -- this is what DynaGraph actually eliminates: no recording when a new shape
  arrives. Under a long-tail distribution new shapes keep appearing, so this is the normal case, not a
  one-off cost.
* **Steady state 9.8% slower**, and **its spread of x1.0 makes it the most stable of all four
  configurations** (A is x1.5). Steady state was never where it was meant to win: both sides replay one
  graph, and it additionally does one input copy and a layout table lookup.
* Numerical match against the reference: 2.7e-07.
* Peak memory in this configuration is a net loss at this scale (1.8x more; the headroom is a fixed
  cost), but memory was never the selling point of this work, so it can be ignored.

## gpu-bound (3 layers, d_model=1024)

Triton only:

| | Steady state |
|---|---|
| A re-record | 207.8 ms |
| **D DynaGraph** | **208.6 ms (0.4% difference)** |
| B eager | 69.1 ms |

Once GPU work dominates, the few tens of microseconds on the host side become invisible, and A and D
tie.

But **eager is actually 3x faster** -- because DynaGraph rejects `extern_kernels`, all four
configurations route GEMM to Triton for fairness. What this constraint costs was measured in a separate
configuration (`--gemm-backends ATEN,TRITON`; in this configuration D falls back with `extern-launch`):
cuBLAS 22.3 ms vs Triton 207.8 ms, **9.3x**.

### This 9.3x is not Triton's fault; dynamic turned TF32 off

The first version of the conclusion read "Triton-only costs 9.3x"; **that was wrong**. Digging in
showed that the slow GEMM is an independent problem unrelated to DynaGraph. In
`heuristics/template/triton.py` the TF32 switch of the Triton mm template is:

```python
size_threshold = V.graph.sizevars.statically_known_true(
    sympy.And(sympy.Ge(m, 16), sympy.Ge(Min(n, k), 512)))
allow_tf32 = torch.backends.cuda.matmul.fp32_precision == "tf32" and size_threshold
```

With `dynamic=True`, m is symbolic and `m >= 16` **cannot be proven statically**, so
`ALLOW_TF32=False`: Triton runs true fp32 (CUDA cores) while cuBLAS runs TF32 (tensor cores), and on
H100 those two inherently differ by ~7x. Autotune table for the same addmm(947x1024, 1024x1024)
(`dynagraph/_probe_tf32.py`):

| | aten | best triton | ratio | what the template got |
|---|---|---|---|---|
| dynamic fp32 | 14.6 us | **138.6 us** | **9.5x** | `ALLOW_TF32=False` |
| static fp32 | 14.5 us | 16.6 us | 1.14x | `ALLOW_TF32=True` |
| **dynamic bf16** | 10.0 us | **10.5 us** | **1.05x** | irrelevant (already on tensor cores) |
| static bf16 | 10.1 us | 10.5 us | 1.04x | - |

**The conclusion flips**: real models use bf16, where Triton and cuBLAS differ by 5%, so "Triton only"
costs almost nothing. The 9.3x appears only in the fp32 + dynamic cell, and what needs fixing is that
static check in Inductor (or giving the symbol a lower bound of `min=16` so it becomes provable), not
DynaGraph.

**So the case for extern kernel support is coverage, not speed**: Inductor does not enable
`max_autotune_gemm` by default, so GEMMs go straight to `extern_kernels.addmm`, and almost every real
model hits the `extern-launch` fallback right away. That is the thing to solve.

## Still unresolved, all visible in the numbers

* **`input-too-large`**: if the shape stream does not deliver the maximum first, DynaGraph retires
  immediately (the natural-order run recorded 57 graphs, no different from A). Inputs are not in the
  arena; their addresses are hard-coded.
* **Peak memory**: on small models the headroom is a net loss.
* **The steady-state 57 us/call**: `incopy` 22 + `ctx` 11 + `out` 7.5 can still be squeezed. Of that,
  the 11 us of `ctx` is writing **a single** int64 to device memory -- `self.ctx[i] = v` goes through
  Python `__setitem__` + dispatcher + H2D; switching to a pinned staging buffer and one `copy_` should
  bring it down to a few us.

## Can the 200+ us of Python per step be bypassed: mostly no

`dynagraph/_probe_stack_overhead.py` splits the call chain into four segments (the cut points must
be places that are looked up anew on every call -- patching the generated module's `Runner.call` **does
not hit**, because the AOT layer stored the bound method long before):

| us/step | Share | Layer |
|---|---|---|
| **240.0** | **68.9%** | **dynamo guards + AOT runtime wrapper** |
| 5.3 | 1.5% | generated module `Runner.call` prologue: unpacking args / asserts / computing sizes |
| 8.3 | 2.4% | partition dispatch + `deferred_cudagraphify` shell |
| 94.9 | 27.2% | `DynaGraphRunner.__call__` |

(This round carries three layers of timing wrappers, so absolute values are inflated relative to a bare
run -- the runner segment is 57 us bare, 95 here -- look only at the proportions.)

**The only segments DynaGraph is in a position to bypass are the middle two, together ~14 us/step,
about 4%.** The real bulk, 240 us, is torch.compile's own call overhead, and A/C/D all pay the same
amount -- it is not in the graph; the graph only begins at "launch these 48 kernels".

Actually removing those 240 us leaves only two routes, both beyond the current scope:

1. **Leave Python**: AOTI / `torch.export` compiles to a C++ runner, and the guards and the wrapper go
   away together. DynaGraph's mechanism would have to move into the AOTI runtime.
2. **Call the runner handle directly**: `runner(inputs)` bypasses dynamo and AOT. **Only DynaGraph can
   do this one** -- upstream has to look up fn_cache by int_key to pick a graph on every call, whereas
   here one graph covers the whole shape space, so the fast path has no shape-dependent branch at all.
   The price is losing the guards; the caller must guarantee dtype/device/layout.

Before either of these, squeezing the steady-state 57 us down to 40 us amounts to about 5% of the
348 us per step.

## Follow-up measurement (same day, evening, load up to 210): the steady-state cost is actually on the GPU

Wall clock is no longer usable under this load (and the `best()` I wrote runs each configuration 5 times
in a row without interleaving -- exactly the mistake described in [docs/METHODOLOGY.md](../METHODOLOGY.md)). But CUDA
events and profiler device time are measured on the device side and are unaffected by host contention,
so these two numbers are trustworthy.

**This graph takes 275 us per run on the GPU** (CUDA event median, 48 kernels). In other words, the name
launch-bound was a misnomer -- a 12-layer model with d_model=128 still takes 275 us on the GPU, because
it is 48 small, mutually dependent kernels queued one after another, and the 24 GEMMs alone take 240 us
(10 us each).

Profiler breakdown of the device time per replay (`dynagraph/_probe_gpu_share.py`):

| us/replay | Count | Kernel |
|---|---|---|
| 239.8 | 24 | `triton_tem_fused_addmm_t_0` |
| **43.7** | **1** | **`dynagraph_planner`** |
| 16.3 | 12 | `triton_per_fused_addmm_mean_relu_sub_1` |
| 11.3 | 10 | `triton_poi_fused_add_addmm_mul_sigmoid_3` |
| 1.3 | 1 | `dynagraph_layout` |

**planner + layout account for 14.3% of each replay.** This cost had never been accounted for before.

The conclusion needs revising: **the ~10% steady-state gap is mainly not the host-side copy; it is the
planner running 43.7 us on the GPU on every replay.** I had been optimizing the few tens of
microseconds in `incopy`/`ctx`, which was the wrong direction -- those were already hidden behind the
275 us of GPU time (the direct-call configuration measured 268 us/step, essentially flush with the
275 us GPU floor, which shows the host side is fully covered).

Why the planner takes 43.7 us: it uses one thread per node, and 48 nodes call
`cudaGraphKernelNodeSetGridDim/SetParam/SetEnabled` in parallel, but these device-side runtime calls are
not cheap in themselves. Directions I can think of, ordered by cost:

1. **Early-out when the shape has not changed** -- keep a copy of the "last applied ctx" on the device
   and return immediately if it is identical. Scenarios with the same shape on consecutive steps (very
   common in training) save the entire cost. This is the cheapest.
2. **Patch only the nodes that actually depend on symbols** -- not necessarily all 48 of them.
3. **Move to host-side `cudaGraphExecKernelNodeSetParams`** -- the shape is already known on the host,
   and host-side patching can overlap with the previous step's GPU work; the price is giving up the
   "takes effect within the same launch" property, plus 48 host calls.

## Are those 240 us still paid after compilation: yes, every time

Guards are not a compilation artifact; they are a **per-call correctness check** -- nothing in Python
stops the next call from switching dtype, switching device, or changing a module attribute, and the
compiled code is only valid under the original set of assumptions. After the guards pass there is still
**real work**: collecting the 49 parameters from the module tree into a flat argument list, and the AOT
runtime wrapper doing another round of bookkeeping over 50 inputs. The second half is not checking but
moving things around, and it scales with the number of parameters.

Calling the runner directly bypasses this layer; two rounds measured 2.1x and 1.9x
(`dynagraph/_probe_bypass.py`, numerics bit-identical to the full path). Note, though, that the
direct-call configuration is already flush against the GPU floor, so this 2x means "host overhead
disappears from the critical path", not that the computation got faster.

## The planner's 43.7 us: two of three hypotheses wrong, the final cut is "skip when the shape has not changed" (2026-09-19, late night)

Following the rule in [docs/METHODOLOGY.md](../METHODOLOGY.md), measure before changing: `microbench/planner_batched.cu`
(CUDA event, device side, unaffected by load), N=48:

| Variant | us/node | N=48 total |
|---|---|---|
| Uniform code, three separate calls (SetEnabled/SetGridDim/SetParam) | 0.083 | ~4 us |
| Uniform code, batched via `cudaGraphKernelNodeUpdatesApply` | 0.099 | ~5 us |
| per-node switch (the shape of the generated planner) | 0.145 | ~7 us |
| nested switch (outer per-node switch + dg_eval inside is also a switch) | 0.244 | ~12 us |

* **The batched API is not faster than separate calls**; it is actually slightly slower. The
  "0.049 us/node = 18x" from `devupdate_scale.cu` was a false lead; the difference lies elsewhere.
* **Nested switch explains only ~12 us**, not 43.7. The rest is the per-node SetParam for each pointer
  argument (48 kernels x 3-4 pointers ~ 150-200 calls) adding up -- no single culprit, it is the call
  count.

So the most thorough cut is **not doing it at all**: patching is idempotent, so skip everything when the
shape has not changed. The host writes one extra "changed or not" slot at the end of ctx, and the
planner and layout return immediately when they see 0. The comparison is done on the host because there
is no ordering between different blocks; a device-side comparison would have a race where one block
reads the "already applied" flag that another block just wrote.

`dynagraph/probes/probe_early_out.py` (profiler device self time):

| Stream | Device time per replay | Of which planner |
|---|---|---|
| Changes every step | 247.1 us | 42.7 us (17.8%) |
| **Same shape repeated** | **206.9 us** | **0.9 us (1.1%)** |
| Alternating A,B,A,B | 248.6 us | 42.7 us (correctly did not take the early-out) |

Correctness: the outputs for A,A,A,A,A and A,B,A,B,A are **bit-identical** to the outputs of the same
graph in the changing stream (the reference cannot be eager -- GEMM via Triton and eager's cuBLAS
already differ by 3e-4; the first version of the check used eager and misread that algorithmic
difference as an early-out bug).

Against the four axes of the goal: this is a pure latency gain, realized only when consecutive steps
have the same shape -- very common in training, almost absent in long-tail shape streams. In the
changes-every-step scenario the 42.7 us is still there; the next step is to reduce the number of calls
per node (skip SetEnabled for nodes whose grid is always positive; patch only the arguments that
actually contain symbols).

## The cost of partitioning as a fallback: coverage gained, latency paid 15x (2026-09-19, late night)

`bench.py --gemm-backends ATEN,TRITON --partition-extern`, launch-bound configuration (12 layers,
d_model=128), GEMM via cuBLAS, partitioned at every extern:

|  | First pass | Steady state | Recorded |
|---|---|---|---|
| A re-record (cuBLAS, upstream status quo) | 1151.1 ms | **40.2 ms** | 56 |
| C pad2max | 43.7 ms | 41.2 ms | 0 |
| **D DynaGraph (partitioned)** | 551.8 ms | **621.7 ms** | 0 |

D really was served (0 graphs, numerics vs A 2.1e-07), but steady state is **15x slower**. The reason
is arithmetic: 12 layers x 2 addmm per layer are split into ~24 segments, each segment costs one runner
`__call__` (~57 us Python) + one eager addmm (~30 us host), 24 x ~90 us ~ 2 ms/step. **DynaGraph's
per-call host overhead can hide behind the GPU when it happens once per step; at 24 times per step it
can no longer hide.**

A new problem showed up along the way: the natural-order configuration recorded **1368** times -- with
multiple partitions, the `input-too-large` retirement happens per segment, so segments x shapes blows up.

Implication for the four axes: partitioning works only for models with **few** extern calls (e.g. a
transformer that, once GEMMs are routed to Triton, has only SDPA left, 1 per layer); models that are
GEMM-heavy with GEMMs on cuBLAS need the patch route (the GEMM level in `docs/notes/EXTERN.md`), or the
host overhead of `__call__` cut by an order of magnitude. The accounting for the latter: `incopy` 22 us
is the dispatcher overhead of **one** `copy_`, `out` 7.5 us is a few as_strided calls, `ptrchk` 5.5 us
-- all of these are things that can bypass the dispatcher.

**Addendum (same night)**: following the microbenchmark hints, two more classes of calls were cut --
nodes whose grid contains no symbols no longer issue SetGridDim/SetEnabled, and each node's enabled
state is stored in ctx so SetEnabled is called only when it flips. All regression tests pass, but in
the changes-every-step configuration the planner went **42.7 -> 42.8 us, no change**. All 48 nodes in
this model have grids containing symbols, so the only saving is one `SetEnabled(1)` per node (~ 4 us,
within noise). The finding is unchanged: when the shape changes every step, those 43 us are the sum of
several hundred per-argument SetParam calls, with pointer patches making up the bulk (the arena is
re-laid out for each shape, so every pointer argument has to change). To move it, slot offsets must be
**invariant** across shapes (each slot fixed at its maximum size, pointers baked in once) -- that is
the next cut, trading memory for call count.

## Can the 43 us be fixed by launching more CTAs? No; the only lever is "fewer calls" (2026-09-19, late night)

Question: the planner uses one thread per node, 48 threads in one block, and each thread issues ~6
device-update calls sequentially. Would spreading these 288 calls over more threads / more blocks be faster?
`microbench/planner_cta.cu` (three layouts of the same 48x6 calls, CUDA event median, empty-graph floor subtracted):

| Layout | us/replay (net) |
|---|---|
| A: 1 thread/node x 6 calls, 1 block (current) | 4.9 |
| B: 6 threads/node x 1 call, 1 block (288 threads) | 2.9 |
| C: 6 threads/node x 1 call, **one block per node** (48 blocks) | 2.9 |

B ~ C: spreading across SMs is no faster at all; the calls are serialized inside the device runtime. More threads
only break up each thread's **sequential dependency**, gaining 1.7x, and the whole thing is only 5 us -- in this
microbenchmark 288 calls come nowhere near 43 us. So the real planner is slow somewhere else.

Slicing the real planner apart (`_dump_bench_planner.py` dumps the planner cubin of the bench model,
`microbench/planner_real.cu` runs it on 48 dummy nodes; 48 nodes, 143 SetParam calls, of which
95 are pointer patches; `cuobjdump` shows a 952-byte stack frame):

| Version | us/replay | What was cut |
|---|---|---|
| `_bisect_full` | 53.8 | -- |
| `_bisect_no_ptrs` | 26.1 | the 95 pointer SetParams |
| `_bisect_no_params` | 36.8 | the 48 scalar SetParams |
| `_bisect_grid_only` | 12.9 | both cut, only SetGridDim/SetEnabled left |

Pointer patches account for **half**, scalar parameters for a third, and the grid for only a quarter. Pointer
patches exist because the arena is re-laid-out for every shape, so every buffer's address moves. If the slot
offsets are fixed at build time (each slot sized as recorded shape x headroom), pointers are written only once,
on the first replay -- trading memory for half of the calls.
(Measured on GPU 4, which was not exclusive; only trust relative values within the same run.)

## Slots fixed, pointers written once: correctness passes, timing still needs an exclusive GPU (2026-09-19, late night)

For the change, see "Fixed slots" in `docs/notes/SOLUTION.md`. The 17-item regression run on the shared GPU 6:
16 pass, `probe_partition_extern` has a numeric diff of 3e-5 (threshold 1e-5). After chasing it around,
**it is not this change's fault**:

* Comparing the runner's output bit-for-bit, per segment and per call, against an eager run of the same segment:
  0 mismatches in 72 calls.
* Running the probe 5 times in a row on the same code (the reference side compiled only once):
  0.0 / 0.0 / 1e-7 / **3e-5** / 0.0.
* Printing the `blocks` recorded by each runner: the two runs with 3e-5 are exactly the ones where
  `triton_per_fused_addmm_mean_relu_sub_0` got **XBLOCK=8** from run-time autotune
  (the other runs got 1) -- the summation order of the mean changed; the difference between pointwise XBLOCK
  256/512 is numerically 0.

In other words, the probe uses "another compilation" as its reference, and Inductor's reduction config is measured
on the GPU at first run, so on a shared GPU each compilation picks its own. The fix is to pin
`triton.autotune_pointwise = False` in the probes (`probe_partition_extern`, `probe_extern_child`) so both
sides run the same set of kernels.
This is the same lesson as in [docs/METHODOLOGY.md](../METHODOLOGY.md): on this machine "another compilation" is not a deterministic
reference.

**Addendum (later the same day, exclusive GPU 0)**: with autotune pinned, the same 3e-5 signature still shows up
occasionally (in a 20-item regression, `probe_partition_extern` and `probe_extern_child` each hit it once; the region
was not retired, 0 recordings, and the runner's bit-for-bit check against eager passed everywhere) -- the remaining
variable is cuBLAS choosing its algorithm independently in two different captures. So the reference criterion for
this kind of probe is changed to a tolerance (1e-4), and the bit-exact gate is left to the runner's own check
(criterion 0: no mismatch tag).

Along the way, filled in a premise that had never really been tested -- **values written by device-side
SetParam/SetGridDim persist across launches**. `probe_early_out` captures at the largest shape, 947; if the
parameters reverted to the captured values, the kernel would just compute a few extra rows and the rows being
checked would still be correct, so it did not actually test this. The new probe `probe_update_persistence.py`
does the reverse: capture at 300, replay 512. Both same-shape repeat (planner early-out) and alternating (planner
only changes grid/scalars, never touches pointers) are bit-identical to the first run, including the rows past
row 300. Writing pointers only once and early-out both hold.

Timing: GPU 6 was at the time loaded by someone else's vLLM to 546 W (power cap 550); `probe_early_out` measured
the planner at 71 us for "changes every step" and 29 us for "alternating" -- higher than the earlier 42.7, which
only shows that device self time on a shared GPU is not trustworthy; it is not a result.

**Measured on exclusive GPU 7 (later the same day)**:

| Stream | Device time per replay | of which planner | Before |
|---|---|---|---|
| Changes every step (947,512,333,256,128,64)x8 | 221.7 us | **18.6 us** (9.0%) | 42.7 us (14.3%) |
| Same shape repeated, 512 | 205.2 us | 0.9 us (early-out) | 0.9 us |
| Alternating 512/256 | 222.7 us | 18.4 us | -- |

In the "changes every step" stream the planner went **42.7 -> 18.6 us, cut by 56%**, a bit more than the half the
bisect predicted (the 95 pointer-patch calls are gone, along with the divergence they caused). Correctness is still
bit-exact.

## Each new mechanism still owes a timing measurement (2026-09-19, late night, still no exclusive GPU)

Every mechanism added today for coverage carries a latency cost, and none has been measured on an exclusive GPU
yet. Listing the costs here first so they don't get forgotten:

| Mechanism | When paid | Estimate |
|---|---|---|
| child route: `torch.ops.*` interception, extern allocates its own outputs | one harvest per new shape (a few small captures); on shape change, `ExecChildGraphNodeSetParams` x number of call sites + one ctx `copy_` | harvest ~ tens of ms; shape change ~ a few us/call site |
| SWITCH | one re-capture per new topology (equivalent to one recording); ~9 us per SWITCH node per replay (`cond_cost.cu`); after a re-capture, pointers are rewritten on every shape change (planner +~27 us) | paid only by regions whose topology changes |
| Warm-up launch | one extra replay after every capture that contains a child | one replay's time, once per topology |
| Rebuild (REBUILD) | when the first shape is too small, rebuild on a larger shape: one build (incl. planner compile, cached) + one recording | at most 3 times |

Three things to measure: (1) `bench.py --extern-child` steady-state replay time vs the cut route and per-shape
recording; (2) how much more per replay a region with SWITCH (the model in `probe_extern_child`) costs than one
without; (3) for a conv region, one graph vs cuDNN per-batch recording, first pass. Waiting for a GPU.

## SWITCH cost per replay: +9.2 us per node (2026-09-19, exclusive GPU 7)

`dynagraph/probes/bench_switch_cost.py`: the same model (4-layer GEMM, cuBLAS, 8 addmm call sites) compiled twice.
A has seen only 512/256 (8 plain child nodes); B has seen 64 (M=64 splitK -> the 8 call sites are re-captured as
SWITCH, 2 bodies each, and after the re-capture pointers are rewritten on every shape change). The two are timed
interleaved, CUDA event median:

| Stream | A plain child | B SWITCH | Diff |
|---|---|---|---|
| Same shape repeated, 512 (planner early-out) | 166.9 us | 240.6 us | **+73.7 us = 9.2 us/SWITCH** |
| Alternating 512/256 (planner runs every time, B also writes pointers) | 227.7 us | 302.3 us | +74.6 us = 9.3 us/SWITCH |

This matches the ~9 us each from `cond_cost.cu` exactly, and the pointer-rewrite half of the planner does not show
up in the alternating stream (both rows have the same diff). **8 SWITCHes add 44% to a 167 us replay** -- SWITCH
must be given only to call sites whose topology really changes, and the topology of these 8 call sites changes
together (all are the M=64 splitK), so it should be **one** SWITCH: a region-level SWITCH where each body holds the
whole chain (Triton nodes go into the body too; `switch_patch.cu` proves nodes inside a body can still be patched),
paying the 9 us only once. That is the next cut.

Also: the child route's harvest blew the bench model (12 blocks, 56 shapes) up to a **79 GB OOM** -- every
harvested small graph had its own private mempool, and each pool reserves a whole segment. Changed to one pool
shared by the whole region (`harvest_pool`); rerun below.

## The child route's bill: coverage bought, first pass 3.5x, steady state 2.8x (2026-09-19, exclusive GPU 0)

`bench.py --regime launch --gemm-backends ATEN,TRITON --extern-child` (12 blocks, d=128, GEMM via
cuBLAS, 24 addmm call sites, 63 distinct shapes; with the shared harvest pool peak memory is 5.4 GB, while the
earlier private-pool version went straight to a 79 GB OOM):

| | First pass (incl. re-recording) | First-pass recordings | Steady state (min of 5 interleaved rounds) | Peak memory |
|---|---|---|---|---|
| A re-record per shape | 519.3 ms | 56 graphs | **37.5 ms** | 0.12 GiB |
| C pad2max | 43.7 ms | 0 | 42.8 ms | 0.21 GiB |
| D DynaGraph child route (max first) | **1815.7 ms** | 0 graphs | 106.2 ms | 1.22 GiB |
| D DynaGraph child route (natural order) | 4735.0 ms (incl. one rebuild) | 0 | 103.7 ms | 2.98 GiB |

Numerics vs A: 8.8e-5 (cuBLAS picks its algorithm independently in the two captures). Steady state of the three
routes on the same model: cut 621.7 ms (15x) -> child 106 ms (2.8x) -> per-shape re-recording 37.5 ms; whereas
with GEMM on Triton, DynaGraph is only 9.8% slower than re-recording.

Where the cost is: each new shape costs one harvest = one warmup for each of the 24 call sites + one small capture
+ one dry run of the wrapper, much more expensive than upstream recording one graph (one capture), hence the first
pass at 0.29x; in steady state every step that changes shape needs 24
`cudaGraphExecChildGraphNodeSetParams` calls (~50 us each, host side, cannot be hidden on the GPU), which add up
to those 70 ms. **Conclusion**: the child route is meant for "a few truly opaque call sites" (SDPA, conv, NCCL);
24 cuBLAS GEMMs should not go through it -- GEMM on Triton (or an open-source implementation such as DeepGEMM) +
the few remaining externs on the child route is the combination that holds up on the latency axis. To cut the cost
of the child route itself: drop the warmup from harvest (cuBLAS's workspace can be created ahead of time), and
batch the child swaps (one `cudaGraphExecUpdate` swaps the whole graph).

## conv region: one graph vs cuDNN per-batch recording (2026-09-19, exclusive GPU 7)

`dynagraph/probes/bench_conv_firstpass.py`: Conv2d 8->16->8 + relu + mean, 12 batches
[16,1,2,3,4,8,32,64,96,128,5,48], 3 passes each, CUDA event per call, compilation outside the timing.

| | First-pass wall clock | Recordings | Pass-3 wall clock | Pass-3 median device time per step |
|---|---|---|---|---|
| cudagraph_trees, per-batch recording | 522.5 ms | 12 | 30.5 ms | 53.8 us |
| DynaGraph (child + SWITCH + one rebuild) | **2657.2 ms** | 6 (the third topology bucket recorded per shape) | **11.4 ms** | 273.8 us |

First pass is 5x slower (12 harvests + one re-capture + one rebuild, every harvest including cuDNN's warmup); the
steady-state wall clock is instead 2.7x faster (upstream has to find the graph in the tree and check inputs every
step; here it is a single replay), but **device time is 5x** (two SWITCHes at 9 us each, the planner, and cuDNN's
4-node variant inside the child is itself slower than the 1-node one). Both sides of the bill are here; the conv
route is still far from a "net win": the first pass depends on removing the warmup from harvest, the steady state
on a region-level SWITCH.

Caught a crash along the way: **after a rebuild the old runner is destroyed and the new runner's replay hits an
illegal address** (always reproduces with conv + SWITCH bodies; it never showed up in the probes because the spy
list holds on to the old runner). The cause is not understood; for now the retired runners are kept (at most
`dynagraph_rebuilds` of them), and all four input/output usage patterns pass all 3 passes.

## Re-measured after removing the workarounds (2026-09-19, exclusive GPU 0, after `cudaGraphUpload` landed)

The SWITCH cost per replay is unchanged: +73.8 us / 8 = **9.2 us each** (same shape repeated 132.6 -> 206.4 us;
alternating 182.0 -> 255.9 us). After a re-capture, pointers are no longer rewritten on every shape change (B's
"pointers always dirty" = False), but that half of the planner was already hidden inside the identical diffs of the
two rows and cannot be measured -- the 9 us of the SWITCH node itself is the bulk, and a single region-level SWITCH
is still the next cut.

conv, 12 batches (`bench_conv_firstpass.py`; the same graph now serves all 12, **0 recordings**):

| | First-pass wall clock | Recordings | Pass-3 wall clock | Pass-3 median device time per step |
|---|---|---|---|---|
| cudagraph_trees, per-batch recording | 457.9 ms | 12 | 2.2 ms | 150.5 us |
| DynaGraph | 2630.1 ms | **0** | 2.9 ms | 218.4 us |

First pass is still 5.7x slower (12 harvests each with cuDNN warmup, 2 re-captures, 1 rebuild); steady state is
1.3x (device time 1.45x: two SWITCHes 18 us + planner + the cuDNN variant). In the previous version (third bucket
recorded per shape), 274 vs 54 us, upstream's 54 was a fluke of that run; this time the same upstream is 150. The
first pass depends on removing the warmup from harvest.

The workaround from "Caught a crash along the way" above, keeping retired runners (`dynagraph_retired`), has also
been removed: the same 12 batches run twice in a row without crashing (0 recordings, pass 3 at 2.2/2.9 ms and
2.2/3.0 ms). Root cause is the same as in `docs/notes/EXTERN.md` section 8.2.

## Where the first-pass money goes: not harvest, but nvcc (2026-09-19, exclusive GPU 6/7)

conv with 12 batches, first pass 2630 ms, previously written down as "12 harvests each with cuDNN warmup". A look
with cProfile (`dynagraph/_harvest_prof.py`, `_harvest_prof2.py`): of the 2.64 s, **1.9 s is two
`select.poll` calls -- waiting on the nvcc subprocess**, one of them for the rebuild (`REBUILD`); the 11 harvests
total 60 ms (76 `torch.convolution` calls), and instantiate, swap and sync are each in the tens of ms. The planner
source hard-coded the build-time slot offsets (constants in the layout kernel), so rebuilding for a different shape
produces different source and misses the in-process cache.

Three things changed together:

| | Before | Now |
|---|---|---|
| Layout kernel | first node in the graph, writes constants into `slot_off` | gone; `slot_off` is written once by the host at build time, and the graph has one fewer node |
| Planner source | contains this build's offset constants | depends only on the wrapper: rebuilds of the same region / new processes hit the cache |
| Compiler | `nvcc -cubin` subprocess | `cuda.bindings.nvrtc` in-process, falling back to nvcc on failure; the cubin is stored in the Inductor cache directory keyed by sha256(source+arch) |

Compile time for the same planner source (205 lines, `_dump_planner.cu`) (`_time_compile.py`, 3 runs each):

| | One compile |
|---|---|
| nvcc subprocess | 906-1104 ms |
| nvrtc in-process | 34 ms (18-22 ms for another compile in the same process) |

nvrtc needs two adaptations: it has no libstdc++ `<cstdint>`, and it does not ship declarations of the device-side
graph update API, so the template has an `#ifdef __CUDACC_RTC__` branch that typedefs int64_t and explicitly
does `#include <cuda_device_runtime_api.h>`; the resulting cubin is the same size as nvcc's (16448 bytes) and loads
directly with `cuModuleLoadData`.

conv with 12 batches, re-measured (`bench_conv_firstpass.py`, exclusive GPU 6, `force_disable_caches`, so no disk cache):

| | First-pass wall clock | Recordings | Pass-3 wall clock | Pass-3 median device time per step |
|---|---|---|---|---|
| cudagraph_trees, per-batch recording | 485.7 ms | 12 | 2.3 ms | 158.0 us |
| DynaGraph (before) | 2630.1 ms | 0 | 2.9 ms | 218.4 us |
| DynaGraph (now) | **440.8 ms** | **0** | 2.9 ms | 213.5 us |

First pass went from 5.7x slower -> 9% faster than upstream; steady state did not move (the two SWITCH nodes at
9 us each are still there; see the next cut). The build itself (the one compiled outside the timing) also dropped
from ~3.7 s to ~2.8 s, and a fresh process saves even the 34 ms.

## After moving topology selection back to the host (2026-09-19, exclusive GPU 6 / 7)

Numbers and tables are in `docs/notes/EXTERN.md` section 10.1. In one sentence: the 74-76 us per replay of the
8-call-site SWITCH is gone; switching between two graphs by shape costs the same as repeating a single shape
(back-to-back 88.6 vs 92.9 us/call), 50 us cheaper than changing shape within the same graph (planner +
8 child swaps, 139.9); conv 12-batch steady state went from 55-63 us more than upstream to 28-44 us more.
Alternately launching two execs costs nothing at the driver level (`_alt_exec_cost.py`, `_alt_exec_cost2.py`: 0.0 us).

## The child route's bill, recomputed (2026-09-19, exclusive GPU 7, host load 113)

Same bench (`bench.py --regime launch --gemm-backends ATEN,TRITON --extern-child`, 12 blocks, 24
addmm call sites, 63 shapes, 256 steps), after host-side graph selection + fixed slots + nvrtc:

| | First pass | Recordings | Steady state (min / median of 5 interleaved rounds) | Peak memory | Numerics vs A |
|---|---|---|---|---|---|
| A re-record per shape | 385.1 ms | 56 graphs | **39.9 / 40.0 ms** | 0.12 GiB | - |
| C pad2max | 65.9 ms | 0 | 65.3 / 65.3 | 0.21 GiB | - |
| D DynaGraph child (max first) | 2069.5 ms | 0 | **61.9 / 63.7 ms** | 1.22 GiB | **0.0e+00** |
| D DynaGraph child (natural order, incl. one rebuild) | 2375.0 ms | 0 | 63.0 / 73.2 | 2.99 GiB | - |

The previous version was 106 ms (2.8x), now it is 62 ms (1.55x); each step costs ~86 us more than re-recording. The
previous version's attribution of the 70 ms to "24 child swaps at 50 us each" was wrong: in today's 8-site profile a
child swap is 2.4 us each, 24 of them = 58 us, plus the planner's 7 us and the input copies and Python, which adds
up exactly to the 86 us. So the remaining bill is three items: child swaps (only on shape change), the planner, and
input copies, each in the tens of us per step, with no single dominant item. The 2.07 s first pass is one harvest
for each of the 63 shapes (24 small captures each); if the first pass doesn't matter, leave it alone. The 1.22 GiB
of memory is the retained harvested graphs (63 x 24 small graphs).

## The bill for the two routes, launch-bound (2026-09-19, exclusive GPU 7, host load 296 / 256 cores)

Same bench, `--share-autotune` (all six configurations share the first configuration's autotune choices, so the GEMM kernels are
identical), `--gemm-backends ATEN,TRITON --extern-child`, 12 blocks, 24 addmm sites, 63 shapes, 256 steps:

| | First pass | Recordings | Steady state (min / median of 5 interleaved rounds) | Per step | Peak memory | Numerics vs A |
|---|---|---|---|---|---|---|
| A re-record per shape | 590.3 ms | 56 graphs | 125.9 / 126.2 ms | 493 us | 0.09 GiB | - |
| B eager | 433.2 | 0 | 405.9 / 440.4 | 1720 us | 0.10 | - |
| C pad2max | 130.9 | 0 | 128.9 / 129.1 | 504 us | 0.18 | - |
| D device route (max first) | 102.5 | 0 | 74.8 / 76.2 | 298 us | 0.19 | 2.0e-07 |
| **D host route (max first)** | 96.3 | 0 | **71.3 / 71.5** | **279 us** | 0.13 | 2.0e-07 |
| D device route (natural order, incl. one rebuild) | 235.8 | 0 | 76.8 / 77.0 | 301 us | 0.14 | - |

First, the quality of these numbers: host load is 296, and configuration C, a single static graph doing `replay()` every
step, also needs 504 us/step -- every configuration on this machine is now host-bound, and the GPU is waiting on Python.
So this table measures **the host-side work per step**, not GPU time:

- The host route is 19 us per step cheaper than the device route. `microbench/host_update.cu` predicted that when
  launch-bound the device route would be 30 us faster (host graph update 36 us vs one launch 3 us); in practice it
  is the other way round, because the host route also saved two pieces of Python along the way: the 57 us of the
  `replay()` wrapper and the 25 us of input `copy_` (section 13.1, pointers are patched away). The 36 us of graph update
  itself was indeed paid, but these two items outweigh it. The device route still goes through `replay()` and aten
  copies; those two items are its next cut.
- Both DynaGraph routes are 40% faster than A: every A step goes through cudagraph_trees' path checks over 56
  graphs + `replay()` + input copy + output reconstruction, while DynaGraph is one graph and one C++ call. In the
  previous round (load 113), A was 40 ms and D was 62 ms, when GPU time still showed through; now the whole
  machine's load has more than doubled, and the host-side bill covers up the GPU-side one.
  **This is not DynaGraph getting faster; it is A having more Python and suffering more under heavy load**. The
  GPU-side differences (planner 7-17 us, child swaps) are invisible in these numbers; they need a re-measurement
  once the load drops, or a look at the gpu-bound regime.
- First passes are all on the order of 100 ms (nvrtc + .so cache hits); A's 590 ms is 56 recordings.
- All six configurations match A to within 2e-7 (fp32 GEMM rounding differences, with identical kernel choices).

The gpu-bound regime produced no numbers this round (the second part of the script exits silently; investigating).

## The bill for the two routes, gpu-bound (2026-09-19, GPU 4 shared with someone else's diffusion ablation, host load 184)

The gpu regime of the same bench (3 blocks, d_model=1024, 6 cuBLAS mm sites, `--share-autotune --extern-child`,
128 steps, 58 shapes, 3 rounds). This regime had never run end to end on the child route before (the cluster problem in section 13.3); these are the first numbers:

| | First pass | Recordings | Steady state (min / median) | Per step | Peak memory | Numerics vs A |
|---|---|---|---|---|---|---|
| A re-record per shape | 930 ms | 36 graphs | 44.9 / 44.9 ms | 351 us | 0.28 GiB | - |
| B eager | 78 | 0 | 91.7 / 107.3 | 716 | 0.32 | - |
| C pad2max | 72 | 0 | 69.3 / 71.3 | 541 | 0.67 | - |
| D device route (max first) | 8164 | 0 | 44.4 / 44.5 | 347 | 1.54 | 2.2e-05 |
| D host route (max first) | 4618 | 0 | **43.9 / 44.0** | 343 | 2.64 | 2.2e-05 |
| D device route (natural order, one rebuild) | 7326 | 0 | 44.9 / 45.0 | 351 | 3.98 | - |

- In steady state the three are tied (DynaGraph is 1.01x of A): the ~350 us of GPU work per step completely covers the host side, and both the host route's SetParams
  and the device route's planner are hidden inside it. The host route is faster than the device route by 0.5 ms / 128 steps = 4 us/step, which is exactly the planner cost.
- First pass 4.6-8.2 s vs A 0.93 s: harvest of 58 shapes x 6 sites (each one cuBLAS warmup + capture) + 6 re-captures
  (6 cluster variants, one graph each). The first pass is not a concern here, but this is the fixed cost of the child route when there are many GEMMs and many shapes.
- Peak memory 1.5-4 GiB vs 0.28: the retained harvest graphs (58 x 6 small graphs) + 6 execs.
- Numerics 2.2e-5: for each shape A records the kernel cuBLAS chose at that shape, and DynaGraph's child is also harvested at that shape;
  the difference comes from cuBLAS's workspace/algorithm choice when each of the 6 main graphs was captured, which gives a different TF32 accumulation order. It is not a bug.
- The 6 cluster variants are what cuBLAS sm90 fp32 GEMM picks by M: (1,2,1) / (4,1,1) / (5,1,1) / (4,2,1) / (6,1,1) /
  splitK with two nodes. Models with more variants will hit `_MAX_BODIES = 8`; those shapes are handed back upstream via `extern-topology`.

## Re-measured on an exclusive GPU 2 after all of tonight's changes (early morning 2026-09-20, GPU idle, host load 360-414)

Same bench (`--share-autotune --extern-child --gemm-backends ATEN,TRITON`, 256 steps, 63 shapes, 5 interleaved rounds):

launch-bound (12 blocks, 24 addmm sites):

| | First pass | Recordings | Steady state min / median | Per step | Variability | Numerics vs A |
|---|---|---|---|---|---|---|
| A re-record | 916.9 ms | 56 | 124.1 / 124.3 ms | 485 us | x1.5 | - |
| C pad2max | 127.7 | 0 | 127.4 / 127.8 | 499 | x1.0 | - |
| D device route | 118.9 | 0 | 71.8 / 72.1 | 282 | x1.0 | 1.6e-07 |
| **D host route** | **77.9** | 0 | **68.7 / 69.4** | **271** | x1.0 | 1.6e-07 |
| D device route natural order (one rebuild) | 497.0 | 0 | 73.4 / 73.4 | 287 | x1.0 | - |

gpu-bound (3 blocks, d_model 1024, 6 mm sites):

| | First pass | Recordings | Steady state min / median | Per step (min) | Variability | Numerics vs A |
|---|---|---|---|---|---|---|
| A re-record | 688.3 ms | 56 | 56.0 / 70.7 ms | 219 us | x2.6 | - |
| C pad2max | 39.9 | 0 | 49.0 / 49.9 | 191 | x1.1 | - |
| D device route | 8238 | 0 | 55.2 / 64.2 | 216 | x1.3 | 2.2e-05 |
| **D host route** | **1212** | 0 | **44.6 / 54.6** | **174** | x1.5 | 2.2e-05 |
| D device route natural order | 5555 | 0 | 60.4 / 62.0 | 236 | x1.1 | - |

- launch regime: in steady state DynaGraph is 44% faster than re-recording, and the host route is 11 us faster per step than the device route; C, a single static graph at 499 us/step, shows the machine
  is still host-bound, so what is being compared is the per-step host work. First pass: host route 78 ms vs re-recording 917 ms (11.8x).
- The gpu regime has heavy interference (A itself varies x2.6). By minimum: the host route is 20% faster than re-recording, the device route ties, and pad2max is actually the fastest (one graph,
  no per-step host overhead). First pass: harvest of 63 shapes x 6 sites + 6 re-captures, host route 1.2 s, device route 8.2 s -- the device route has ~1 s of extra fixed overhead per re-capture,
  not profiled. Memory: device route 1.66 GiB vs 0.45.
- The host route is faster in both regimes: in the launch regime it wins by running less Python per step; in the gpu regime it wins because the graph patching hides inside the previous step's GPU time, while the device route's planner is paid in full.

## The bill for the two layouts, launch-bound (early morning 2026-09-20, exclusive GPU 5, host load 190-260)

Same bench (`--regime launch --share-autotune --extern-child --gemm-backends ATEN,TRITON`, 12 blocks, 24
addmm sites, 256 steps, 63 shapes, 5 interleaved rounds), changing only `TORCHINDUCTOR_DYNAGRAPH_LAYOUT`:

| | Layout | First pass | Steady state min / median | Per step | Tag |
|---|---|---|---|---|---|
| A re-record | - | 634 / 679 ms | 124.0 / 125.2 | 485 us | - |
| D device route (max first) | dynamic | 87.9 | **77.6** / 77.6 | 303 | - |
| D device route (max first) | fixed | 108.1 | **75.3** / 75.4 | 294 | - |
| D host route (max first) | dynamic | 76.6 | **69.7** / 69.7 | 272 | - |
| D host route (max first) | fixed | 89.9 | **70.6** / 70.6 | 276 | - |
| D device route (natural order) | dynamic | 107.8 | 79.4 / 79.5 | 310 | - (grow) |
| D device route (natural order) | fixed | 330.4 | 77.2 / 77.2 | 302 | arena-too-small (one rebuild) |

- Host route: the two layouts tie (69.7 vs 70.6, within noise): when the shape changes, the nodes need SetParams anyway, and the pointers are written into the same struct.
- Device route: the dynamic layout costs 9 us more per step (77.6 vs 75.3 / 256 steps): that is exactly the runtime calls for the pointers that moved; under launch-bound the
  GPU has idle time, which hides part of it (the pure planner difference is 24 us).
- Natural order: the dynamic layout does not rebuild, first pass 108 ms vs 330 ms for the fixed layout (one REBUILD = 220 ms); steady state 79.4 vs 77.2,
  the difference being the re-harvest after growing, amortized into it.
- The bench's "peak memory" row is the `max_memory_reserved` over the whole row's run, which includes compile and autotune allocations; both layouts show 0.07 GiB,
  so the arena cannot be told apart; the arena comparison itself is in `probe_packed_layout.py` (0.30 vs 1.43 MB).
- Numerics: both layouts and both routes match A bit for bit.
