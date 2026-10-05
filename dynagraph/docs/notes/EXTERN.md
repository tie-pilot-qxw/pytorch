# extern kernels: a tiered design after measurement (2026-09-19)

The starting point is the `extern-launch` fallback label. Inductor does not enable `max_autotune_gemm`
by default, so GEMMs go straight to `extern_kernels.addmm`, and as a result almost every real model
has the whole region rejected right away. Four measurement tracks
(cuBLAS / cuDNN / PyTorch's own fallback kernels / what Inductor actually sends into extern, each paired
with an independent verification agent trying to refute it) changed our original framework twice. The scripts are in
`dynagraph/verification/_wf_*`, the microbenchmarks in `microbench/`.

## 1. What the measurements overturned

### 1. The criterion is not "transparency" but "does the node count change"

| | Nodes per call | Takeaway |
|---|---|---|
| **cuDNN SDPA** | 10 sequence lengths (128-4096) give **only 1 topology and the same kernel** | Easiest. Parameters have a closed form (derived from 3 captures); the kernel is persistent, and with the grid changed to (1,1,1) the result is still bit-identical -- the grid is not a shape axis at all |
| **cuBLAS GEMM** | Mostly a constant 1; fp32 addmm gives 2 for M in [3841,3872] (memset + segment_k) | In between. bf16 M=1..4096 swept **point by point** yields 94 kernels and 3 clusters, **interleaved** rather than in segments (the kernel changes 357 times between adjacent M) |
| **cuDNN conv** | **14 batch sizes give 5 topologies**; the MEMSET node is non-monotonic (present at b=10, absent at b=11, present again at b=18/19) | Hardest. Once the node count changes, no per-node patch can express it |
| **PyTorch's own ATen kernels** | 42 ops x 7 shapes: **38/42 patchable** (36 do not change kernel, 2 can be rescued by swapping func) | The remaining 4 are all stuck on node count: topk 19999->20000 goes from 2 to 22, sort **4096->4097** from 5 to 14 |

So "cuDNN should be fine, cuBLAS is the nastiest" has to be reversed: **conv is the hardest, SDPA the easiest, cuBLAS in between.**

Two more non-shape variables must be pinned down up front: **pointer alignment** (same numel, but shifting the base
by 1 float turns `vectorized_elementwise_kernel<4>` into `unrolled_elementwise_kernel`, with the argument count going 3->7)
and **dtype**.

### 2. (B) parameter semantics, (C) kernel swapping -- both walls came from picking the wrong API

- `cuGraphExecKernelNodeSetParams` cannot change the cluster dimensions (that is a node attribute); crossing clusters always crashes.
- **`cudaGraphExecUpdate` (full graph) can swap func / grid / smem / cluster all together.** Measured: one
  bf16 M=947 graph updated in turn to M=4/16/113/236/947/1024/2759/4096, across 3 clusters,
  switching back and forth, bit-exact.
  **Cost added on 2026-09-22** (`dynagraph/verification/_wf_execupdate_scale.py`, GPU 7, colocated):
  **fixed 1.23 us + 0.085 us per node** (measured 1.31/1.86/4.53/15.22/86.67 us at 1/8/48/200/1000 nodes).
  Changing the cluster **costs nothing extra**: switching a single-node fp32 addmm graph to (1,8,1) takes 1.65 us, to (2,1,1)
  1.66 us, no different from the 1.61-1.70 us for the same cluster; all four targets give `max|err|=0`.
  (The 0.47-0.53 us/node at `docs/notes/FEASIBILITY.md` (section "2. All measured data") came from an earlier synthetic bench, 6x slower than this one,
  and was never reconciled; this measurement takes precedence.)
  **But it cannot be used as a SWITCH: ExecUpdate overwrites every parameter already patched on the exec back to the template values**
  (`dynagraph/verification/_wf_execupdate_clobber.py`: after SetParams redirects the output pointer to z,
  an ExecUpdate **back to the same template graph** makes the output jump back to y). The arena pointers / symbolic grids / scalars
  that DynaGraph patches per shape are therefore all lost, and every branch switch would require re-patching the whole graph.
  So for cluster changes the SWITCH body is the only route, and the "crossover at 91 nodes" comparison is void.
- **But `cuGraphExecChildGraphNodeSetParams` does not carry launch attributes** (overturned late on 2026-09-19;
  the illegal instruction in gpu-regime was exactly this), so swapping a child graph **cannot change the cluster**.
- WARNING: **in `_wf_swap*.log`, every line after the first failure is invalid.** CUDA errors are sticky:
  after one cross-cluster crash, every subsequent call in the context returns the same 715.
  Re-checked on 2026-09-22: the "crashes even within the same cluster" entry at `_wf_swap2.log:36` is false --
  running fp32 host graph M=512 -> target M=2048 on its own (func swap, cutlass 64x64 -> 128x64,
  smem 98304 -> 147456, same cluster (0,0,0)) gives `max|err|=0`;
  if M=1022 is first made to crash across clusters, the same M=2048 immediately turns into 715.
- **`cuGraphExecKernelNodeSetParams` can change `sharedMemBytes`.** Same run as above: deliberately pinning smem
  at the host graph's 98304 while running a kernel that needs 147456 gives an illegal memory access;
  passing the target value gives bit-exact results. So what it can swap is **func + grid + smem + parameters**; only the cluster is out of reach.
- Going this way **does not require understanding that 1784B parameter block** -- problem (B) disappears entirely.
- The libdivide magic numbers have a closed form: `magic(d) = floor(2^(31+ceil(log2 d))/d)+1`, which matches the measurements exactly.
  So "magic numbers can only be obtained by capture" was wrong.
- **Already-captured nodes can be made device-updatable after the fact**: `CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE`
  is documented in the 13.1 headers as "Valid for **graph nodes**, launches", and `cuGraphKernelNodeSetAttribute` turns it on
  for an already-captured node; measured to work. **Extern nodes can also get handles and enter the existing planner** -- what I said earlier,
  that "cuBLAS/NCCL having no handles is a dead end for the device-side route", was wrong.

### 3. The only remaining wall: topology changes -- and SWITCH is made exactly for that

`microbench/switch_patch.cu` (2026-09-19): nodes inside a SWITCH body **can be patched both ways** --
the device-side planner can, in the same launch, `cudaGraphSetConditional` + change the grid of the selected body;
the host-side `cudaGraphExecKernelNodeSetParams` can also modify body nodes; when both are used, the device side runs later and wins.
`cond_cost.cu` measured this long ago: each SWITCH costs ~9 us, independent of body size.

**It also works from Python (late night 2026-09-19, `dynagraph/probes/probe_switch_py.py`)**: during stream capture,
`GetCaptureInfo` -> `cudaGraphConditionalHandleCreate` on the graph being captured -> `cudaGraphAddNode`
(SWITCH, size=2) -> in each of the two bodies, `cudaGraphAddChildGraphNode` places a small graph captured beforehand ->
`UpdateCaptureDependencies` moves the dependencies onto the SWITCH; the condition is set with `cudaGraphSetConditional`
by a device kernel placed earlier in the main graph that reads ctx; the main graph uses keep_graph + an explicit instantiate.
Select 0 / select 1 / out of range (nothing runs) / `cudaGraphExecChildGraphNodeSetParams` to swap the child in body 1 and then select
-- all six steps bit-exact. Every call pattern the runner needs is in this file.

### 4. Frequency: opaque calls are the majority

| Model | Opaque calls | Triton launches |
|---|---|---|
| 4-layer transformer forward, bf16 | 17 addmm + 4 cuDNN SDPA = **21** | 13 |
| Same, backward | 34 mm + 4 SDPA bwd = **38** | 75 |
| resnet18 forward, fp32 | 20 conv + 1 addmm = **21** | 20 |

Counted by op kind, the closed-source surface is narrow (there are only 9 CUDA-related members of `extern_kernels`, and the common ones are just
mm/addmm/bmm/baddbmm/convolution + SDPA); counted by launches, it is not narrow at all.

Also: the class that truly "changes shape without changing kernel" (TensorIterator-style elementwise / index)
is exactly what Inductor already lowers to Triton by default, so it never shows up as extern at all. **There are no free wins on the extern side.**

### 5. Must be split unconditionally

- `nonzero`: there is a Memcpy DtoH host sync in the middle, and the wrapper reads u0 on the host.
- `_c10d_functional.*`: a third kind of entry point (neither extern_kernels nor aten fallback).
  Whether NCCL switches algo/protocol by message size was **not measured**.

## 2. Tiering: use the mechanism that matches what changes

| What changes | Mechanism | Cost | Status |
|---|---|---|---|
| Parameters / grid | device-side patch by the planner | ~43 us/replay (summed over several hundred calls); **0.9 us when the shape has not changed** | Exists; early-out done |
| Kernel variant (incl. cluster) | child-graph node + lazy per-shape harvest + `cudaGraphExecChildGraphNodeSetParams`, host side | One small capture per new shape (capturing only those few extern kernels); N host calls to swap nodes, 0 when the shape is unchanged | **Implemented** (`dynagraph_extern_child`): with the default config, 3 of 4 shapes are served by one graph, bit-identical; shapes with a different topology are handed back upstream per shape |
| **Topology** | **SWITCH, one body per topology, condition computed by the planner from ctx** | ~9 us each | Measured feasible, not implemented |
| None of the above | **Split**: `dynagraph_partition_extern` makes Inductor partition at the extern | Graph fragmented; extern host overhead retained | **Implemented, 8/8 segments served, bit-identical** |

Three things to think through:

1. **The set of topologies can only be known after the fact** (the boundaries are non-monotonic). Lazy enumeration: record one body per topology seen; when a new topology appears,
   a body cannot be added to an already-instantiated SWITCH, so it has to be re-instantiated -- but that is "once per new topology"
   (conv has 5 in total), not "once per shape".
2. **The 9 us goes only to call sites that actually change topology.** Putting all 21 opaque calls of the transformer forward on SWITCH would be
   190 us, which is not acceptable. SDPA has 1 topology and does not need it; addmm only changes at the split-k boundary; only conv truly needs it.
   The criterion is "measured topology count > 1", added selectively per call site.
3. **Whether host-side patching can be hidden depends on whether the GPU has work to do** (`docs/notes/TIERS.md`).

## 3. Interface

```python
@register_dynagraph_reparam(aten.addmm.default)
def _(call):
    return Variants(
        # Finite and enumerable -- at capture time every variant is recorded as one body of the SWITCH.
        # None = I cannot enumerate them (cuBLAS can honestly only write this) -> automatically falls back to splitting.
        topologies=lambda: [...],
        select=lambda env: ...,            # which topology the current env falls into
        # How to modify within the same topology: grid / param offset->value / or swap the whole func
        reparam=lambda env, topo: Reparam(grid=..., params={...}, func=None),
    )
```

Key points:
- **How the variants are enumerated** goes into the declaration, not just "which one is current". The SWITCH bodies must all be recorded at capture time.
- `topologies` returning `None` is legal and important: **an op that cannot be enumerated cannot enter the interface, and that is exactly the property the interface should have.**
- When one call produces **multiple** nodes, `Reparam` is indexed by node index.
- The granularity is (op, backend), not op: the same aten op has different node structures under different backends.

## 4. In what order to do it

Reverse order of difficulty, which also happens to be the order of payoff for transformers:

1. **SDPA**: 1 topology, closed-form parameters, the grid does not even need to change. Only needs (A) node identification + parameter patching.
2. **GEMM**: the node count is basically stable; `ExecUpdate`/child-graph swaps func+cluster. addmm's split-k
   interval (fp32 [3841,3872]) and mm's 1<->2 use SWITCH or splitting.
3. **conv**: SWITCH per topology, lazy enumeration. There is no open-source alternative (Inductor's CUTLASS backend is GEMM-only,
   Triton conv shows "no gains" on H100), so there is no route around this one.
4. Collectives: first measure whether NCCL switches kernels with size; this is the first step on the multi-GPU axis.
   **Measured (late night 2026-09-19, GPU 4+5, `probe_nccl_variants.py`)**: all_reduce fp32 at six sizes from 256
   elements to 64 MB; every captured graph is 3 nodes `[WaitEvent, Kernel, EventRecord]`,
   the kernel symbol is always `ncclDevKernel_AllReduce_Sum_f32_RING_LL` (newer NCCL has a single entry point and
   dispatches protocol/algorithm in the device-side work batch), and replay is correct throughout. **1 topology, 1 kernel**:
   NCCL does not need SWITCH; the child route (harvest per size, swap the child) is enough, even less hassle than cuBLAS.
   The parameters are in the kernel's args struct (the work descriptor) and are opaque, but swapping the whole small graph as a child does not require understanding them.
   The constraint specific to multi-GPU is that **the collective order must be the same on every rank** -- DynaGraph only changes how things are launched, not
   the order, and one rank falling back to eager locally does not break matching either (NCCL pairs by order and does not care whether a call was issued from a graph).
   Pitfall in the container: `destroy_process_group` / torchrun teardown hangs, so the probe calls `os._exit` directly.
   Not done yet: in the wrapper these calls are written as `torch.ops._c10d_functional.all_reduce_.default(...)`,
   not `extern_kernels.*`, and today the whole region is rejected as `extern-launch`; to hook them into `_run_intercepted`,
   the `torch` in the wrapper's globals has to be replaced (with a proxy object) so the call can be intercepted.

The split fallback already exists, so a failure at any step has somewhere to land.

## 5. Pitfalls hit while getting the child-graph route working from Python (late night 2026-09-19)

`dynagraph/_probe_child_node_py.py`: in the middle of the main capture, use `cudaStreamGetCaptureInfo` to get the graph being captured,
`cudaGraphAddChildGraphNode` to insert a child node, and `cudaStreamUpdateCaptureDependencies(..., Set)`
to make subsequent launches depend on it -- this half worked on the first try, and replay was bit-identical.

**Swapping the child kept failing with `invalid argument`, unrelated to topology** (the small graphs for all four shapes were 1 Kernel node each,
and even swapping back to the same graph was rejected). Cause: `torch.cuda.CUDAGraph()` defaults to `keep_graph=False`,
**so after instantiate the cudaGraph_t is destroyed**, and the `cudaGraphNode_t` obtained during capture dangles;
`cudaGraphExecChildGraphNodeSetParams(exec, node, ...)` uses that handle to find the node in the exec,
and when it cannot find it, the result is invalid argument. **The main graph must use `keep_graph=True` and an explicit `instantiate()`**;
after that, swapping back and forth among the 4 shapes was bit-exact throughout.

`cudaStreamGetCaptureInfo` in cuda.bindings returns a 7-tuple:
`(err, status, id, graph, deps: list, edgeData: list, numDeps)`.

## 6. Results of implementing the child-graph route (late night 2026-09-19)

`triton.dynagraph_extern_child` (env `TORCHINDUCTOR_DYNAGRAPH_EXTERN_CHILD=1`).
`dynagraph/probes/probe_extern_child.py`, Inductor default config, 4-layer GEMM model, 8 addmm call sites:

| | Recordings | Notes |
|---|---|---|
| Baseline | 4 | one graph per shape |
| dynagraph on, child off | 4 | whole region rejected as `extern-launch` |
| **dynagraph on, child on** | **1** | 512/256/333 served by **one graph**, **bit-identical**; 64 handed back upstream |

Why 64 is handed back: cuBLAS fp32 at M<=128 has an extra `splitKreduce` node (the verification agent measured this
long ago); all 8 call sites go from 1 node to 2 nodes, and the child cannot be swapped in. At harvest time the node counts are compared; if they differ,
this key is put into `skip_keys`, `__call__` returns `SKIP_SHAPE`, and `deferred_cudagraphify` takes the upstream recording path
for this one shape, **without retiring the runner**. This is a stopgap until SWITCH is done.

Mechanism highlights:
* Harvest (`_harvest`): replace the wrapper's `empty_strided_cuda` with "hand out arena views in allocation order"
  (layout from `slot_offsets(env)`, consistent with the layout kernel), replace `extern_kernels.<name>` with
  "warm up once, then capture into a small `keep_graph=True` graph", and run the wrapper eagerly once. The Triton
  kernels in between run for nothing, which is harmless.
* Main capture: replace `extern_kernels.<name>` with "do not execute; insert a child node into the graph being captured and move the capture dependencies
  onto it"; allocations still go through the graph's private pool (the pointers of Triton nodes are re-pointed by the planner anyway).
  The main graph must use `keep_graph=True` + an explicit `instantiate()` (the pitfall in section 5).
* Replay: swap the child only when the shape has changed (`child_applied` cache), one
  `cudaGraphExecChildGraphNodeSetParams` per call site.
* The harvest table is cached by key and cleared once it reaches a cap of 1024 (same as `plans`: the shape space is long-tailed).

Not measured yet: its latency (vs the 15x of splitting, vs re-recording). The bench already has `--extern-child`.

## 7. Externs with self-allocated outputs and `torch.ops.*` call sites also join the child route (late night 2026-09-19)

The child route originally only recognized `extern_kernels.<name>(..., out=buf)`: the output is allocated by the wrapper and lives in the
arena. conv is different -- the output of `buf0 = extern_kernels.convolution(...)` is allocated by cuDNN itself,
and the relu after it even does `buf1 = buf0  # reuse` and writes it in place; collectives are not `extern_kernels` either:
Inductor generates `torch.ops._c10d_functional.all_reduce_.default(buf1, ...)` +
`wait_tensor(buf1)`. Three changes:

* **Call-site recognition** is extended to `torch.ops.<ns>.<op>.<overload>(`, with the name recorded as `ops:...`; it also records, for each
  call site, which buffer the result is assigned to (`extern_site_outputs`). Intercepting `torch.ops.*` works by replacing the `torch` in the
  wrapper module's globals (`_TorchProxy`: only the selected op paths are replaced with interceptors; all other
  attributes pass through unchanged), without touching the global `torch.ops`.
* **Extern output pointers**: at harvest time, the tensor returned by each call site is kept (it lives in the private pool of the small graph's capture and
  stays alive together with the small graph); during the main capture, the one for the build shape is handed to the wrapper to carry on with (`assert_tensor_metadata`
  checks the build shape, so it matches); the planner gets an extra patch section that is not governed by the "pointer dirty bit", which points the
  kernel parameters that reference these buffers at `ctx[after the node state + call-site index]`; the addresses are written there by a single
  `copy_` when the host switches shape. Pitfall hit: after the main capture, `child_applied` already equals the build key, and the original
  child-swap early-out skipped this address write as well -- the planner used the initial value 1 as a pointer, and `misaligned address` stuck to the whole
  context. The address write is no longer under that early-out.
* **`wait_tensor` is skipped entirely** (in both harvest and the main capture): it only does stream ordering, and the child node itself is already
  a dependency of its successors; moreover, running it eagerly during harvest waits on an event recorded in the small graph that just finished,
  and `cuStreamWaitEvent` fails outright with `Event is not valid`.

**conv results** (`probe_conv_child.py`, Conv2d 8->16->8 + relu + mean, batch 16/24/16/20/24/12
x 2 passes, GPU 4): baseline recordings 4 -> **0**, two conv call sites, two self-allocated buffers, 4 harvests,
all four batch sizes **bit-identical**. cuDNN did not change topology on these batch sizes; batch sizes where it does go through `extern-topology`
back to upstream, to be hooked up once SWITCH is done.

Scaled up to 12 batch sizes (1,2,3,4,5,8,16,32,48,64,96,128): all 12 **bit-identical**; cuDNN
node counts show three patterns -- the two convs at 1 node each `[1, 1]` (N<=16), `[1, 4]` (N=32/48/64), `[4, 4]`
(N>=96) -- i.e. conv topology falls into three buckets by batch size, which is exactly what SWITCH is meant to hold: at most 2
bodies per call site. N=64 also hit `input-too-large` once (built at 16 with 2x headroom) and went through a rebuild.

Still open: in the child small graph, NCCL's WaitEvent/EventRecord nodes reference the events of that particular ProcessGroupNCCL
work; whether the events are still valid after the work is reclaimed has not been verified (the variant probe replays right after capture, so it cannot tell).

## 8. SWITCH implemented: topology changes are no longer handed back upstream (late night 2026-09-19)

Lazy enumeration, per call site: at build time each call site has seen only one topology, and all of them are plain child nodes; when harvesting
some shape yields a new node count at some call site, that small graph is recorded as a new body of that call site, the **main graph is re-captured once**
(`_recapture`), and that call site becomes a SWITCH (a `cudaGraphAddNode` conditional node, size = number of topologies seen,
one child node in each body). Number of re-captures = number of topologies - 1, not number of shapes. ctx gets two extra slots per call site:
which body this shape selects (written by a single `copy_` when the host switches shape), and the conditional handle (planner thread 0 uses it for
`cudaGraphSetConditional`; single-topology call sites have handle 0 and are skipped). The same-shape early-out works as before: when the planner does not run,
the condition value stays at the previous one. Swapping kernels/parameters within the same topology is still done by `ExecChildGraphNodeSetParams` swapping the
child inside the body; each body records "which graph is currently loaded", and only differing ones are swapped.

**Result**: `probe_extern_child` (cuBLAS fp32 addmm, M=64 goes from 1 node to 2 nodes, 8 call sites):
baseline 4 recordings -> **0**, all four shapes **bit-identical**, skip 0, 1 re-capture. Previously this M=64 used the
stopgap of "hand back upstream and record per shape"; that is now gone.

**Three pitfalls hit** (all caught by per-call comparison scripts in the style of `dynagraph/_switch_dbg*.py`;
the probe's own criteria cannot see a retirement -- after retiring, the remaining shapes are all first-time warmups and the recording count is still 0 --
so `probe_extern_child` added "criterion 0: no mismatch-type labels"):

1. **After a re-capture, device-side pointer patches do not persist.** The persistence proven by `probe_update_persistence` only half holds
   in a graph with a SWITCH: on same-shape repeats (planner early-out) everything is intact; but on the call where the shape changes, the planner only modifies
   grid/scalars and does not rewrite pointers, so the kernel nodes revert to the pointers from capture time (20 bit-level comparisons,
   modes A/B/C of `_switch_dbg2.py`). No explanation was found in the documentation. Countermeasure: after a re-capture the pointer dirty bit is never cleared,
   so the pointers are rewritten on every shape change, and the early-out is unaffected; only regions with topology-changing call sites pay this half of the planner.
2. **A re-capture must run the wrapper on the current shape's inputs**: for externs with self-allocated outputs (conv), what is handed back to the wrapper is
   the one harvested for the current shape, and the wrapper's `assert_tensor_metadata` checks against the shape it is running at.
3. **The re-capture launch itself computes wrong results, and every launch after it is correct** (`_region_dbg5.py`: a region with all_reduce re-captured at
   M=64; that output was off by 8.1, while the immediately following 512/256/333 and the second 64 were all bit-exact). The first two suspects
   were both ruled out by micro-probes (`_switch_first_launch.py`: a new exec selecting a body within the same launch on its very first launch,
   or the host swapping an in-body / plain child before the launch -- all four combinations correct). Along the way, capture was changed so that each call site directly loads
   the current shape's own small graph (one fewer host swap), but that did not fix it. Localized to: that launch is wrong from the very first node
   (slot 0 off by 2.3), and **with the host changing nothing, launching again from the same state is bit-exact**.
   A model with 8 SWITCHes and no NCCL child does not show this; it only appears with an NCCL child carrying external event nodes plus
   a SWITCH. The countermeasure is one extra warm-up launch on the re-capture call (once per new topology, negligible);
   the cause is **not understood** and is recorded here.

conv's three topology buckets (`[1,1]`/`[1,4]`/`[4,4]`) and the multi-GPU region (mm becoming 2 nodes at M=64) go through the same machinery;
the results are in later subsections.

## 9. Multi-GPU: a region containing all_reduce is served by one graph (2026-09-19, late night)

`probe_nccl_region.py` (2 ranks, GPU 4+5; Linear -> relu -> functional all_reduce -> Linear ->
sum, shape stream 512/256/333/64 x 2 passes): on both ranks the **region is served, recordings 4 -> 0, no fallback tags**;
the runner's bitwise check against the eager wrapper reported no mismatch; the difference from the control group (cuBLAS
captured by cudagraph_trees itself) is 1.9e-6, which is the magnitude of cuBLAS picking its own algorithm in each of the two
captures, so the probe was changed to compare with a tolerance.

Chain: `torch.ops._c10d_functional.all_reduce_.default` is intercepted as an `ops:` call site and harvested into a
small graph of `[WaitEvent, Kernel, EventRecord]` (`_nccl_clone.py` proves that after it is cloned into a child node,
replay still reduces correctly); `wait_tensor` is skipped entirely; at M=64 the mm's splitK made this region also go
through one SWITCH re-capture. The two ranks each decide harvest/re-capture independently, but the **order** of the
collectives does not change, so they cannot mismatch.

Two pitfalls: (1) `wait_tensor` cannot run eagerly (it waits on an event inside an already-finished capture, and
`cuStreamWaitEvent` errors out directly), so it can only be skipped; (2) the **first launch of a new exec computes the
wrong result, the second is correct** -- with a single GPU and 1 rank it only shows up after a re-capture, with 2 ranks it
shows up on the first launch after build, so now every capture that contains child nodes is followed by one extra warm-up
launch (`warm_launch`). The cause is not understood; recording it here.

### 8.1 The second re-capture crashes: add a guard first (2026-09-19, late night)

`dynagraph/_conv_seq.py` reproduces it with the shortest batch sequence (`CUDA_LAUNCH_BLOCKING=1`):

| Sequence | Result |
|---|---|
| [96, 64] (one re-capture) | all correct, 0 recordings |
| [64, 96, 5] (two re-captures) | first launch after the second re-capture: `illegal memory access` |
| [64, 5, 96] (order reversed) | same as above |
| [64, 96, 5, 64] | same as above |

Unrelated to rebuild (`REBUILD`), unrelated to which topology comes first: **the new exec from the same runner's second
re-capture hits an illegal address as soon as it is launched**. The `stash_on_self` illegal access in the badly-written-code
scan and the crash of conv with 12 batches are both this. Cause not found (suspects: the small graphs cloned into the SWITCH
body when the old graph is destroyed / reuse of conditional handle values; neither verified). Guard: `_MAX_RECAPTURES = 1`
-- each runner re-captures only once; any new topology that appears after that is handed back to upstream to record per
shape (`extern-topology`, the transitional behavior from before SWITCH), no crash. conv's three topology buckets thus become
"two buckets served by one graph, the third recorded per shape".

Another one that is unresolved but caught by the check: `tril_mask` (`_tril_dbg.py`: the rebuilt runner re-captures at 64,
the mm's 2-node splitK graph becomes body 1, after which every shape that selects body 1 (64/200/128) produces entirely wrong
output (off by 80~120), while 333, which selects body 0, is bitwise correct; same with a shared or a private harvest pool;
the same 2-node mm graph used as a plain child in the first runner is correct). The check catches it and the runner retires,
results are still correct, but the coverage is lost. Together with the issue above, everything points to the semantics of
"an exec with conditional nodes: swapping a child on the host / device-side update / destroy and rebuild" not yet being
understood; a dedicated set of micro-probes is needed.

### 8.2 Understood: the upload at the first launch overwrites the device-side update (2026-09-19, exclusive GPU 0)

`dynagraph/probes/probe_cond_semantics.py`, bypassing Inductor: a planner kernel modifies the pointer and a scalar of a
device-updatable worker node; optionally the graph also contains a SWITCH / a child node:

| | No SWITCH | With SWITCH | With SWITCH + `cudaGraphUpload` first |
|---|---|---|---|
| L1 change pointer + scalar | OK | FAIL (neither took effect) | OK |
| L2 change scalar only | OK, pointer still there | FAIL: scalar took effect, pointer never did | OK |
| L3 planner does nothing | OK, everything retained | - | OK |
| child with an external event node, first launch | - | FAIL | OK (2nd launch without upload also OK) |
| A->B->C three captures (old graph destroyed / kept) | - | first launch FAIL in every case | all OK |

In one sentence: **an exec with conditional nodes or external event nodes is uploaded only at its first launch, and that
upload overwrites the device-side updates the planner made during the same launch**; later launches are normal and the
updates persist. Explicitly calling `cudaGraphUpload` after instantiate and before the first launch avoids the problem entirely.

So the three earlier "workarounds" were all shadows of this: (1) "pointers do not persist" -- the pointers were only sent in
the launch that got overwritten; (2) "first launch wrong, second correct" -- the second launch has no upload; (3) "the second
re-capture crashes" -- the first launch after a re-capture runs entirely with the capture-time parameters, which point at pool
addresses that no longer exist. The runner now calls `cudaGraphUpload` once after instantiate in `_capture`, and all three
workarounds (warm-up launch, pointers always dirty after a re-capture, re-capture limited to one) are removed (the re-capture
limit is now 4, only to guard against an unbounded number of topologies).

Re-run after removing the workarounds (exclusive GPU 0): `probe_extern_child` all pass (pointers written only once, no
warm-up launch); `_conv_seq` [64,96,5,64] and [64,5,96] (two re-captures) **0 recordings, no fallback**; conv with 12 batches
**12 -> 0 recordings** (all three topology buckets served by one graph, including one rebuild); `tril_mask` after rebuild +
re-capture, all 8 calls bitwise correct. The "second re-capture crashes" and "tril body 1 wrong" from 8.1 no longer exist.

The fourth workaround was also removed: in `cudagraph_trees`, "after a rebuild, the old runner is kept alive instead of
destroyed" (`dynagraph_retired`, for the illegal address caught in the conv run in BENCH.md). After removing it,
`bench_conv_firstpass` run twice in a row (exclusive GPU 0) gave 0 recordings both times, pass 3 normal step by step, no
crash. It has the same root as the first three: the first launch of the new runner after a rebuild overwrites the
device-side parameters, so the pointers land in the old runner's pool -- if the old runner is alive, it "happens" to read the
old addresses without an error; once it is destroyed, it is an illegal access. Now after a rebuild the old runner is simply
GC'd and no longer holds GPU memory.

## 10. Next cut: move topology selection back to the host (design, 2026-09-19)

The SWITCH bill (BENCH.md): each SWITCH node costs **+9.2 us** per replay, independent of what is inside the body. conv's
two sites = 18 us / 213 us; a model with all 24 GEMMs going through children has 8 sites = 74 us. A region-level "one SWITCH
manages all sites" could only press this down to 9 us, which still has to be paid.

**Observation**: which body a site selects is a function of the shape -- cuDNN / cuBLAS pick the kernel by (N, M, alignment),
and the same shape always gets the same topology; `key_bodies[key]` is exactly what the host computes at plan time and writes
into ctx so the planner can `SetConditional`. Since the host already knows before the launch, there is no need for the device
to select again.

**Approach**: hold one exec per "combination of the body indices of all sites" (`tuple(bodies)`), and the host picks the exec
to replay by combination. Inside each exec the sites are plain child nodes (one body), and swapping children by shape works as
it does now. The first time a new combination is seen, capture a new graph (the current re-capture is already a full-graph
re-capture, so the cost is the same), but the old graph is **kept** instead of replaced.

| | Device-side SWITCH (current) | Host-side exec selection |
|---|---|---|
| Device cost per replay | 9.2 us x number of sites | 0 |
| Host cost | write body indices (on shape change) | one dict lookup |
| New topology | re-capture one graph (replace) | capture one graph (add) |
| Memory | one main graph + one child clone per site per topology | one main graph per combination (with one clone per site); number of combinations ~ maximum number of topologies, not the product -- the sites are bucketed together with the shape |
| First-launch upload | needed (conditional nodes) | only needed for children with external event nodes like NCCL, as before |
| Multi-GPU | each rank decides independently; consistent if shapes are consistent | same |
| When topology is not a function of shape | correct | wrong (captures a few more graphs, but each one is correct; the host follows the combination it sees) |

State owned by each exec (to be packed into an `_Exec`): graph, handles, ctx (states / dirty bits / EXT pointers),
applied / flag_on / ptr_dirty / ext_applied, child_applied / site_applied_raw / site_body_nodes,
verified. Shared by the region: arena, slot_off, input_store, plans, harvest results (child_graphs / extern_outs /
key_bodies), the planner function. Limit: `_MAX_EXECS = 8` graphs per region; combinations beyond that are handed back to
upstream per shape (`extern-topology`).

The SWITCH code is kept as `config.triton.dynagraph_topology = "switch"` (default `"host"`) -- it is the only correct answer
when topology is not a function of shape, and it is also the baseline that has already been measured. On the four axes:
latency gains 9.2 us x sites, coverage unchanged, multi-GPU unchanged, badly written code unchanged; what is spent is GPU
memory for one main graph per combination. Numbers still to be measured (8-site bench, conv with 12 batches).

## 11. Training regions: not a single one had ever been served, and the cause was parameter names (2026-09-19, exclusive GPU 7 / 0)

`probe_messy_code`'s dropout_train had always been `no-symbol-args`; tracing it down: training forward / backward wrappers
unpack as `primals_4, primals_1, mul_3, gt, s33 = args` -- `_input_symbol_map` only recognized `argN_1 / sN / bufN`, and did
not recognize a single training wrapper. DynaGraph's original motivation was variable-length training, yet as of today every
region it served was an inference region. After changing it to accept arbitrary identifiers (only the symbol `sN` has a fixed
spelling), training regions came in, and then ran into three more things:

| What we hit | What it is | How it is handled |
|---|---|---|
| `extern-launch` | dropout's seed `aten.randint.low_out(..., out=buf1)` -- the wrapper calls it through its own `aten` global, not with the `torch.ops.aten` spelling | `aten.<op>.<ovl>(` is also a site (`ops:aten.*`); during harvest the wrapper's `aten` global is also replaced with a proxy |
| The seed op as a child is wrong | randint captured separately as a small graph and inserted into the main graph produces **the same set of numbers** on every main-graph replay (`probe_rng_child.py`: randint captured in the main graph goes through torch's graph-safe RNG and differs every time; nobody advances the philox offset for the one in the child) | Random ops (`_EAGER_OP_NAMES`: rand/randn/randint/bernoulli/...) do not go into the graph: at harvest, record (fn, args), run it once on the host before each replay, writing into its `out=` arena slot; ones without `out=` are copied into the storage from harvest. One small launch |
| The check does not match | the seed drawn by the reference eager run is not the same as the one drawn on the host before replay; the eager run during harvest also advanced the RNG once | `_rng_kept`: after the reference run and the harvest run, put the CUDA RNG state back, so both sides draw the same seed and are bitwise comparable |
| `no-outputs` | BatchNorm backward returns `buf12 = reinterpret_tensor(buf7, (1, 64), (64, 1), 0)` -- the alias table only recognized `bufA = bufB` | reinterpret is also an alias (owner buf7, whose lifetime is extended accordingly); on return it uses its own geometry (`_buffer_views`: sizes / strides / offset; for a view of a view the offsets add up) |

`dynagraph/probes/probe_train.py` (Linear->Dropout->Linear, Linear->BatchNorm1d->ReLU->Linear, train mode,
forward + backward per step, 4 lengths x 2 passes, dynagraph off / on):

| Model | Regions asked / served | Recordings off->on | Numerics |
|---|---|---|---|
| dropout | 2 / 2 (forward + backward) | 6 -> **0** | random, not compared |
| batchnorm | 2 / 2 | 6 -> **0** | loss, every parameter's .grad, running stats all **0.0e+00** |

backward's static inputs include intermediates saved by forward (`mul_3`, `gt`, `relu`); upstream treats them as
static because they are outputs of the forward graph and have stable addresses. When forward is served by DynaGraph they sit
in fixed slots and their addresses do not change across shapes, so backward's static check still passes; if forward does a
REBUILD (the arena is replaced), backward will retire with `static-input-moved` and be handed back to upstream to record --
that is the next thing to handle.

## 12. Non-contiguous inputs (2026-09-19, exclusive GPU 7)

`input-not-contiguous` used to be rejected outright. Kernels are specialized on the input's geometry (a transpose is a
transpose; guards ensure the same region only ever sees the same stride pattern, with values that vary with the symbols), so
when copying into the store it is enough to **keep the input's own strides**: the store is sized by extent
(1 + sum((size-1)*stride)) rather than numel, and the view uses `as_strided(shape, stride)`.
Contiguous inputs take the original `view` fast path. `probe_noncontig.py`: a transposed input across a break, 2 / 2 regions
served, 0 recordings (plus 2 static regions without symbols that upstream records once each), bitwise correct; `x[:, ::2]`,
a view with holes, 3 / 3 served, 0 recordings (one `input-too-large` rebuild), bitwise correct.

### 10.1 Landed + measured (2026-09-19, exclusive GPU 6 / 7)

`config.triton.dynagraph_topology = "host"` (default) / `"switch"` (environment variable
`TORCHINDUCTOR_DYNAGRAPH_TOPOLOGY`). Each graph's state is packed into an `_Exec` (graph, handles, ctx,
pointer dirty bits, applied / flag_on, which key the children are swapped to, verified); the runner picks from `execs` by
`tuple(key_bodies[key])`; when harvest finds a new combination it captures another graph (`_MAX_EXECS = 8`), and old graphs
are kept. The SWITCH path is kept as-is, except that its re-capture becomes "replace `execs[()]`".

`bench_topology_mode.py` (4-layer GEMM, 8 addmm call sites, M=64 goes splitK; A host has only seen 512/256,
B host has seen 512/256/64 = 2 graphs, C switch same as B = 1 graph with 8 SWITCHes; interleaved timing, median per shape):

| Stream | Shape | A host 1 graph | B host 2 graphs | C switch | C - A |
|---|---|---|---|---|---|
| Same shape repeated (planner exits early) | 512 | 179.9 us | 182.5 | 255.7 | **+75.9** |
| Alternating 512/256 (same graph, planner runs every time) | 512 | 250.8 | 255.1 | 324.6 | **+73.8** |
| | 256 | 237.0 | 240.6 | 312.3 | **+75.4** |
| Alternating 512/64 (B switches between two graphs, C swaps body) | 512 | 322.3 | 333.1 | 332.6 | +10.3 |
| | 64 | 316.6 | 326.5 | 321.8 | +5.2 |

First two streams: 8 SWITCHes = **74-76 us** per replay (9.4 us each), while host-side graph selection costs nothing; holding
one more graph (B) adds only 3-4 us (the ctx / handles of each of the two graphs exit early separately). In the third stream
every configuration is ~70 us slower than in the second, for both shapes -- it is neither SWITCH nor graph switching
(`_alt_exec_cost.py` / `_alt_exec_cost2.py`: alternately launching two execs -- plain nodes, device-updatable nodes, and ones
with 8 cuBLAS children -- AAAA vs ABAB, compared exec by exec, differ by 0.0 us); it is an artifact of that particular run: in
`_alt_stream_prof.py`, the same runner running the 512/64 stream has a planner device time of 1.2 us per call (each of the two
graphs exits early; alternating 512/256 on the same graph is 7.3 us + 8 `cudaGraphExecChildGraphNodeSetParams` per call
totaling 19 us of host time), so the device time is actually lower; `_clock_check.py`
(A has seen 512/256/64 from the start, two graphs):

| Stream | Per-call event + sync, median | 40 back-to-back calls, amortized |
|---|---|---|
| [512] | 141.8 us | 92.9 us/call |
| [512, 256] (same graph alternating: planner runs + 8 child swaps) | 194.7 / 191.8 | 139.9 |
| [512, 64] (switching between two graphs: each exits early, no child swaps) | 138.5 / 141.7 | **88.6** |

Switching between two graphs costs the same as repeating a single shape, and is 50 us cheaper than changing shape within the
same graph -- host-side graph selection turns "changing topology" from the most expensive category into the cheapest. Also,
the ~50 us difference between per-call sync and back-to-back is host launch overhead
(`cudaGraphLaunch` 35 us CPU + Python); training loops do not sync every step, so for steady state look at the back-to-back column.

conv with 12 batches (`bench_conv_firstpass.py`, GPU 7, host -> switch -> host, three runs; the upstream baseline itself drifts
between 159 / 190 / 181 us, so only look at differences within the same run):

| | First-pass wall clock | Recordings | Pass-3 per-step device time, median | vs upstream |
|---|---|---|---|---|
| Upstream, record per batch | 473 / 588 / 565 ms | 12 | 159 / 190 / 181 us | - |
| DynaGraph host | 447 / - / 684 ms | 0 | 203 / - / 210 us | **+44 / +28 us** |
| DynaGraph switch | - / 695 ms | 0 | - / 425 us | +234 us (abnormally large this time; the same configuration measured +55 in the morning) |

In the SWITCH era conv's steady state was 55-63 us above upstream; now it is 28-44 us above, and what remains is the planner
itself + input copies + cuDNN variants. First pass 447 vs 473 ms is on par (after compilation moved to nvrtc).

## 13. Two paths: host-side C++ graph patching landed (2026-09-19)

Previously there was one mixed path: Triton nodes were patched on the device by the in-graph planner, cuBLAS/cuDNN children
were swapped by host Python + cuda.bindings, and symbol values were sent up one by one through `ctx[i] = int`, each as a
pageable memcpy (15 us each time). Now it is split into two,
`config.triton.dynagraph_update = "auto" | "host" | "device"` (environment variable `TORCHINDUCTOR_DYNAGRAPH_UPDATE`):

| | Host path `host` | Device path `device` |
|---|---|---|
| Who patches the graph | per region, codegen one C++ file (`generate_host_patcher` -> g++ -> .so, cached on disk by source sha), one `dg_step(syms, ...)` call: computes grid / scalars, `cuGraphExecKernelNodeSetParams`, `cuGraphExecChildGraphNodeSetParams`, pointers written only once | the planner as the first node in the graph (device graph update API) |
| How symbol values arrive | they are simply function arguments | one `dynagraph_setctx` launch, with the values carried as kernel arguments (the old per-value memcpy is gone) |
| What extra is in the graph | nothing | the planner node |
| Device time per step | 0 | planner runs 7-17 us, 1 us when the shape is unchanged |
| Host time per step | in C++ 0.5 us/node + 0.44 us/child, only touching what differs from the previous step (each node compares for itself) | launch 3 us |
| Overlap with the GPU | yes: `probe_host_overlap.py` -- while the exec is running a 28 ms kernel, patching its parameters returns in 2.8 us; the running launch uses the old parameters, the next one uses the new ones (CUDA's guarantee for exec update, holds in measurement) | no, serialized before the kernels |
| How nodes are found | in the captured graph by launch order, matched by function (all autotune variants of each kernel count); copy kernels issued by the wrapper itself are skipped; the Triton static launcher's device handles are no longer needed | the static launcher collects device handles |
| What auto picks | regions with extern child sites | regions with only Triton nodes |

The host path's C++ patches the packed parameter buffer at byte offsets (`CU_LAUNCH_PARAM_BUFFER_POINTER`, the same approach
as the static launcher); the offsets come from `cuFuncGetParamInfo`, the same table the planner uses; the grid formula is the
same C emitted by `_expr_to_c`.

Correctness (GPU 1/2, both other people's cards sitting idle, correctness runs only): `end_to_end`, `probe_extern_child` (both
topologies), `probe_conv_child`, `probe_train`, `probe_inplace`, `probe_noncontig`, `test_flag` all pass in host mode,
with numerics bitwise identical to the device path.

Which is faster (host numbers from `microbench/host_update.cu` + the device numbers above): when launch-bound, with the GPU idle
waiting for the CPU, the device path is faster by ~30 us/step (the host spends only 3 us); when GPU-bound, the host path is
faster by 17 us/step (its 36 us is hidden behind the previous step, while the device path's planner is paid for nothing). The
differences are all tens of us; the ~200 us of extra Python in the mixed path is something neither of the two paths has.
Bench numbers (launch / gpu regimes, each run with host / device) are waiting for an exclusive GPU.

### 13.1 Inputs are no longer copied; patch the pointers instead (2026-09-19)

Input handling on the host path changed: previously every call copied the inputs into the runner's store (the kernel nodes
record the store's address, which is also what upstream does), and issuing that copy on the host cost ~25 us (aten `copy_`
dispatch + building views). Now, the nodes that read an input have to issue a `SetParams` for the new shape anyway, so writing
the input's new address into it at the same time costs nothing -- each node remembers the address from last time and issues
nothing if it has not changed. Automatic rule (`_inputs_by_address`, decided once at build):

| Input | Handling | Why |
|---|---|---|
| Read by an extern child (the parameters of the cuBLAS/cuDNN small graphs are fixed at capture) | copy into the store | the small graph cannot be modified from outside |
| Read by more than 16 kernel nodes | copy into the store | the address changes every time; 16 SetParams cost more than one copy |
| Everything else (most inputs) | patch the address | rides along with the SetParams that would be issued anyway |
| A patched input whose address this time is not 16-byte aligned | copy into the store this time, patch the store's address | Inductor specialized the kernel on 16-byte alignment |

Knock-on changes: patched inputs have no store size limit, so the `input-too-large` rebuild only remains for the copied kind;
inputs written in place are written directly into the caller's tensor, with no write-back needed; at the end of the step
`cuGraphLaunch` is also issued within the same C++ call, no longer going through torch's `replay()` wrapper (the RNG and
memory pool liveness work that layer spends 57 us on is not needed by our graphs). Output views are cached by shape (arena
slots are fixed, so the outputs for the same shape are the same set of tensors).

The same set of probes (end_to_end, extern_child, conv, train, inplace, noncontig, partition_extern) all pass in host
mode; in host mode the expectation of the "exceeds input headroom" item of `test_verify` was changed to "no fallback at all, 0 recordings".

### 13.2 Neither path goes through torch's `replay()` anymore; another cut to per-step Python (2026-09-19)

The launch-regime numbers for the two paths (section 13 above / `docs/notes/BENCH.md`) show that the 19 us by which the device
path loses to the host path is not a difference in the patching mechanism; it is two pieces of Python that the device path still
carries and the host path has already shed: the `replay()` wrapper at 57 us and the input `copy_` at 25 us. This cut makes the
shell of the two paths identical, so the remaining difference is "who patches the graph" itself:

| | Before | Now |
|---|---|---|
| Launching the graph | device path: torch `CUDAGraph.replay()` (RNG prologue with two `fill_`, pool liveness check, hooks) | both paths use `cuGraphLaunch(exec, stream)`, one ctypes call; the host path still launches at the end of `dg_step` |
| Input copies (device path: all inputs; host path: only those read by extern children) | aten `copy_` (dispatch + building views) | contiguous and same dtype: one `cuMemcpyDtoDAsync_v2` driver call; the rest still `copy_` |
| In-place write-back | aten `copy_` | same as above |
| Fixed per-step Python | symbol sorting, two held dicts, output zip, `config.triton.*` reads (each ~5 us through `_config_module.__getattr__`), the `torch.cuda.current_stream()` object | tables computed once on the first call (`_hot_tables`); exec cached by shape; `_cuda_getCurrentRawStream`; `verify_shapes` cached |

**Correctness precondition for launching directly**: torch's `replay()` is expensive because every time it advances the CUDA
generator for the philox ops captured in the graph (`replay_prologue`). Our graphs should not contain such nodes in the first
place -- Inductor's random numbers go through a seed tensor, the seed is produced by
`aten.randint`, and DynaGraph classifies that as "run on the host every time" (`_EAGER_OP_NAMES`). But that list is maintained
by hand; a custom random op that is not on the list and gets captured into a child graph would draw the same set of numbers on
every launch, silently wrong. Now `_harvest` reads `default_generators[dev].get_offset()` before and after the warm-up run of
each extern site: if it moved, the site is automatically classified into the run-on-host category (and added to `eager_sites`,
after which reference / harvest both protect the RNG state). The list becomes a backstop, no longer the only line of defense.
This is the "badly written implementations" axis.

**Pitfall**: in ctypes, `libcuda.cuMemcpyDtoDAsync` resolves to the CUDA 3 era v1 symbol (32-bit pointers),
which returns 201 (invalid context); cuda.h maps the name to `_v2` with a macro, but ctypes does not. That is why the first
regression run failed a chunk in each of the two modes. The C++ side compiles with cuda.h, so it does not have this problem.

Fixed along the way: when the per-key cache was cleared at 1024 shapes, `host_args` was not cleared, so a re-harvested shape
would get a child handle that had already been freed.

Numbers (`_call_cost.py`, splitting the per-step host time into three layers: compile shell / runner Python / C++ call):

`_call_cost.py`, GPU 4 (shared with someone else's diffusion ablation, 100% util, host load 232), the bench's 24-site model,
250 steps of random shapes (the shape changes every step), min of 5 rounds, per step on the host side:

| | Host path | Device path (with the setctx arguments not yet prepared in advance) |
|---|---|---|
| Compile shell (dynamo / aot / cudagraph_trees, outside the runner) | 265 us | 397 us |
| The runner's Python | 72 us | 141 us |
| The single C++ / driver call | **224 us** (`dg_step`: ~60 kernel node SetParams + 24 child swaps) | 92 us (`setctx` launch, including ~50 ctypes objects built on the fly in Python) |
| Total | 560 us | 630 us |

The two columns cannot be compared against each other (run one after the other, other people on the card, 70-130 us swing
between rounds); only the breakdown within each column is trustworthy. What to look at:

- The bulk of the host path's per-step cost is no longer Python, it is `dg_step` itself: the shape changes every step, so all
  60 nodes + 24 children need SetParams, which is the "host cost linear in the number of nodes" described in TIERS. The
  unloaded microbench gives 0.5 us/node + 0.44 us/child ~ 42 us; everything on this machine is currently 3-5x that. When
  launch-bound there is nothing to overlap this with, so it is pure cost.
- Most of the device path's 92 us launch is Python building the arguments on the fly (`_launch_vals` creates ~50 new ctypes
  objects every time). Changed to prepare them once per shape (`ctx_args`, one copy each for flag up/down), so a call is just
  `cuLaunchKernel`.
- So the launch-bound criterion holds: the device path's host cost is constant (one launch), the host path's is O(number of
  nodes). When GPU-bound it is the other way around (the host path's SetParams hide behind the previous step, while the device
  path's planner 7-17 us is actually paid). The auto path selection was changed accordingly to measure once at build: temporarily
  capture a plain graph, replay it three times and take the median as G; if `G >= 0.5*N + 60 us` choose the host path,
  otherwise the device path; regions with extern child sites are still fixed to the host path (swapping a child takes 0.44 us
  in C++, 2.4 us in Python). The 60 us term is a placeholder for "per-step Python outside the region", which is what the GPU
  is waiting on.

### 13.3 The gpu regime hits illegal instruction right away: when a child is swapped in, its cluster configuration does not follow (2026-09-19)

`bench.py --regime gpu` (3 blocks, d_model=1024, 6 cuBLAS mm sites) had never run through on the child route: after the second
graph was captured, launching the next shape immediately gave `cudaErrorIllegalInstruction`, the same on the host path and the
device path, the same when running row D alone, and independent of whether A/B/C ran first.
`CUDA_LAUNCH_BLOCKING=1` pinned the error on `cuGraphLaunch` itself; `_child_sig_probe.py` printed the child node for each shape:

| M (s77) | cuBLAS kernel | grid | cluster |
|---|---|---|---|
| 947 | `sm90_xmma_gemm_..._tilesize64x128x32` | (64,2,1) | **(1,2,1)** |
| 150 | `sm90_xmma_gemm_..._tilesize64x64x32` | (60,1,1) | **(5,1,1)** |
| 200 | same as above | (16,5,1) | **(1,5,1)** |
| 110 / 64 | `cutlass_80_tensorop_s1688gemm` + `splitKreduce` (2 nodes) | - | (0,0,0) |

The node count is 1 in all of them, so under "node count = topology" they are the same body, and
`cudaGraphExecChildGraphNodeSetParams` swapped the child for 150 into the exec instantiated for 947: the function and the
parameters were swapped, **the cluster dimensions were not** -- the exec still launches a kernel that needs (5,1,1) with a
(1,2,1) cluster, the grid does not divide evenly either, and the GPU reports illegal instruction. The launch regime
(d_model=128) never hit this because cuBLAS does not use cluster kernels for small matrices (all (0,0,0)), and swapping back
and forth among 94 variants was always correct.

Fix: the topology signature changed from the node count to each node's (type, clusterDim, cooperative) (`_node_sig`). A
different cluster means a different body, which under host-side graph selection means a different exec; a different function
with the same cluster is still the same body (which is exactly why the child route is cheap). Cost: cluster variants inflate
the body count -- this model sees four kinds across 64 shapes, (1,2,1)/(5,1,1)/(1,5,1)/splitK, and
`_MAX_BODIES = 8`, `_MAX_EXECS = 8` are enough; models with more variants will have shapes handed back to upstream with
`extern-topology` (the coverage axis pays, correctness does not). The other option is to go through `cudaGraphExecUpdate`
whenever the cluster changes (that one can change the cluster, verified on the morning of 09-19), but in a stream where the shape
changes every step that amounts to a full-graph diff every step, so it was not chosen.

A cuda.bindings pitfall along the way: the value object returned by `cuGraphKernelNodeGetAttribute` is a view onto the same
`CUlaunchAttributeValue` union buffer, so querying another attribute right after it (cooperative) overwrites clusterDim with
0 -- the first version of the signature read everything as (0,0,0) and crashed just as if nothing had been fixed. You have to
copy x/y/z into ints before querying the next attribute.

The full numbers for the gpu-bound regime are in `docs/notes/BENCH.md`, "The two paths' bill, gpu-bound": the three
configurations tie (DynaGraph 1.01x A), the host path is 4 us/step faster than the device path (the planner cost), first pass
4.6-8.2 s (harvest of 58 shapes x 6 sites + 6 re-captures).

## 14. Coverage filled in this round (2026-09-19 evening)

Ordered by "simple and important first"; one probe per item, run on both update paths:

| Item | Probe | Result |
|---|---|---|
| SDPA (the fallback call of flash / efficient attention: multiple outputs, self-allocated outputs) | `probe_sdpa.py` | bf16 causal / bf16 / fp32: all three cases served by one graph, 0 recordings, bitwise identical to eager (fp32 2.4e-7). The child route already harvests per call site; multiple outputs and self-allocated outputs use the EXT pointer slots from section 7, no code change |
| NCCL: all_gather / reduce_scatter / all_reduce -> all_gather -> reduce_scatter chained | `probe_nccl_more.py` (select with COLL=) | 2 ranks (GPU 4+5): all three served, recordings 4 -> 0, both ranks agree, numerics 0 / 1.9e-6. Same as section 9: a size change does not change NCCL's topology |
| Training: saved activations for backward move after a forward REBUILD | `probe_train.py` (shape stream 64 -> 200 triggers a rebuild) | Host path: static inputs go into the patch set, so a move is re-pointed by the next `dg_step` according to what changed (`_rebind_static`) and no longer exits; device path, and static inputs read by an extern child: return REBUILD (rebuild within the `dynagraph_rebuilds` limit) instead of exiting with `static-input-moved` |
| 6 unmodelled grid types | `probe_grid_types.py` | Modelled the 4 formula-based ones (`_GRID_AXES` per axis: cdiv / numel / const): BatchMatmulGrid3D, CooperativeReductionGrid, SplitScanGrid, MixOrderReductionGrid. The remaining two are not done: PrecomputedGrid as distinct from FixedGrid (per-config lookup table for user Triton kernels), and the ComboKernel family (horizontal fusion, off by default) |
| Cooperative-launch nodes | same | Both paths hit it: the device path cannot get the handle (it does not go through the static launcher), and on the host path `SetParams` returns INVALID_VALUE -- **the root cause is the parameter form**: Triton's own launcher builds the node with `kernelParams` and does not accept updates in the packed `extra` form. The host patcher now always uses the `kernelParams` form (one pointer per parameter, pointing into the packed buffer), which accepts nodes from both sources. Under auto, a region containing a cooperative kernel is pinned to the host path |

**Applicability sweep** (`applicability.py`, default Inductor config + extern child, 13 real models, GPU 4): see the table below.

### 14.1 The first thing the sweep hit: ViT's `buf14 = buf13[0]` (2026-09-19 evening)

In the applicability sweep the first three convnets (resnet18 / mobilenet_v3_small / efficientnet_b0) were all served by one graph, bitwise identical;
vit_b_16 fell back with `unmodelled: triton_poi_fused_clone_permute_7 argument in_ptr0 is buf14, which no allocation owns`.
The wrapper has `buf13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(...)` (returns a tuple),
`buf14 = buf13[0]`, and later `buf19 = reinterpret_tensor(buf14, ...); del buf14  # reuse`. The alias table only recognizes
`bufA = bufB` / `reinterpret_tensor(bufB, ...)`; nothing recognizes taking an element of a tuple, so `buf14` became a buffer with no owner.
`probe_sdpa.py` did not hit this because on the flash path the wrapper directly does `reinterpret_tensor(buf13[0], ...)`.

Fix: the pointer slots for extern outputs changed from "one per site" to "one per site + one per tuple element that is extracted" (`ext_slots`:
`(site, elem)`; `extern_outs_of[name] -> slot`). The planner's ctx, the setctx arguments, the host patcher's `ext[NEXT]`,
the addresses sent by `_host_step` / `_write_ctx`, and output pass-through all go by slot; each element has its own address under each shape,
taken from `results[site][elem]` after harvest. After the fix ViT is served, recordings 4 -> 0, the four batches bitwise identical.

### 14.2 split scan: the kernel itself is not bitwise identical across two runs (2026-09-19 evening)

The cumsum in `probe_grid_types.py` (8 rows x 1M) generates a `SplitScanGrid`; the grid formula is right, but the self-check reports
`runtime-mismatch`. `_splitscan_determinism.py`: the same input run 5 times, max diff 4.9e-4, not bitwise identical --
atomic ordering in the decoupled look-back. The self-check compares bitwise on purpose (same kernel, same config, same input: a one-bit difference means we broke something),
but that does not hold for this kind of kernel. Inductor marks `atomic_add_found` in `inductor_meta`; the kernel table now carries it, and a region
containing such a kernel is compared with a tolerance (`_same_values(..., exact=False)`: diff <= 1e-3 x the reference's max magnitude; absolute noise shows up
near zero crossings, so it cannot be element-wise relative).

### 14.3 Device path: large inputs are no longer copied either (2026-09-19 evening)

The host path has long patched inputs by address (section 13.1). The device path always copied -- now a single `cuMemcpyDtoDAsync`, 1.5 us on the host,
the GPU side goes through the copy engine, 4 MB is only about 2 us -- cheaper than having the planner run once (7-17 us), so copying small inputs is right.
What is wrong is large inputs: activations of tens of MB, decode's KV cache; the bandwidth of each copy is paid in full, and copied inputs have a store size limit
(exceeding it means REBUILD). Now `config.triton.dynagraph_patch_bytes` (default 16 MiB, env var
`TORCHINDUCTOR_DYNAGRAPH_PATCH_BYTES`, 0 = copy everything): at build time inputs are split by their byte size, and non-static,
non-extern-read inputs at or above the threshold go into the device path's patch set.

The mechanism is the same as the EXT slots for extern outputs: after the SWITCH handles, ctx also holds an "input moved" flag + one address slot per argument position;
the `setctx` arguments carry them (in the argument array prepared once per shape, the address holders are modified in place on every call); the planner's
early-out becomes "return if the shape has not changed and no input has moved"; each node first does the input pointer patch (placed before the grid section: a node
disabled under this shape must still take the address, so that it is correct when it is enabled again), then checks the shape flag to decide whether to compute the grid. On the host side, each call compares
the `data_ptr()` of the inputs in the patch set with the address this exec last patched (`_Exec.last_in`), and raises the flag only if it changed.
Static inputs do not go into the device path's patch set (comparing the addresses of 48 weights on every call is 20 us of Python); a move still goes through REBUILD.

`probe_input_patch.py` (L x 4096 fp32, L up to 2048 = 32 MiB, two different tensors at the same length): device-path patch set
[0], 0 recordings, no fallback, the runner's own bitwise check passes.

### 14.4 Applicability sweep (2026-09-19 evening, GPU 4 shared, default Inductor config + extern child, auto path selection)

`applicability.py --list survey/sweep14.txt`: one subprocess per model, 4 batches (8 first, then 2/5/3);
the control group is cudagraph_trees recording per batch on its own; we look at recording count, fallback labels, and numerics.

| Model | Result |
|---|---|
| resnet18 / mobilenet_v3_small / efficientnet_b0 / convnext_tiny / regnet_y_400mf / mobilenet_v2 | **Served**, recordings 4 -> 0, all four batches bitwise identical to the control group (all convolutions are cuDNN children) |
| vit_b_16 | `unmodelled` during the sweep (tuple output `buf14 = buf13[0]`); after the section 14.1 fix, run alone: served, bitwise identical |
| inception_v3 | `runtime-mismatch` during the sweep (the offset of the cat slice view was lost, section 14.5); after the fix: served, 83 kernel nodes + 95 cuDNN children, 4 batches bitwise identical to eager |
| swin_t / maxvit_t | Compile timeouts at both 1500 s and 3600 s: the subprocess sat at 100% of a single core compiling the whole time (maxvit used 38 minutes of CPU in 2300 s); with host load at 300-400, Inductor's single-threaded codegen/autotune is too slow, and the sweep has to compile twice; to be filled in when the load drops |
| fasterrcnn_resnet50_fpn | Skipped: the first argument is not a variable-batch tensor (the probe's input-reshaping rule, not a DynaGraph problem) |
| beit / BertForMaskedLM | After installing timm / transformers: both served, bitwise identical (BERT's inputs are keyword arguments; the sweep script's batch-varying rule was changed to take the first tensor) |

Of the 9 that could be judged, 7 are served (including vit after the fix), and the numerics are all bitwise identical. The fallback reasons are no longer `extern-launch`
(before the child route, every model with a GEMM/conv fell on this one) but specific things that can be fixed one by one.

### 14.5 Root cause for inception: cat's slice views lost their offset (2026-09-19 evening)

`_model_check.py` + `TORCHINDUCTOR_DYNAGRAPH_DEBUG=1` (on self-check failure, prints each kernel's grid inputs and the first
wrong buffer in the arena) narrowed it to a split partition of 2 kernels: the output of `max_pool2d_with_indices_15` is
`reinterpret_tensor(buf57, (8,288,12,12), ..., 41472)` -- `torch.cat` is implemented by having each producer kernel write directly into
its slice of the concatenation buffer; 41472 = 288 x 12 x 12 is the channel offset (the spatial size is symbolic, so the offset is symbolic too). After resolving the alias to
`buf57`, the pointer patch wrote the **slot base address** and the offset was lost: the kernel wrote into the first 288 channels, and its own segment was all 0. Every net with a cat
(inception, densenet, all FPNs) would hit this.

Fix: `_view_of(raw)` resolves a call-site argument (an alias, a `reinterpret_tensor` view, or an inline `reinterpret_tensor(...)`)
into (the buffer that owns the storage, compound element offset); pointer arguments with a nonzero offset form their own section (the planner's `VIEWPTRS`, run on every shape
change because the offset may contain symbols; in the host patcher, "write only if the computed value differs from the stored one"), address = slot base address (or the input address: inputs read in place
use ctx/`last_in`, copied ones use the store's fixed address) + offset x itemsize. Offset views of extern outputs are rejected for now (itemsize is unknown
before harvest). `probe_cat_views.py`: a three-way cat (including a channel slice of an input) + the mean of another cat, with both batch and spatial dimensions dynamic.

After the fix inception_v3 (child route, auto -> host path): all 4 batches bitwise identical, two graphs (cuBLAS bucketing).

`probe_cat_views.py` passes on both paths (relative diff 1e-7, from the reduction order of mean). The first version on the device path was wrong in generation order: the offset expressions were
registered into the EXPRS table via `idx()` inside `generate_view_pointer_patches`, but the EXPRS source section had already been assembled before that, so in the planner
`dg_eval` could not find them and the offset evaluated to 0 -- the same as not fixing it. Now the view patches are generated first and the table is assembled afterwards.

### 14.6 Wrap-up: regression (2026-09-19 late night, GPU 5 shared)

`_regress.sh`'s list gained `probe_sdpa probe_grid_types probe_input_patch probe_cat_views` (29 scripts):
host 29/29, device 29/29, auto quick list 9/9. The last two small pitfalls fixed along the way: the cooperative flag was read from the compiled-variant object
instead of the autotuner (auto chose the device path for a cooperative region -> handle-mismatch; it now reads `obj.triton_meta`, and treats
`CooperativeReductionGrid` itself as cooperative); `probe_cat_views`'s tolerance was changed to relative 1e-5 because of mean's reduction order.

Not done this round, left on purpose: unbacked SymInt (last); PrecomputedGrid / combo kernel grid (off by default; for user
kernels the user writes the wrapper); nonzero-offset views of extern outputs (rejected); >2-rank NCCL (no GPUs); exclusive-GPU timing of the two paths
(tonight only the shared eval GPUs were available); swin_t / maxvit_t compile timeouts under load 200+ (not a fallback); beit / bert lacking
timm / transformers.

## 15. unbacked SymInt, tier 1 (2026-09-20 early morning)

First look at the wrapper (`_unbacked_wrapper.py`, `x[x[:, 0] > 0]` + `capture_dynamic_output_shape_ops`): Inductor splits the graph into two pieces at
the op that produces an unbacked size (here `aten.index` / nonzero); that op runs eagerly between the two pieces, and
the second piece's signature is `buf1, s27, u0 = args` -- `u0` is an **int argument on the host**, no different from `s77`. So in
the cudagraph_trees flow, the unbacked value never needs to be read on the device: the graph-split step has already synced it back to the host (upstream
pays this sync too; it is the cost of `.item()` semantics itself). DynaGraph had been rejecting it only because four regexes recognized only the spelling `s\d+`
(`_is_symbolic`, the kernel table's symbol collection, two places in `_input_symbol_map`). Changed to `[su]\d+`.

`probe_unbacked.py` (bool mask / nonzero / masked_select / unique, shape stream 64 -> 200 -> 33 -> 128 x 2):
bool mask and nonzero are served by one graph on both paths, 0 recordings, bitwise identical. masked_select is a different issue: for an unbacked
size there is no such thing as "largest first"; the slot is sized by the first u0 seen x headroom, and a later, larger u0 can only REBUILD (limit
`dynagraph_rebuilds`) -- this is a path within the design; the probe now counts "only REBUILD-type labels and 0 recordings" as a pass. The real fix is to give
unbacked slots an upper bound (the upper bound for nonzero is the input numel; Inductor's value range has it, but it is not printed into the wrapper),
left for tier 2. unique hits an exception; under investigation.

Xinwei's question "should we insert kernels into the graph that change things based on results in device memory": with Inductor's current way of splitting graphs, this tier does not need it. What needs to read values on the device
is the case without a graph split -- a pure `capture_scalar_outputs` `.item()` used within the same graph (that messy case is currently Inductor's
own NameError) -- and that is what tier 2 has to do: put pointers in ctx, have the planner read symbol values from device addresses, place planner nodes in sections split by unbacked symbol,
and allocate by upper bound.

**Headroom for unbacked slots**: `dynagraph_unbacked_headroom` (default 4.0, env var `TORCHINDUCTOR_DYNAGRAPH_UNBACKED_HEADROOM`):
slots whose size expressions contain `u<n>`, and the input stores of regions with unbacked symbols, get this multiple of headroom instead of 2x.
In the masked_select part of `probe_unbacked` (`buf4: 1 + (u0 - 1)`, `buf5: ...min(u0, ...)`) both slots are now 4x;
the remaining `arena-too-small` rebuilds all come from the **previous piece** (the one computing the mask, where input x grows from 64 rows to 200 -- this is the ordinary
"inputs did not come largest first", unrelated to unbacked, and REBUILD has a limit). unique is a separate case: Inductor treats the
**whole tuple** of `_unique2` as the input of the next piece (`buf1, s27, s77, u0 = args; buf2 = buf1[0]`); checking how upstream itself records it.

**Conclusion on unique**: it is upstream's. `_unique_trace.py`, with dynagraph off, also dies while recording the second piece (`FunctionID 1`) on
an assertion in `_allocate_and_copy_recording_inputs` -- cudagraph_trees does not accept tuple inputs, and Inductor made the whole return tuple of `_unique2`
an argument of the next piece. DynaGraph's fallback for it (`buf2 = buf1[0]`: a tuple element of an input, `unmodelled`)
merely surfaces this upstream crash earlier. To do better than upstream we would have to model "tuple elements of inputs" (`argv` position -> tuple -> element),
a niche case, not done for now; the probe records it as "upstream cannot record it either".

## 16. The remaining three items (2026-09-20 early morning)

### 16.1 combo kernel grid: sum of the sub-kernels' block counts

`config.combo_kernels=True` (off by default) horizontally fuses independent pointwise/reduction kernels into one `triton_poi_fused_0`;
in the wrapper `xnumel_0 = s97*s98; xnumel_1 = s20*s97; ...` are passed in as **runtime arguments**, and `inductor_meta`
gets an extra `combo_grid_meta` (`num_kernels`, each sub-kernel's `no_x_dim_i`, `xnumel_i` (a constant when static,
None when dynamic, meaning "look at the argument"), `ynumel_i` when there are 2D tiles, `min_blocks`, `default_config`).

Inductor has three dispatches: `SequentialDispatch` (x = sum of cdiv(xnumel_i, XBLOCK); sub-kernels with no x dimension contribute
their numel directly), `RoundRobinDispatch` (x = max(...) x num_kernels), `SequentialFlattenGridDispatch` (each sub-kernel
has its own XBLOCK_i/YBLOCK_i; requires `combo_kernel_per_subkernel_blocks`). In `select_dispatch_strategy`,
**as long as any numel is symbolic, Sequential is always chosen** (`any(isinstance(e, str) for e in x_numels_list)`);
RoundRobin only appears with fully static shapes -- that is, for the dynamic regions we want to serve, the grid is always the summing kind. With 2D-tiled
sub-kernels, the y axis takes the max of the `ynumel_i` and then folds into z the same way as `Grid2DWithYZOverflow`.

Landing it: the kernel table gets a `combo` entry (which is just `combo_grid_meta`); `_grid_numel_names(k)` replaces the two places that used to read
numel names from `_GRID_AXES` (for combo they are `xnumel_i`/`ynumel_i`); `_combo_grid_stmt(k, ev)` generates the C statements that set
gx/gy/gz -- `ev` is "wrapper expression -> C", which is the `S(i)` scheme on the host path and `dg_eval(i, ctx)` on the planner path,
so one piece of code serves both paths. Both Sequential and RoundRobin are modelled (the latter only for completeness of the table); the Flatten kind gives
`Unsupported` (non-default config, not done for now). The generated code is:

```c
{ gx = dg_floordiv(S(1) + 32 - 1, 32) + dg_floordiv(S(1) + 32 - 1, 32);
  int64_t ym = S(2); { int64_t t = S(0); if (t > ym) ym = t; }
  int64_t raw = dg_floordiv(ym + 32 - 1, 32); int64_t div = dg_floordiv(raw + 65535 - 1, 65535);
  gy = (div == 0) ? 0 : dg_floordiv(raw + div - 1, div); gz = div; }
```

`probe_combo.py`: three 1D pointwise kernels (different numel) and two 2D transposes, shape stream 256 -> 100 -> 300 -> 17 -> 256; on both paths:
one graph, 0 fallbacks, differs from eager by float32 rounding (sigmoid/tanh implementations differ, 1.2e-7; the transpose group is bitwise identical).

`PrecomputedGrid` (the grid lambda that comes with a user Triton kernel) is not done: that is Python outside the wrapper,
and "let the user write the wrapper" is already decided, so we do not parse it.

### 16.2 Non-0-offset views of extern outputs: unreachable

Last round left an `Unsupported`: "a Triton kernel's pointer argument is a non-0-offset view of an extern output". This round we first looked for
a way of writing code that hits it, and only then decide whether to do it. `_extoff_wrapper.py` tried six: in-place add on a view (`y[:, 1:] += 1`), zeroing a view,
extern writing directly into a 0-offset slice of a cat, writing into a non-0-offset slice (`buf0 = reinterpret_tensor(buf2, ..., s98);
extern_kernels.mm(..., out=buf0)`), in-place multiply on a view of the sdpa tuple output, and `index_put_` on a view. **All six were served
by one graph, bitwise identical; none reached that `Unsupported`.** The reason is Inductor's codegen itself:

- A Triton kernel's **inputs** are always base buffers; the view's offset is folded into the index expressions inside the kernel (the loader of `ReinterpretView`),
  and `reinterpret_tensor(<extern output>, ..., off)` never appears in the wrapper as an input;
- In-place writes to a view become `slice_scatter` via functionalize, and Inductor lowers that to a pointwise kernel that reads the whole buffer and writes the whole buffer
  (`buf1` newly allocated); the output is still a base;
- The only thing that makes a Triton kernel **write** into an offset view is cat (`NonOwningLayout`), and cat's buffer is the wrapper's own
  `empty_strided_cuda` -- an arena slot, already covered by 14.5;
- When the extern itself does `out=` into an offset view, that pointer is computed by the wrapper at harvest time and does not go through patching.

So this `Unsupported` is a guard, not a gap, and it stays. If we ever hit it, the approach is already clear:
read itemsize from the `assert_tensor_metadata(bufN, ..., torch.<dtype>, ...)` line, and replace the base with
`e->ext[slot]` / `ctx[ext0 + slot]` plus offset x itemsize.

### 16.3 NCCL on three GPUs

`probe_nccl_more.py`'s shape stream is rounded to a multiple of `WORLD_SIZE` (reduce_scatter requires divisibility); three ranks on GPUs 2/4/5:
all_reduce, all_gather, reduce_scatter, and chain: each served by one graph across the three ranks, recordings 4 with DG off -> 0 with DG on,
max on-off diff 0 (chain 7.6e-6, cuBLAS algorithm selection). Logs `_regress_logs/nccl3_*.log`.

### 16.4 Design of tier-2 unbacked (not done; last in Xinwei's ordering)

Xinwei's three questions -- can it only be solved on the device, do we need to insert kernels into the graph that act on results in device memory, does allocation get more complex --
the answers are: yes, yes, and yes, and they are three faces of the same thing. The current tier 1 works only because cudagraph_trees splits the graph at
ops that produce unbacked sizes, so `u0` enters the next piece as a host int. What tier 2 has to do is **not split**: nonzero / masked_select /
unique stay in the graph, which requires

1. A graph-safe implementation: these aten ops all internally do a synchronous `cudaMemcpy` to read the count that sizes the output, which cannot be recorded into a graph. They need to be replaced by
   a version that "allocates the output at the upper bound (input numel) + writes the count into a device int64" (cub's select/compact has exactly this shape);
   on the Inductor side this is a decomposition or lowering replacement, enabled only when DynaGraph is on;
2. The planner reads symbols from the device: the `u0` slot in ctx is written by **the preceding kernel** rather than by the host; planner nodes have to be
   split into sections by unbacked symbol -- nodes before u0 is produced are patched by the first planner, nodes after it by a second
   planner inserted right after the count (that is what "inserting kernels into the graph" refers to). The host path cannot do this tier; it can only be the device path;
3. Allocation by upper bound: all slots whose size contains `u0` are fixed at the upper bound of the value range (Inductor's `bound_sympy` has it, but it is not
   printed in the wrapper, so it has to be taken from the graph); the `dynagraph_unbacked_headroom` scheme is then retired;
4. Region output: the returned tensor's size contains `u0`, so on leaving the graph we still need one sync to read the count and then `as_strided` it to the real size -- once per piece,
   not once per op; this is the part where it saves over upstream.

The work is in 1 and 2, and neither is internal to DynaGraph (1 is Inductor lowering, 2 requires changing the planner's generated structure and
the node order in `_capture`). There is currently no use case that triggers it (in-graph `.item()` under reduce-overhead is Inductor's own
NameError), so this tier waits until a real model needs it.

### 16.5 Regression (2026-09-20 early morning, GPU 3 exclusive)

`_regress.sh`'s list gained `probe_combo`, 31 scripts: host 31/31, device 31/31 (logs
`_regress_logs/full_host/`, `full_device/`). lintrunner clean. Committed `DynaGraph: model combo kernel
grids in both update paths`, pushed to `fork/dynagraph`.

At this point everything doable on the previous round's "not implemented" list is done: combo grid done, extern offset views shown to be unreachable, NCCL on three GPUs passes;
the two items not done each have a clear reason -- PrecomputedGrid is a choice made on the "user writes the wrapper" line, and tier-2 unbacked is
a separate project that requires changing Inductor lowering (16.4), placed last.

## 17. Layout goes back to being laid out per shape on every call (2026-09-20 early morning)

Xinwei pointed out that fixed slots are wrong: the memory plan was moved onto the GPU in the first place for the packed-batch scenario where the total seqlen stays constant while each sample varies
-- fixed slots turn memory into "sum of each slot's max x 2", which is exactly the worst case in that scenario, and the 2x multiplies every
buffer and every copied input. Fixing the slots on 9-19 was done for planner latency (moving the pointer SetParam from every replay to once, 42.7 -> 18.6 us),
treating a single optimization on the latency axis as the design. This round changes it back; fixed slots remain as an option.

### 17.1 Two layouts, one code path

`config.triton.dynagraph_layout`: `dynamic` (default) lays out by symbol values on every call -- slot assignment (who shares with whom, by lifetime)
is still static, a slot's **size** is the max of the size expressions of the buffers in it, offsets are 256-byte-aligned prefix sums; the arena is only as large as "the largest
total seen", and when that is not enough it is swapped for a larger one (`_grow_arena`, `dynagraph_grow` default 1.0 = exact, nothing extra). `fixed` is
the old behavior: fixed at build time from the first shape x `dynagraph_headroom`, REBUILD when exceeded.

Both layouts go through the same code: the layout formulas are generated into the device path's `setctx` (written into ctx's `OFF0..`) and the host path's `dg_layout`
(an array computed onto the stack at the start of `dg_step`); for the fixed layout the generated values are constants. Pointer patches all became "compare, then write" -- on the device path each pointer
argument records the last written address in ctx (`LASTP`) and issues the runtime call only if it changed; on the host path it compares against the existing value in the node's parameter struct,
and marks the node for SetParams only if it changed. So under the fixed layout the device path still "writes pointers only once" (the comparison costs nothing), under the dynamic layout the host path
is free (when the shape changes the node needs SetParams anyway, and the pointer is written into the same struct along the way), and only dynamic + device path pays extra for the calls on
pointers that moved.

The arena base address also went into ctx (`ARENA`), and the planner no longer takes the arena pointer as a kernel argument -- swapping the arena is just a new value in the next setctx,
without re-recording the main graph. After an arena swap the only things to redo are the harvested extern child graphs (they recorded absolute addresses; `_invalidate_harvests`
clears the cache, and they are re-harvested when the shape comes again); the main graph's child nodes are swapped into the new graphs on the next `_swap_children`.

### 17.2 Inputs no longer 2x

The store for copied inputs is sized exactly for the first shape and swapped when something larger arrives (`_grow_store`); the 2x headroom is gone. This is possible because all tensor inputs
(copied, read in place, static) are now read from the same address table (`in_ptrs`), and both paths patch from the table: the host path already did,
and the device path extends `INPTRS` to all inputs -- a copied input's address changes only once per growth, a negligible cost. The generated code no longer contains any literal
input addresses. The two REBUILD kinds `input-too-large` and `arena-too-small` now only occur with the fixed layout; `static-input-moved`
only REBUILDs when the input moved to an address that is not 16-byte aligned; otherwise the table is updated, and if the input is read by an extern it is re-harvested.

### 17.3 Budget for topology graphs

`_MAX_RECAPTURES=4`, `_MAX_BODIES=8`, `_MAX_EXECS=8` were numbers we picked ourselves, not driver limits; they are merged into one
`dynagraph_max_graphs` (default 8). In host path-selection mode, exceeding it no longer hands the shape back to upstream; instead the least recently used exec is evicted
(each exec records the sequence number of the last call it served), and it is recorded again when needed (milliseconds).

### 17.4 packed batch probe (`probe_packed_layout.py`, GPU 3)

Five splits of B x L = 4096, then one with double the total, then back to the small ones, each twice; the intermediates come in three kinds, (B,D), (L,D), (B,L), so
when the split changes, slot sizes grow and shrink against each other.

| Layout | Update | Regions | Recordings | arena | Input store | Growth | Max relative diff |
|---|---|---|---|---|---|---|---|
| dynamic | host | 1 | 0 | 0.30 MB | 1.05 MB | arena 2 times | 1.3e-7 |
| dynamic | device | 1 | 0 | 0.30 MB | 2.10 MB | arena 2 times, input 1 time | 1.4e-7 |
| fixed | host | 4 (3 REBUILDs) | 1 | 1.43 MB | 4.19 MB | - | 1.4e-7 |
| fixed | device | 4 (3 REBUILDs) | 1 | 1.43 MB | 5.24 MB | - | 1.2e-7 |

With the dynamic layout the arena is 1/4.8 of the fixed layout's and the input store 1/4 to 1/2 (the host path reads inputs in place, so its store does not grow), and there is no
REBUILD and 0 recordings. The fixed layout built the region four times and also recorded one graph -- that is the cost of "the first shape is not the largest".

The device path's latency cost (under the dynamic layout, extra patches of moved pointers on every shape change) has not been measured yet; once the regression below passes, measure
both layouts with bench on GPU 3.

### 17.5 Two bugs the layout change turned up

1. **Inputs read in place have no store to grow.** When harvest runs the wrapper, it makes "a view of the store at this call's shape" for every non-static input;
   inputs read in place were not copied this time and their store is still the build-time size, so once the shape gets bigger it hits `setStorage ... out of bounds`
   (probe_inplace / probe_noncontig, host path). Changed so that inputs read in place hand the caller's tensor directly to harvest -- externs
   do not read in-place inputs (excluded by `_inputs_by_address`), so the addresses harvest captures are still all stores held on our side.
2. **Child handles may be reused after a re-harvest.** probe_noncontig on the host path showed `runtime-mismatch` 1 time in 7 runs.
   Growth clears the old harvested child graphs (`child_holds`), and the driver may hand out the same handle value to a newly created graph, while both paths decide whether to swap a child
   by "did the handle change" (host `child[s] != e->child[s]`, device `site_applied_raw`); with the same handle no swap happens, and the exec
   keeps the old child graph that reads the old addresses. Now `_invalidate_harvests` marks all execs `ptr_dirty` and clears applied_raw, and
   `dg_step` swaps children unconditionally when ptr_dirty. In the 12 runs after that (6 per layout) it did not recur. The old "clear the harvest cache when it reaches
   1024" code also goes through this path; it had the same hole.

### 17.6 Regression (2026-09-20 early morning, GPU 3 / GPU 5 shared)

Dynamic layout, all 31 scripts: host 30/31, device 30/31; the only failure is the third scenario of `test_verify` -- it expects the large shape to hit
`input-too-large`/`arena-too-small`, but under the dynamic layout it grows instead of falling back. Changed to judge by layout (dynamic expects no fallback and 0 recordings,
fixed expects `arena-too-small` -- the input store grows under both layouts, so `input-too-large` no longer exists); all four combinations
(two layouts x two paths) pass. Plus `probe_packed_layout` (four combinations) and 12 host-path reruns of probe_noncontig.

Not yet measured: the extra pointer calls the device path pays on each shape change under the dynamic layout (expected +20 us @ 48 nodes), and the
peak memory of the two layouts on bench. Needs an exclusive GPU; right now someone else's job has shown up on GPU 3.

### 17.7 Latency of the two layouts (exclusive GPU 5, see the last section of docs/notes/BENCH.md)

The host path is even (69.7 vs 70.6 ms / 256 steps); the device path with the dynamic layout is +9 us per step (77.6 vs 75.3); in natural order the dynamic layout does not rebuild
(first pass 108 ms vs 330 ms). So defaulting to dynamic costs essentially nothing; fixed is kept for scenarios that only use the device path and where shapes really do come "largest first".

## 18. Real training steps (2026-09-20 early morning)

### 18.1 transformers Llama / GPT-2, variable-length training steps

`probe_train_step.py`: transformers LlamaForCausalLM (hidden 256, 4 layers, 4 heads, SDPA, vocab 1024) or
GPT2LMHeadModel (same size, dropout 0), `torch.compile(model, dynamic=True, mode="reduce-overhead")`,
forward + backward + AdamW (eager), batch 4, seqlen stream [64,128,96,256,32,160,128,224] run twice, DG off/on with the same seed
and the same data; compare recording counts, fallback tags, per-step loss, and per-step time.

First Llama run: forward was served (25 kernel nodes, 33 extern sites), backward was rejected:
`triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_5 argument in_out_ptr0 is buf15, which no
allocation owns`. In the backward wrapper there is `buf15 = reinterpret_tensor(mm_26, ...); del mm_26  # reuse` --
Inductor renames the storage of a saved activation (the input `mm_26`) into a buffer and uses it in place, and alias resolution only recognized
sources of the form `buf\d+ = buf\d+`, not input names. Relaxed it to any identifier; along the way closed a hole: if an output is a `reinterpret_tensor` view of an input (with its own
geometry) it cannot be treated as "input passed straight back", and is still rejected.

After that, both models: **0 recordings, per-step loss matches DG off** (Llama final step 7.0208 = 7.0208; GPT-2 per-step relative diff
3e-6, baseline diff between two DG-off runs 1.2e-6). Llama's backward has 36 kernel nodes and 62 extern sites;
the topology combinations split into 5 execs by seqlen; forward into 4.

### 18.2 But 1.3 s per step: the cost of harvest

Second-pass per-step median 1292 ms (DG off is 165 ms, including upstream's recording in the second pass). cProfile: 1331 harvest captures,
`torch.cuda.graph.__enter__` cumulative 18.3 s, 13.7 ms each -- this context manager does
`torch.cuda.synchronize()` + `torch.cuda.empty_cache()` (+ `_host_emptyCache`) before every capture; the backward of one seqlen
with 62 sites is 0.85 s, and empty_cache makes every later allocation cudaMalloc again. Another one: `_same_values` takes
188 ms each time -- backward returns about 60 gradients, and each `torch.equal` synchronizes once.

Fix: harvest calls `capture_begin/capture_end` directly (warmup and capture are both on the same side stream, so they are already ordered),
removing the two per-site `synchronize` calls; `_same_values` computes one device-side flag per output, `stack`s them, then reads once.
After the fix, Llama second-pass per-step median **1292 -> 35 ms** (DG off 281 ms, including upstream's second-pass recording), per-step loss relative diff 4.7e-7,
recordings 14 -> 0.

Why the second pass is still harvesting: in the first pass seqlen 64->128->256 hit a new maximum three times, and each arena growth invalidates the shapes harvested before it
(child graphs record absolute addresses), so when those shapes come again in the second pass they are re-harvested. This is the cost of `dynagraph_grow=1.0` exact growth, and it only happens
in the phase where "the largest total seen so far" is still growing; from the third pass on it is gone.

### 18.3 User-written Triton kernels: no PrecomputedGrid needed

What `_user_triton_wrapper.py` shows: Inductor has already computed the grid lambda into a wrapper expression passed as a launch argument
(`scale_kernel_1.run(buf3, buf2, s27*s77, 0.5, (255 + s27*s77) // 256, 1, 1, ...)`, `grid_type: FixedGrid`),
and the kernel table already recognized FixedGrid. It was previously rejected at `no-kernels` because of argument alignment: constexprs the user kernel declares itself
(`BLOCK`, `inductor_meta['declared_constexpr_names']`) do not appear at the call site, while scalars that Inductor specializes itself do,
so call_order counting one extra misaligns the whole `.run`. Dropping them by `declared_constexpr_names` fixes it. User kernels go through Triton's own
launcher and have no device handle (forcing device gives a `handle-mismatch` fallback), so auto picks the host path. `probe_user_triton.py`
(RMSNorm + elementwise scale, one grid a tuple and one a lambda): host path, one graph, 0 recordings, 1.6e-7.

An upstream behavior: a user kernel's grid lambda makes Dynamo specialize the first compile on the first L (same with `dynamic=True`);
upstream records that static graph once and only recompiles to symbolic shapes at the second L; a `mark_dynamic` in the probe makes it go away.

### 18.4 Steady state 4x slower than upstream: harvest again, this time caused by layout

Per-step median of each of the three passes (GPU 5, Llama): DG off 132 / 113 / **6.3** ms, DG on 116 / 33.9 / **31.5** ms -- by the third pass all shapes
have been seen, upstream replays at 6 ms, we are at 31 ms. Breakdown: forward 4.7, backward **31.1**, optimizer 3.0; runner internal 28.9 ms,
all in backward. A separate cProfile of runner.__call__ in the third pass: `_arena_views` 8 times, `_node_sig` 496 times, `torch.mm` run by the wrapper itself
928 times -- backward is re-harvesting every step.

The cause comes from dynamic layout itself: backward's static inputs (the ones upstream cudagraph_trees promises will not change address) are forward's
outputs, and forward's arena is now laid out per shape, so switching seqlen puts these tensors at a different address. The runner sees a static input move,
and the moved one is read by cuBLAS (harvested child graphs record absolute addresses), so it invalidates all harvests, and the 62 sites are redone every step.
With a fixed layout this does not happen, because addresses do not change with shape.

Fix: harvest validity is judged per key, not invalidated globally. Each harvest records the set of addresses it captured (arena base + the addresses of all inputs read by externs,
`_harvest_addr`); at call time compare against this key's record, and on mismatch drop only this key's harvest (`_drop_harvest`,
which also marks the execs holding its child graph as needing the child swapped again). Under the same shape, forward's layout is deterministic, so when the key comes again the addresses match.
Input-store growth and static-input moves no longer invalidate globally; `_invalidate_harvests` is now only used when the arena switches blocks.

After the fix, third pass: runner **28.9 -> 1.7 ms**, backward 31.1 -> 3.3, full step 31.5 -> 11.3 ms (DG off 7.8 on the same shared GPU).
Remaining gap: forward 4.7 vs 2.6, optimizer 3.4 vs 2.6, under investigation. Commit `b190f852f4`.

### 18.5 pack + varlen: how LLM training is actually written (2026-09-20 morning)

Xinwei pointed out that LLM training usually packs samples into one dimension, puts cu_seqlens on the device, and runs attention as varlen, so the shape axis is mostly padded away.
Two probes, the same Llama (transformers), a fixed stream of document lengths, packed into one token stream per step:

**flex_attention (torch-native varlen, `probe_train_varlen.py`)**, GPU 5:

| | Recordings | Per-pass per-step median (ms) | Loss diff |
|---|---|---|---|
| pad=0 (total varies with batch) DG off | 64 | 130 / 130 / 14.0 | - |
| pad=0 DG on | 0 | 369 / 27 / 11.2 | 0 |
| pad=1024 (total is fixed) DG off | 8 | 7.8 / 6.7 / 5.0 | - |
| pad=1024 DG on | 0 | 13.9 / 14.9 / 14.4 | 0 |

The first run hit two things: in flex's wrapper the grid is written as `(-1)*math.floor((-1/64)*s12)`, and the expression translator did not know `math.floor`
or true division (it used to translate `/` as `//`, a latent bug); and two regions return an input's `reinterpret_tensor` view directly as an output
(`buf137 = reinterpret_tensor(add_27, ...)`, the residual stream passed on with a different shape), which was rejected before. Now the translator is typed (int64 / double,
`math.floor/ceil` convert back to integer), and outputs that are input views are returned via `as_strided` by geometry. After that both settings have 0 recordings and bit-identical loss.

The row that matters is pad=1024: with all shapes static, upstream with 8 graphs replays at 5.0 ms/step, DynaGraph at 14.4 ms -- each region pays
about 1 ms of fixed overhead per call (input copy, address table, output views, Python), and 8 partitions make 8 ms, 3x more expensive than upstream's replay.
This is the next item on the latency axis: the per-call fixed overhead needs to drop by an order of magnitude, especially when the shape has not changed at all.

**flash-attn varlen**: the flash_attn 2.7.4 in the container is built by NVIDIA and is binary-incompatible with the in-tree torch; as Xinwei asked, I installed
the CuTe/JIT version (`flash_attn.cute`; setup in the top-level README, "Optional third-party dependencies"); FA3 is building (hopper, sm90 only). The CuTe version is correct in eager
(vs an fp32 MATH reference 7.7e-3, half a bf16 ulp; document isolation verified), and `torch.compile(dynamic=True)` works across shapes;
`probe_train_varlen_fa.py` is running. On the SDPA side: PyTorch's `F.scaled_dot_product_attention` has no cu_seqlens interface,
so varlen can only go through nested (jagged) tensors dispatched to the flash backend; the cuDNN library itself has a THD/ragged layout, but PyTorch's SDPA
does not expose it.

### 18.6 FA3 / CuTe varlen working: three autograd version-counter pitfalls (2026-09-20 noon)

FA3 is installed (built from the `hopper/` source; the cutlass submodule needs `git submodule update --init --depth 1 csrc/cutlass` first,
`import flash_attn_interface`). Both FA versions are hooked into Llama with transformers' `AttentionInterface.register`,
`probe_train_varlen_fa.py --attn fa3_varlen|fa4_varlen --pad 0|1024`. On the first run DynaGraph blew up everywhere on the same error:
`RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`.
Three layers of cause, peeled one at a time:

1. **The wrapper writes inputs it did not declare.** One input a partition receives is a buffer from the previous partition, and the wrapper writes into it directly
   (a kernel's `in_out_ptr`, or an extern's `out=` resolving to an input), but Inductor's `mutated_input_idxs`
   does not include it -- that lists mutation "in program semantics", not the wrapper's temporary reuse. We used to treat it as a read-only input: if static,
   read it in place, and never write it back. In the FA probe such inputs happen to be saved tensors held by autograd; the graph wrote them, the version counter did not move,
   but the values changed. Now we scan the wrapper's write targets (`written_scan`): such inputs always go through a store copy and are never read in place;
   if the partition returns one as an output, it is written back to the caller's tensor.
2. **Pass-through outputs were views of the store.** For an input returned as-is, we used to return an `as_strided` view on the store.
   A view shares its version counter with its base, so the next call copying into the store bumps the version, and the previous output autograd saved
   is now "modified by an inplace operation". Changed to return the caller's own tensor (written ones have already been written back into it).
3. **Arena outputs were all views of the same base.** Each arena output was `self.arena[base:base+n].view(dtype).as_strided(...)`,
   all views of that one arena tensor, sharing one version counter; when the caller does an inplace op on any output (the optimizer,
   `add_`, FA's `out=`), autograd rejects every other output it saved. Now each output is placed directly on the arena's storage with `set_`
   (`_over_storage`), so it is its own tensor with its own version counter; "input view" outputs are likewise placed on the
   input's storage.

After the three fixes, both FAs and both pad settings pass, per-step loss relative diff 8.7e-3 (FA backward uses atomic accumulation and is itself nondeterministic; for the baseline diff see below),
and the subset passes 8/8 in both modes. GPU 3, 4-layer Llama, 8 document streams, 3 passes:

| | Recordings | Per-pass per-step median (ms) |
|---|---|---|
| fa3 pad=0 DG off | 328 | 160 / 617 / 24 |
| fa3 pad=0 DG on | 96 | 104 / 498 / 66 |
| fa3 pad=1024 DG off | 307 | 79 / 559 / 23 |
| fa3 pad=1024 DG on | 96 | 83 / 287 / 64 |
| fa4(CuTe) pad=0 DG off | 296 | 84 / 723 / 21 |
| fa4 pad=0 DG on | 88 | 402 / 387 / 111 |
| fa4 pad=1024 DG off | 37 | 32 / 34 / 33 |
| fa4 pad=1024 DG on | 11 | 97 / 94 / 86 |

Two readings. First, upstream records an absurd amount under the FA custom ops: with pad=1024 the total token count is fixed, but FA's
`max_seqlen` is a host int that enters as a symint, and every partition of the whole graph is re-recorded by it -- 307 times, 0.56 s per step in the second pass.
DynaGraph brings that down to 96: the rest are all partitions containing only that single FA extern call (`no-kernels`, which we rejected, and upstream records per shape).
Second, and the most pressing right now: in the third pass (all shapes seen) DynaGraph is at 64-111 ms per step, upstream at 21-33 ms. This is not FA's fault;
the flex static setting in 18.5 was also 14.4 vs 5.0 -- it is the per-call fixed overhead of each region, and here there are many regions (FA splits every layer into a front half and a back half,
twenty-some across forward and backward), which amplifies it to 50 ms. That is the next thing to work on.

### 18.7 The 8.7e-3 loss diff is not noise: one region is called 4 times in a step (2026-09-20 afternoon)

I first took the loss diff in the table above as the baseline from FA backward's atomic accumulation; only after measuring did I find the baseline is 0 (two DG-off runs are bit-identical;
`torch.compile` in plain mode without cudagraphs is also bit-identical to upstream cudagraph_trees), while with DynaGraph on the
forward loss differs by 1e-3 from step 1 onward. Verifying every key against eager (`VERIFY_SHAPES=100`) found not a single mismatch --
each region's output is correct at the moment it returns, and gets modified by someone afterwards.

I counted each runner's calls per step (`_count_calls.py`): 6 regions are each called 4 times per step. The FA custom op cuts
Dynamo's graph, so the decoder layer's first half, second half, and MLP are each compiled into one frame, and 4 layers call each one 4 times. Layer 2's call
overwrites the same arena, and layer 1's outputs (activations autograd holds for backward, residuals used across layers in forward) are gone.
Upstream cudagraph_trees solves exactly this with the "tree": within one step, if the same function is called again while its previous outputs are still alive, it records a new node
and uses new memory -- which is also one reason it records 296-328 times (once per layer x per shape x per region).

DynaGraph's approach: **lanes**. The i-th call of a region within one step uses the i-th arena (allocated on demand, capped by `dynagraph_max_lanes`;
past the cap, that call is handed to upstream to record). The host path already compares `arena + slot_off` and patches pointers on every call, so switching lanes comes for free;
on the device path `ARENA` in ctx is swapped and the planner recomputes. The "step" boundary copies upstream's `can_start_new_generation`:
a new top-level torch.compile call starts a new step, unless some forward is waiting on its backward, or the user calls `cudagraph_mark_step_begin` --
whether a region is forward / backward / inference is passed to the runner through `cudagraphify`'s kwargs.

The same issue takes another form on extern children: a child graph bakes in the arena address and the addresses of the inputs the extern reads; we used to
store one harvest per shape key and drop it and re-harvest when the addresses did not match -- with 4 layers taking turns, every call's addresses "did not match", so every call re-harvested,
and that is where the 64-111 ms steady state in the 18.6 table came from. Now harvests are stored by `(shape, lane, addresses of inputs the extern reads)`,
and in steady state every layer hits, with zero harvests. Output tensors are no longer cached per shape either (the arena is this call's lane, extern outputs are this call's
harvest); geometry is cached per shape, and each call `set_`s a fresh tensor.

Results (fa3 pad=0, GPU 3): loss bit-identical across the three modes; per-pass per-step median plain 33/27/21, upstream 94/436/17,
DynaGraph 88/110/23.5 ms (runner internal 5.4 ms/step, 29 calls). Where upstream re-records at 0.44 s/step in the second pass we are at 0.11 s;
the third pass is still 7 ms slower -- backward 12.1 vs 6.2, and these 6 ms are the next thing to profile.

### 18.8 Extern-only regions are served too: FA probe 328 -> 0 recordings (2026-09-20 afternoon)

The remaining 88-96 recordings above were all partitions rejected with `no-kernels` -- the single FA custom op in a partition of its own, with no Triton kernel.
The reason for rejecting them is long gone: on the child route an extern site is just a child node, and the main graph can have no kernel nodes at all
(the host template already has `Node nodes[NK > 0 ? NK : 1]`). `unusable_reason` now rejects only when there is "no kernel and no extern site",
so an FA partition harvests one child per `(shape, lane, address)`, with one main graph. Upstream records this kind of partition once per shape per layer.

fa3 pad=0 (GPU 3, plain / upstream / DynaGraph): recordings 0 / 328 / 0; per-pass per-step median 35 / 29.4 / 24.6 ms,
second pass 37 / 786 / 28.8 ms; loss bit-identical across all three; the runner makes 41 calls per step totaling 6.9 ms.
In the third pass DynaGraph is already faster than upstream (the tree in upstream cudagraph_trees has its own per-call overhead too),
but the 6.9 ms of runner time is still why the GPU cannot stay busy; profiling in the next section.

Also confirmed along the way that the 1.8e-6 loss diff on the flex probe is the baseline from flex backward's atomic accumulation (two DG-off runs 1.77e-6,
plain vs DG on 1.56e-6), not ours. The self-check at the end of build, `_replays_match`, goes through `__call__`,
which used to occupy this step's lane 0, so the first real call landed on lane 1 and harvested one extra child (caught by `probe_extern_child` criterion 4);
restoring the counter around the self-check fixes it.

### 18.9 Two GPUs: DDP and FSDP2 (2026-09-20 afternoon, GPU 3+5, `probe_train_ddp.py`)

The same 4-layer Llama, `torchrun --nproc_per_node=2`, the two ranks' seqlen streams staggered, forward + backward + AdamW,
3 passes. Criteria: for each rank, per-step loss matches between DG off and on, recording counts, per-step time; the baseline is measured by running DG off twice.

| | Recordings (rank0 / rank1) | Per-pass per-step median (ms) | Loss diff | Baseline |
|---|---|---|---|---|
| DDP upstream | 28 / 28 | 130 / 393 / 13.5 | - | 3.6e-6 |
| DDP DynaGraph | 0 / 0 | 1463 / 14.4 / 14.6 | 3.6e-6 / 4.2e-6 | |
| FSDP2 upstream | 206 / 122 | 133 / 326 / 246 | - | - |
| FSDP2 DynaGraph | 0 / 0 | 295 / 131 / 39.3 | 0 / 0 | |

DDP: all-reduce happens in autograd hooks, outside the regions; Dynamo's DDPOptimizer splits the graph per DDP bucket, so there are many small regions;
steady state is even with upstream (13.5 vs 14.6), and the diff is within the baseline (SDPA backward is nondeterministic). The 1.46 s/step in the first pass is the cost of build
(nvcc compiles one host library per region), profiled in the next section.

FSDP2: this torch version traces FSDP2's hooks into the graph (`skip_fsdp_hooks` is gone), so all-gather /
reduce-scatter are NCCL sites inside regions and take the child route; both ranks have 0 fallbacks and bit-identical loss. Upstream here
recorded 206 / 122 times and was still recording in the third pass (246 ms/step) -- the unsharded parameters coming out of all-gather change address every step,
and cudagraph_trees treats that as "a static input moved" and re-records; DynaGraph patches the addresses as per-call inputs, 39 ms/step in the third pass,
6x faster than upstream. This is the most convincing table so far: real training code, two GPUs, bit-exact, and upstream never settled at all.

### 18.10 Where the first-pass cost actually goes (2026-09-20 evening, GPU 3)

Item 3 on Xinwei's list was "the first pass when there are many extern sites and many shapes". Measure first: fa3 pad=0, one pass only, DG on,
timing each `capture_begin` / `capture_end` / `_harvest` individually (`_time_capture.py`, without cProfile --
cProfile charges the hook overhead of C calls to `capture_begin` and reports 3 ms per call, when it is actually 13 us):

| | Count | Total | Median | Slowest |
|---|---|---|---|---|
| capture_begin | 749 | 12 ms | 13 us | 0.2 ms |
| capture_end | 749 | 7 ms | 7 us | 0.2 ms |
| One harvest (one region, one shape, all sites) | 240 | 756 ms | 1.5 ms | 153 ms (the first one: pool, cuBLAS init) |

An isolated microbenchmark (`_bench_capture.py`) agrees: begin 10 us, captured mm 32 us, end 6 us; without a shared pool
the mm takes 0.5 ms (each capture opens a new chunk of device memory). So harvesting itself on the child route is already on the order of ~50 us per site,
1.5 ms per new shape per region; even a few hundred GEMM sites come to only a dozen or so ms per new shape.

Of the first pass's 147 ms per step (plain 35), harvesting is only ~45 ms. The rest is eager verification for each new shape (`dynagraph_verify_shapes=3`:
each region runs eager once on each of its first three shapes as a reference) and building ctx / host params on the first call. The first step also pays for build:
g++ compiles one host library per region, ~160 ms (stored by source hash in Inductor's cache directory, so it is compiled only once per machine;
the probes all have `force_disable_caches` on, so they compile every time), and `_pick_update` still uses the
`torch.cuda.graph` context (with synchronize + empty_cache) to time regions. Both are one-time costs, not on the "long tail of shapes" path.

A pitfall noted along the way: the pool passed to `capture_begin(pool=...)` drops to use_count zero after the last graph is freed, and capturing again with the same pool handle hits
an internal assertion in `CUDACachingAllocator.cpp` (reproduced in the microbenchmark with `del hold`). DynaGraph did not hit it because `site_holds`
always holds the first graph of each topology; `_invalidate_harvests` clearing `child_holds` does not reach it.

## 19. symint tier 2: the value stays in device memory, the planner reads it inside the graph (2026-09-20 late night)

### 19.1 I went in the wrong direction first

In the first version I took the route "Inductor splits the graph at `.item()`, and `u1` enters the next segment as a host int", fixed two bugs in upstream partitioning
(`DynamicSliceSize` does not bind its NoneLayout buffer name; the partition signature only filters NoneLayout buffers on the output side),
and then declared tier 2 unnecessary. Xinwei pointed out that this is not tier 2: tier 2 is where the symint's value is produced by a GPU kernel and stays in device memory, and the
planner, as a kernel node in the graph, reads it and patches the nodes after it; branching means the on-device value directly decides which part of the graph runs next.
The two upstream fixes stay (`6efb901669`; they were bugs anyway), and what follows is the real tier 2.

### 19.2 Design

`dynagraph_unbacked="device"` (default `"host"` = the original graph splitting):

1. **Inductor does not split.** `should_partition` does not split at `DynamicScalar` (`.item()`), `DynamicSliceSize` (`x[:n]`'s
   `u1 = max(0, u1_end - u1_start)`), `AssertScalar` (`if not (u1 <= s77): raise`), or at nodes that use these symbols,
   so the `u0 = buf0.item()` line in the wrapper stays in the partition. This is only done for symbols "whose every use is on the GPU"
   (`_device_resolved_unbacked`): if the symbol is returned by the graph (a bool `.item()` as a SymBool output), or used by a CPU op /
   DeviceCopy / a custom op that cannot go into the graph, it is split as before -- the host really needs that value. The partition signature removes the
   unbacked symbols the partition defines itself from the inputs, and adds the backed symbols that appear in asserts to the inputs (`s77`). In the wrapper, each unbacked symbol
   is followed by a line `# unbacked u1 in [0, 64]` (`var_to_range`).
2. **Upper bound.** DynaGraph reads the wrapper: the `.item()` lines, the derived lines (`u1_start = ...`, `u1_end = ... if u0 >= 0 else ...`,
   `u1 = max(...)`; the expression translator gained ternaries and comparisons), the range comments, and the `if not (u <= E): raise` check lines. Each symbol's
   upper bound = the `min` of the range upper bound and all `u <= E` (`u1` -> `min(64, s77)`, evaluated on the backed shapes). A symbol used in sizes
   without an upper bound is rejected (`unbacked-unbounded`). **Slots are laid out by the upper bound**, and during harvest and capture the wrapper gets the upper bound:
   the wrapper source is rewritten to `u0 = __dg_item("u0", buf0)`, `u1 = __dg_item("u1", None)`, the callback returns the upper bound,
   and nothing synchronizes inside capture. Externs (addmm) run on the upper-bound shape and children are stored by `(shape, lane, address)` -- pad-to-max;
   only externs whose output size contains u are allowed (row-independent); those that reduce over u or consume u-sized integer buffers (indices) are rejected.
3. **The planner is split into segments.** The main planner still handles pointers and grids / scalars that contain only backed symbols; it does not touch nodes containing u
   (`static_grid = true`). Each `.item()` gets a `dynagraph_planner_u<j>` kernel, which during capture is launched onto the current stream from the callback,
   so in the graph it sits right after the kernel that writes `buf0`: thread 0 reads the value from
   `ARENA + SLOT_OFF(slot)`, computes the other symbols from the derived lines and writes them into ctx (compares with last time; if nothing changed, the whole segment exits early),
   then one block sweeps over the nodes it manages and patches their grids (0 -> disable the node) and scalar parameters. In ctx, u's slot is first written with the upper bound by setctx
   (for the layout), and planner_u then overwrites it with the real value.
4. **Read once at exit.** Only when an output's size contains u: after launch, `cuMemcpyDtoHAsync` the ctx symbol block into pinned
   memory, `cuStreamSynchronize`, and `set_` the outputs by the real values (geometry cached by `(shape, value)`). The first version used
   `ctx[idx].tolist()` (a gather kernel + a sync) and was 80 us slower than upstream; switching to the pinned copy made it win.
   For regions whose outputs do not depend on u (e.g. `x[:n].sum(0)`), the host never learns n from start to finish, and there is no sync at all.

Regions of this kind that DynaGraph rejects cannot be handed to upstream to record (during recording `.item()` would synchronize inside the capture), so `deferred_cudagraphify`
runs them eagerly directly (`device_scalar_region`).

### 19.3 Results (`probe_unbacked_device.py`, GPU 6 exclusive, L=200, us / call, single step with sync / back-to-back)

| case | Upstream (graph split) | DynaGraph device path | plain compile |
|---|---|---|---|
| item->slice->pointwise+sum, output size = count (read once at exit) | 138 / 116 | 96 / 79 | 69 / 58 |
| item->slice->row-wise reduction, fixed-length output (no sync) | 136 / 113 | 75 / 57 | 70 / 59 |
| item->slice->Linear (extern at the upper bound)->sum | 136 / 114 | 109 / 89 | 128 / 133 |
| bool `.item()` as a scalar (the graph returns it: stays split on the host) | 167 / 148 | 139 / 118 | 95 / 83 |
| two `.item()`s, two planner_u segments | 186 / 162 | 72 / 55 | 99 / 88 |

All five cases: 0 recordings, no fallbacks, bit-identical with plain compile across two passes of the shape stream (the extern one 1e-7). The two cases that need no exit read
are even faster than plain compile -- the GPU never waits for the host. Upstream turns every `.item()` into "replay half a graph, sync,
replay the other half", and has to re-record the second half for every distinct count value.

### 19.4 Branches: in the graph too (2026-09-21 early morning)

`torch.cond` looks like this in Inductor: partition_0 computes the predicate -> top-level `buf2_selector = int(buf1.item())`,
a Python `if` calls `true_graph_0` / `false_graph_0` (two subgraph functions that do not go into a cudagraph) -> partition_1.
The device version is what Xinwei described: the predicate stays in device memory, the two subgraphs are captured as the two bodies of one conditional node, and the handle is set by planner_u.

**Two changes on the Inductor side.** `cond`'s lowering used to call `disable_cudagraphs_reason` as soon as it saw a cond, so the whole compile did not
record graphs -- in device mode it no longer disables them; `should_partition` used to split a partition at every `ir.Switch`; now, in device mode,
as long as it is a cond, does not bind new unbacked symbols itself, and its selector is a tensor (not a `ShapeAsConstantBuffer`),
it stays in the partition, and the selector's `.item()` goes into the wrapper together with the `if` block.

**DynaGraph side.** `_cond_blocks` parses `buf2_selector = int(buf1.item())` plus the following `if/elif` block into a
`_Cond` (the selector name, each branch's function name and arguments, the output renaming `bufN = buf2[i]`); each branch is a site,
named `cond:buf2:<b>`, and site order follows the position of the selector line in the body, interleaved with the extern sites.
`_fold_branches` folds the `empty_strided_cuda` calls in the two subgraph function bodies into `alloc_body` and renames the outputs to the wrapper's own
names, so both branches get the same set of slots and lifetime resolution works as before. The rewritten wrapper has, after the `__dg_item` line, an extra line
`buf2 = __dg_cond(site0, (lambda: false_graph_0([...]), lambda: true_graph_0([...])))`, and the original `if` block
is skipped. During harvest each branch first runs a warmup with the module's own allocator (kernel compilation and autotuning all happen in this step),
then is captured on arena views into its own small graph -- the warmup that harvest originally did for extern sites cannot be done for branches, because it would consume the views reserved for the
capture; at capture time, branch 0's site builds a SWITCH conditional node with N bodies, and each branch's subgraph
is placed into its own body as a child graph node; the selector's planner_u, after reading `buf1`, adds one more statement:
`cudaGraphSetConditional(handle, selector)`. The host reads no value anywhere along the path.

**Not supported.** Kernel nodes inside a body are not in the main graph's handle table, so the planner cannot patch their grids; therefore branches whose arguments or allocations
contain `u` symbols are rejected (`cond-unbacked`), branches that call externs are rejected (`cond-branch`), and the region runs eagerly.
A cond whose output itself binds new unbacked symbols (`item_then_cond`: slice first, then cond) is still kept outside the partition by the scheduler,
and the following partition receives elements of a tuple input, `unmodelled` -- moreover, upstream plain compile cannot even compile this case itself
(the branch subgraph generates the assertion `u1 <= s77` but does not pass s77 in, NameError), so it is not our boundary.

**Results** (`probe_unbacked_device.py --time`, GPU 7 exclusive throughout, process count checked once before and once after the run,
three rounds with configurations interleaved, median taken; L=200, us / call, single step with sync / back-to-back):

| case | Upstream (cond disables cudagraph) | DynaGraph | plain compile |
|---|---|---|---|
| predicate -> cond (two single-kernel branches) -> sum | 93 / 78 | 98 / 74 | 103 / 89 |
| branches each return two tensors -> multiply -> sum | 105 / 91 | 101 / 80 | 114 / 105 |
| one kernel on one side, reduction+pointwise on the other | 98 / 83 | 99 / 73 | 96 / 80 |
| cond -> count -> slice -> sum (tier 1 after the branch) | 142 / 120 | 97 / 73 | 125 / 115 |

All four cases: 0 recordings, no fallbacks, both branches taken in the two passes of the shape stream (the probe prints the true/false counts), and bit-identical with plain compile.
On the upstream side, once a cond appears the whole compile does not record graphs, so the "upstream" column is just non-recorded eager partitions.
In the back-to-back column DynaGraph is fastest in every case; in the single-step column it leads on the two heavier cases and ties on the two lighter ones --
the regions are too small, and the overhead of one sync swamps the difference.

(The first measurement, on GPU 3, had someone else start a job during the third round; the whole batch was discarded per the rule, and this table is the re-measurement.)

Commit `e3763ad0c9` (`fork/dynagraph`, local `0173ad6685`): the device-mode rules for `cond` lowering and `should_partition`;
in dynagraph.py `_Cond` / `_fold_branches` / `_sites_in_order` / `_dev_cond` / SWITCH capture / planner_u's
`cudaGraphSetConditional`; `generate_pointer_patches` moved into the `unmodelled` fallback path. Regressions in both update modes 36/36 each;
upstream `test_cudagraph_trees.py -k graph_partition` in a clean worktree: 57 passed, 2 skipped; on the main tree,
`test_graph_partition_with_memory_plan_reuse` is flaky on a busy GPU -- with DynaGraph off, HEAD's own files fail the same way
(line 14 of the output is entirely stale, a race in that upstream path), so it is not from this change.

### 19.5 User Triton kernels also go onto the device path (2026-09-21 early morning)

Two decisions by Xinwei: opaque externs inside branches (the cuBLAS kind) are not supported, `cond-branch` keeps rejecting them and the region is cut and runs eagerly;
such conds are rare, so no effort goes there; as for user `@triton.jit` kernels not having handles, "just patch the launcher".

The handle does not come with the kernel; it is returned by the driver during capture when the kernel is launched with the `DEVICE_UPDATABLE_KERNEL_NODE` attribute.
Inductor's own kernels go through our modified static launcher, so they have one; user kernels go through Triton's own launcher, so they do not.
Upstream already has the switch `static_launch_user_defined_triton_kernels` (off by default, "not yet supported"), and turning it on makes user
kernels go through the static launcher too. Fix: when DynaGraph is on, a user kernel's `inductor_meta` gets an extra `static_launch` entry,
and `can_statically_launch` looks at that entry instead of the process-global config -- it is written into meta because the kernel source is the key for all in-process caches:
the same source compiled once with DynaGraph off (Triton launcher, no handle) would be reused directly when compiled again with it on, and the probe
ran into exactly that when running the three modes plain -> off -> on in one process. The `user` flag in the kernel table now means "a user kernel that did not go through the static
launcher", and only those force the host path.

Validation: the new case `user_kernel_item` in `probe_unbacked_device.py` (user kernel + `.item()` in the same region) is served on the device path,
bit-identical; in `probe_user_triton.py`, forced device was changed from "falls back as expected" to "must be served", and it passes too.

### 19.6 Clearing out three cheap ones (2026-09-21 early morning)

After reading the open-problems list, Xinwei said "clean these up first, none of them look like substantive problems". Three items, plus a pre-existing bug run into along the way.

**PrecomputedGrid (autotuned user kernels).** The blocker was a single line: in `extract_kernel_table`,
`n_grid = 3 if grid_type == "FixedGrid" else 0`, while PrecomputedGrid likewise passes the grid symbols as trailing arguments
(`extra_launcher_args`), so the signature and the call site did not line up and the whole table was abandoned. Changed it to `n_grid = len(extra_launcher_args)`
(FixedGrid sets three itself, so its behavior is unchanged), and stored `precomputed_grids` and "symbol -> call-site expression" into `k["pgrid"]`;
after warmup the autotuner settles on a config, then upstream's `GridExpr.from_meta` picks that entry, `_launcher_sN` is
replaced with the call-site expression (it needs parentheses, otherwise putting `a*b` into `(127 + X) // 128` changes the result), and the result fills `k["grid"]`.
Both codegens take the existing `if k["grid"]:` branch, and `_GRID_AXES` did not change at all. New probe `probe_autotuned_user.py`.

**The pre-existing bug run into along the way.** In the "grid is constant" branch the planner did a bare `continue`, which also skipped the symbolic scalar patches after it --
a fixed grid does not mean fixed scalars. Persistent kernels (a constant number of programs, `n` varying with shape) were therefore frozen on the device path at the recording-time `n`;
the self-check flagged `runtime-mismatch` and retired them (no wrong numbers, but lost coverage); the host path does not have this branch, so it was always correct. Changed it to not `continue`.
New probe `probe_const_grid_scalar.py`: before the fix the device path failed and the host path passed; after the fix both pass.

**Two-way branch self-check.** The two bodies of the conditional node are wired at capture time, the selector is set on the device, and data only flows through one of them,
so wiring the two bodies backwards would still pass the ordinary self-check. A force slot is added per cond at the end of ctx (0 means not forced, b+1 means pinned to branch b);
planner_u reads it, overrides the symbol value, then sets the handle; the reference is the same wrapper run eagerly with its selector forced to the same branch -- it must be the
kernels the wrapper itself generated, because a hand-written equivalent has a different reduction order and will not match bit for bit. A mismatch gets the new tag `cond-branch-mismatch`.
This check has to be checked too: `probe_cond_selfcheck.py` runs three times: the clean run must not raise a false alarm, inverting the forcing semantics on the graph side must be caught,
and removing the forcing on the eager reference side must also be caught (otherwise both sides fail together and "all pass" means nothing).

**Parse failures state the reason.** The five `return None, None` in `extract_kernel_table` all swallowed the reason, and what finally got printed was
`no-kernels` (and when the region has extern sites it becomes `no-symbols`, which is even more misleading). Changed them to `raise Unsupported(...)`
carrying the kernel name and what failed to line up; the constructor catches it and stores it as `parse_problem`, and `unusable_reason` returns it before the counting checks.
On the logging side, the message is also split on `"tag: detail"` before printing, so the applicability scan, which counts by tag, is not fragmented by details that carry kernel names.
Now it prints `[unparsed]: triton_red_fused_mul_relu_sum_0: no parameter offset for [...]`.

Commit `2364972e6f` (`fork/dynagraph`, local `64a6b581cf`). Regressions in both update modes 39/39 each (three new probes).
The two upstream suites run in a clean worktree: `graph_partition` 56 passed, 1 failed; `test_static_triton_launcher` 38 passed, 1 failed --
these two failures are identical on a clean HEAD on the same GPU in the same period, so they are not from this change; the first one is also load-sensitive and passes on an idle GPU.

## 20. Capture branches directly into the body and get their handles (2026-09-21 morning)

The item left open in 19.4: "kernel nodes inside a body are not in the main graph's handle table, so the planner cannot patch them". First a standalone script
(`dynagraph/_scout_body_capture.py`, GPU 7 exclusive) settled the key question: while the parent graph is being captured, can `cuStreamBeginCaptureToGraph`
capture kernels directly into a SWITCH body graph, and do the handles come back? The answer is yes:
four handles come back from the same sink in launch order (main-graph and body handles interleaved, no separate call needed),
the in-graph planner patched the body nodes' grid, scalars and pointers in the same launch, all return codes were 0, and the conditional still selected correctly.

So the change is plumbing:

**The kernel table is now parsed from the folded body.** `_fold_branches` already folded the wrappers of both branches into `alloc_body`
for allocation and lifetimes; now `extract_kernel_table` reads it too, so branch kernels become ordinary rows in the table. Folding must
keep the selector's `.item()` line (previously it was deleted along with everything else), otherwise the planner's addressing by "which launch number" goes out of sync.

**At capture time, capture directly into the body.** The cond branch of `on_extern` no longer attaches the small harvested graph as a child graph;
instead it opens a side stream, does `cudaStreamBeginCaptureToGraph` into body b, runs the branch once, and ends the capture. Two pitfalls:
the side stream must not `wait_stream` the main stream (that pulls the side stream into the parent graph's capture, and next comes CUDA 401),
and `cudaStreamUpdateCaptureDependencies` must wait until all bodies are filled (copying the order validated in the script).

**Warmup must be per branch.** Once branch kernels are in the table, the autotuner of the branch that never ran has never settled,
`settled_blocks` returns None, and the whole region is rejected with `unsettled-config`. Using the "forced selector"
eager model from the branch self-check in 19.6, each branch is warmed three times.

**Gain.** Pure cond regions no longer re-harvest per shape: when `child_sites` (sites other than cond) is empty, new shapes reuse
the same exec directly, and `_swap_children` is skipped too. For cond_pointwise over two passes of the shape stream, harvests drop from 8 to 2 (only the one per compile
build remains); capture is still 2.

**Two other things fixed along the way.**
1. Upstream bug: a cond branch subgraph generates assertions like `u1 <= s77` without passing s77 in, so plain compile hits
   NameError -- both `item_then_cond` and the newly added `cond_over_slice` fail to compile. Changed to: inside a real cond subgraph
   (not a graph partition; the two share the same wrapper class and are told apart by `partition_signatures`), an assertion that mentions a
   backed symbol the subgraph did not receive is not generated; the outer graph asserts the same thing anyway. The first version did not distinguish partitions and also swallowed the upper-bound assertions inside partitions,
   and DG takes u's upper bound precisely from those assertions, so `cond_then_item` immediately hit `unbacked-unbounded` --
   caught by the regression suite.
2. A line like `u1 = u1` means "the symbol is passed in as an argument from the previous partition", not an assignment; it is now treated as an ordinary symbol (previously it reported
   `unbacked-order`).

**Two not wired up yet** (marked in the probe as "must reject cleanly"; numerics must still be correct): the second partition of `item_then_cond` receives
elements of a tuple input (`unmodelled`); the selector of `cond_over_slice` reads a tensor passed in from the previous partition, which is not in the arena
(`unbacked-source`). The latter is not hard to wire up -- the planner has the input addresses at hand -- but that is for the next round.

## 21. Extern calls reducing over an unbacked dimension: zero the padding (2026-09-21 morning)

On the device path an extern call can only run at u's upper bound (cuBLAS picks and launches its own kernel; the planner cannot reach it).
When the output is sliced by rows along u this is fine, the extra rows are simply dropped; but once u is the reduced dimension (gemm's K, bmm's sequence length),
garbage left in `[u, bound)` by the previous tenant gets added in, so this whole class used to be rejected outright with `unbacked-extern-reduce`.

Zero is the identity of addition: zero that range before the extern call, and the result computed at the upper bound **equals** the result computed at the true value; it is not an approximation.
The implementation is a generated kernel `dynagraph_zeropad<i>` whose only parameter is ctx; the slot, element type, span expression and upper bound
are all baked into the source (same approach as `planner_u`); fixed grid plus a grid-stride loop, so this node itself never needs to be patched,
and it also must not go through the static launcher -- if it took a device handle, the handle count would no longer match the kernel table. At capture time it is
`_launch`ed before `cudaStreamGetCaptureInfo`, and the driver naturally turns it into a dependency of the extern child graph node.
Replacing `u` in the span with a local variable gives the upper-bound end; `lo`/`hi` are both numbers already in ctx (the layout is laid out by the upper bound,
and planner_u has already written the true u into ctx, so both numbers are at hand).

**Guards** (if any one fails, the whole site stays rejected): the op must be in the additive-reduction whitelist (mm/addmm/bmm/baddbmm;
for max and the like, zero is not the identity); the operand must be our own buffer in the arena, not a graph input (we cannot write someone else's memory);
u must be the first dimension and must not appear in the strides or in any other dimension; the view must start at 0 and cover the whole block.
Every operand carrying u must be zeroed -- zero only one side and a NaN on the other side turns everything into NaN after one multiply.

**Not bitwise identical**: cuBLAS may pick a different split-K at K=bound than at K=u, so `exact` is relaxed for regions with padding.

**Validation** (`probe_unbacked_reduce.py`, added to the regression list): `varlen_gram` (both operands of size u, u is K)
and `varlen_addmm` must be served and match eager (1.9e-07 / 2.0e-07, 0 recordings); `varlen_gram_input`
(operand is a graph input) must stay rejected; `varlen_bmm_stride` may be served or rejected, but its numerics must be correct.
Plus a negative test `--sabotage`: make the loop bound of the generated zeroing kernel empty, and the self-check of the two must-serve cases
must catch it (`selfcheck-mismatch`) -- if it is not caught, the zeroing does not affect the result at all and the feature is fake. In the run it was indeed caught.
Regression under both update modes: 36/36 each. Commit `c0c4377d0d` (`fork/dynagraph`, local `9d5eaa0dd5`). Upstream
`test_static_triton_launcher.py`: 38 pass, 1 fail (`test_device_tma_gemm_falls_back_from_fast_launcher`);
that one also fails on a clean HEAD, it is an environment issue.

Commit `b1c5b6f201` (`fork/dynagraph`): the `dynagraph_unbacked` config; three places in Inductor (the device-symbol rule in `should_partition`, partition symbol inputs,
range comments); `_DevScalars` / `_planner_u_source` / exit reads in dynagraph.py; `probe_unbacked_device.py`
added to the regression list. Upstream `test_cudagraph_trees.py -k graph_partition`: 57 pass, 2 skip.

## 22. Elastic expert parallelism: the width changes, the graph does not have to be redone (2026-09-21 afternoon)

Problem: in elastic EP the parallel width changes at runtime (in vLLM an HTTP request changes the DP width). When the width changes,
the number of experts held per GPU changes, the all_to_all buffers change, and **the communicator itself changes**. Upstream's approach is to throw all graphs away and re-capture,
or simply use `--enforce-eager`.

**First ask whether the driver allows it.** Standalone script `dynagraph/_scout_nccl_widths.py` (3 ranks, GPUs 4/6/7) measured widths 2 and 3:

- Under both widths, the child graphs captured for all_reduce and all_to_all_single are **exactly the same**: 3 nodes, 2 edges
  (WaitEvent / Kernel / EventRecord); kernel name, blockDim, shared memory, cluster and cooperative are all identical.
- Two things change: gridDim (NCCL's channel count, popcount(channelMask)), and the kernel parameter block --
  the first 8 bytes of the parameter block are the `ncclDevComm*`, **the communicator is baked into the captured kernel parameters**.
- The key question: take an exec captured at width 2, swap in the child harvested at width 3 via `cudaGraphExecChildGraphNodeSetParams`,
  and afterwards all_reduce gives the correct 3.0 and all_to_all gives the correct three-way result; swapping back to width 2 is also correct.
  A separate control showed that this call **swaps gridDim along with it**.

So elastic width can be one graph, rather than one graph per width.

**But a hole that fails silently turned up.** `_hkey` contains nothing related to the communicator (shape, lane, input addresses).
`_swap_children` short-circuits when `hkey == child_applied`. So: the group name is unchanged, per-GPU shapes are unchanged,
only the group has been rebuilt underneath at a different width -- harvest hits the old cache, the swap is skipped, the graph keeps running with the **old communicator**,
no error, and the result is the old width's answer. The script measured this silent case (replaying the width-2
exec in a 3-rank world returned 2.0 without a word).

**Fix: let the ops declare it themselves; the compiler knows no op.** The first version wrote a table in dynagraph.py of
"which parameter of which op to watch". That was wrong -- specializing the compiler for a particular op has zero extensibility, and
ops like fa3 are clearly able to describe themselves, so there is no reason they should only be treated like cuBLAS.

Now there is `torch/utils/_capture_deps.py`: a neutral registry that ops write into themselves, stating
"which of my parameters identify what I baked in at capture time" plus a function that turns those parameters into values.
`torch.distributed.distributed_c10d` registers the eight functional collectives with
`_capture_identity_of_group`, which resolves a group name to `(size, id(pg))`.

DG does only three op-agnostic things: look up the registry, use the op's own schema to find which position in the call
the declared parameter occupies, and take the literal at that position in the wrapper and hand it to the op's function; the result is folded into `_hkey`.
dynagraph.py now contains not a single op name. Undeclared ops fall back to fully opaque, which is what cuBLAS gets now:
re-harvest only when the shape or an address changes. Non-distributed regions pay not a single extra byte (when all declarations are None it returns an empty tuple directly).

Verified at a real call site; the resolved result is directly visible in the harvest key:

```
key ((('s77', 512),), 0, (139801821316096, ...),
     (('_c10d_functional::all_reduce_', ('0',), (2, 139813059954096)),))
```

`('0',)` is the `group_name` argument found via the schema, and `(2, ...)` is the
world size and pg identity computed by the function the op registered. The easiest place for this path to fail silently is the schema-based parameter-position lookup returning None,
so it was checked specifically on real generated collectives, not with stubs.

**Validation** (`probe_elastic_ep.py`, 2 ranks on GPUs 6/7): one MoE layer (routing -> all_to_all dispatch ->
per-expert bmm -> all_to_all gather back), with widths taken as the factors of the world size (with 2 ranks: 1 and 2), three
token counts per width, and finally a switch back to the first width. Result: both ranks have 0 recordings, no fallback tags, and differ from eager by 1e-7.
Plus a check aimed directly at that guard: under the same group name, swap the resolved result for another group; the harvest key must change -- measured "changed".

**One honest caveat.** The probe shows `compiles 4` with 2 widths, i.e. **Dynamo compiles once per width**.
Safe (new compile = new region = new communicator), but it means "one graph across widths" is not yet delivered at the torch level --
this probe passes pg as an argument, so Dynamo specializes on the object. In a real system pg is a module attribute, the group name is fixed,
and it is rebuilt underneath; that path is what the guard above is there to save, and it is the only case this time that could not be reproduced end to end on this machine
(`_abort_process_group` hangs once graph capture has captured a communicator; the script measured that too).

The upstream path (vLLM 0.29's elastic EP) is not being done this week: `torch._inductor.cudagraph_trees` appears 0 times in the vLLM
source -- vLLM captures graphs itself, so DG has no attach point; also all 8 GPUs are fully occupied, the 31GB model is not downloaded,
/nvme2n1 has only 53G left, and NIXL is not installed, so its own tests simply skip. Details in `docs/notes/WORKLOAD.md`.

## 23. TMA: what gets passed in is not a pointer but a descriptor (2026-09-21 afternoon)

The kernel table can patch parameters in place because three things line up: `*fp32` in the Triton signature means a pointer,
`cuFuncGetParamInfo` gives the byte offset of every parameter, and call-site arguments map one-to-one onto the signature by position.
TMA descriptors break the third.

**Two kinds of TMA, only one has a problem.** The TMA that Inductor generates itself is **device-side**:
the kernel writes `tl.make_tensor_descriptor(in_ptr0, shape=[ks0], ...)`, the parameters are still
pointers plus integers, the shape is a plain `ks0`, and it was always servable. The problem is **host-side** descriptors --
a user kernel that receives a descriptor built by `TensorDescriptor.from_tensor(x, [32, 256])`,
or Inductor with `enable_host_side_tma` turned on. Then the signature contains `tensordesc<fp32[32, 256]>`
(new API) or `nvTmaDesc` (old API), neither of which starts with `*`.

**Two independent harms.**

First, alignment. One signature entry expands into several real parameters in the cubin: on this machine Triton 3.8 chose
`arg_tys='MiillMiill'`, i.e. 5 parameters per descriptor (one CUtensorMap passed by value, followed by the
block shape and strides); via the decomposed path it would be 11 (base pointer + per-dimension shape/stride + two
flags + block shape/stride). So every parameter after the descriptor has a real index several positions larger than its signature position,
and `cuFuncGetParamInfo` still returns a valid offset for the misaligned index. No error; it simply patches someone else's bytes.
On top of that, `tensordesc<...>` does not start with `*`, so it is queued for patching as an integer scalar.
Actual behavior on the probe: **no fallback tag, 0 recordings, DG reports "served by one graph", numeric difference 9.5e-02.**

Second, something more fundamental than alignment. The 128-byte CUtensorMap encodes the global address and globalDim,
i.e. the shape at the moment the descriptor was built. For one graph serving a whole shape space, the shape is the last thing that may be frozen.
Even with correct offsets, patching parameters alone is not enough -- the map itself must be re-encoded per shape.

**Step one is to recognize it and reject** (not recognizing it looks like the silent wrong-byte patching above). **Step two is to serve it**:
a kernel the table cannot read should become opaque, like cuBLAS -- capture this launch into its own child graph,
re-harvest per shape, and let Triton's own launcher re-encode the descriptor, so we need not understand a single byte of the tensormap encoding rules.
Both steps are done; step two is in 24. Regions with host-side descriptors are now **0 recordings, no fallback tags,
correct numerics**, i.e. served, not fallen back.

**Two upstream bugs dug up along the way**, both on this path; without fixing them DG would never even see a host-side descriptor:

1. Under `dynamic=True`, integers read from an outer scope are not specialized, so a plain
   `BLOCK_SIZE_X, BLOCK_SIZE_Y = 16, 32` written outside the compiled function has become a symbol by the time it reaches
   `CreateTMADescriptorStableVariable`, the signature comes out as
   `tensordesc<fp32[s58, s99]>`, and Triton compilation fails with `invalid literal for int(): 's58'`.
   The same value passed as a `tl.constexpr` is **already** specialized; the descriptor's block shape now goes through the same helper to be specialized.
   (The tracing path needs no change: Triton's own `validate_block_shape` already rejects SymInt in
   `TensorDescriptor.__post_init__`, so a descriptor object can never be born carrying a symbol.
   This was found by trying, not by reasoning -- originally `maybe_unpack_host_tma_descriptor` was also changed by symmetry,
   and that change was reverted when no case that reaches it could be constructed.)
2. `_build_fast_launcher` has long had a fallback for TMA, but it checks
   `inductor_meta["host_tma_descriptor_args"]`, a key set only by Inductor's own
   host-TMA codegen. A user-written kernel puts `tensordesc<...>` in its signature without that key,
   so the vectorcall launcher still gets built, expects 14 parameters according to the expanded `arg_tys` but receives only 6,
   and launch fails with `_FastCudaLauncher: expected 14 args, got 6`. Changed to check the static kernel's own
   `_has_tensordesc` -- the same fact, read from the place that establishes it rather than from a meta key that merely happens to accompany it.

**Probe**: `dynagraph/probes/probe_tma.py`, one process for each of three modes. `inductor` and `device` first assert that this
compile really did generate device-side descriptors (otherwise the result is "inconclusive" with exit code 2; no false passes allowed), then require the region to still have
0 recordings, no tags and correct numerics; `host` requires the region to still be served by one graph with correct numerics, **and that the log actually says
this kernel took the opaque path** -- otherwise "reject the whole region" would also satisfy the earlier criteria, and the check would be worthless.
Delete the descriptor-recognition code in DG and `host` reverts to the silent-wrong behavior above -- that is its counter-proof.

## 24. Kernels the table cannot read become opaque sites (2026-09-21 afternoon)

Previously, if even one kernel could not be read, the whole region was rejected. That floor was set wrong. **If you cannot read it, do not read it**:
capture this launch into its own child graph and re-harvest per shape, which is exactly what cuBLAS gets now --
not a single byte is patched, so not a single byte can be patched wrong, and the rest of the region is still served by one graph.

So `extract_kernel_table`, besides the table and the symbols, also returns the names of the kernels it cannot read,
and those launches become sites. Along the way, three duplicate site scans were merged into one `_scan_line`:
the one that names extern calls, the one that pairs each site with the buffer it writes, and the one that reads each site's declarations
were the same decoding copied three times -- and this change is exactly where they would drift apart.
`_sites_in_order` now carries the argument text along, so the declaration lookup no longer rescans the body itself
(that second scan did not know about cond branches, and when a region had both branches and collectives it would match declarations to the wrong site).

**A Triton launch differs from an extern call in two ways.**

First, it **passes the stream to launch on as an argument**, rather than following the current stream. Once harvest switches to its own stream,
that argument must be rewritten, otherwise the work lands on the wrapper's stream and runs outside the capture. `_on_current_stream`
does this at the place where harvest actually runs each site.

Second, it does not say what it writes. Extern calls declare outputs with `out=`, and kernels in the table declare them through parameter names
given by Inductor; an opaque kernel has neither, so every graph input that appears on its call line
goes into `written_scan`. Copying one extra read-only input costs one copy; missing one input that is written means
the tensor handed back to the caller has been quietly modified by the region.

**A known gap (safe, only a coverage issue)**: if an opaque kernel appears only in the part folded in from a cond branch,
its launch line is not in `body` and cannot become a site; during the main capture it still launches
and still hands over a handle, so the handle count does not match the table, and `handle-mismatch` rejects the whole graph.
It is a rejection, not a wrong result, but that region is lost for nothing.

**The motivation is host-side TMA descriptors** (see 23): the table cannot read such kernels at all; with this path
they go from "rejected" to "served", the descriptor is re-encoded by Triton's own launcher on every harvest,
and we need not understand a single byte of the tensormap rules.

## 25. Can TMA be "patched" instead of re-harvesting the whole segment? (2026-09-21 evening)

24 demoted unreadable kernels to opaque sites, at the cost of one child swap per shape. The question is:
the descriptor is already a kernel parameter (128 bytes passed by value), so can it be patched directly like any other parameter?
The host path certainly works (re-encode with `cuTensorMapEncodeTiled`). **Whether the device path works is the real unknown**,
and the whole point of the device path is that replay does not touch the host.

Measured with a standalone script, `dynagraph/_scout_tma_device.cu` (GPU 7 exclusive, H100 / CUDA 13.1).
Six levels, so that if one fails you know where it breaks:

| Level | What it asks | Result |
|---|---|---|
| 1 | Host-encoded descriptor passed by value: is the rig itself correct | ok |
| 2 | Device modifies a descriptor in global memory, a later kernel reads it from global | ok |
| 3 | The same bytes passed by value as `__grid_constant__` | ok |
| 4a | In-graph planner node patches a 4-byte scalar of a later node | ok |
| 4b | Same, patching 128 bytes (host-encoded) | ok |
| 4c | Same, split into 8 writes of 16 bytes | ok |
| 4d | **Same, patching 128 bytes that the device rewrote itself** | ok |
| 4e | Shrink global_dim, read out-of-bounds rows, should get TMA's zero fill | ok |
| 4f | Sabotage test: the planner does nothing | **FAILED as expected** |

4e is the key one: a changed address is not enough, what the descriptor freezes is the **shape**. Reading out-of-bounds rows returns all zeros, which shows the new global_dim
really was honored by the hardware. 4f proves that 4d's pass was not free -- if the planner does nothing, the consumer still reads the original map.

**Result: the device path works.** The recipe is:
use `tensormap.replace.tile.global_address` / `.global_dim` to modify a copy in global memory,
publish it with `fence.proxy.tensormap::generic.release.gpu`, then use
`cudaGraphKernelNodeSetParam(node, off, bytes, 128)` to write it into the downstream node's parameter block in one go.
ptxas rejects the `.param` space outright, so the planner **cannot** modify the consumer's by-value copy in place; it must
build the descriptor in global memory and push it over. DG's device path already uses this sized whole-block write API.

**Two pitfalls, each of which cost a fair amount of time, written down here:**

1. **The descriptor parameter is not at offset 0.** On this machine it is at 112, with a preamble in front of it. Writing at a guessed offset
   corrupts the parameter block, and launch reports `an illegal instruction was encountered` -- which looks nothing like an offset problem.
   The offset must be queried with `cuFuncGetParamInfo`, which is exactly what DG has always done for every parameter it patches.
2. **`nvcc -arch=sm_90a` with `-cubin` is enough** (this is exactly how DG's planner is compiled; verified
   `.target sm_90a` in the cubin), but whole-program compilation is not: nvcc also generates compute_90
   PTX as a JIT fallback, and ptxas rejects `tensormap.replace` on that. A standalone binary needs
   `-gencode arch=compute_90a,code=sm_90a`.

**A shortcut that looks beautiful but is actually a trap.** Triton has another, "decomposed" lowering for `tensordesc<>`:
the base pointer plus per-dimension shape/stride are all passed as ordinary int64 parameters, and the descriptor is built inside the kernel --
then DG could patch it without doing anything. It is selected by whether `tensordesc_meta` is empty, and
`TRITON_OVERRIDE_ARCH=sm80` forces it. But that is **pre-Hopper software emulation**:
for the same kernel, a normal compile contains `cp.async.bulk.tensor` 4 times, the forced decomposed path **0 times**.
In other words, taking that path turns off the TMA hardware entirely. Not acceptable.

**The host path is also ready-made**: of the 12 parameters of `cuTensorMapEncodeTiled`, only 3 vary with the shape
(globalAddress, globalDim, globalStrides), and the rest all come from Triton's `tensordesc_meta`;
moreover upstream already has a byte-for-byte matching encoder, `initTMADescriptorWithMetadata`
(`torch/_inductor/codegen/cuda/device_op_overrides.py:387`, used by AOTI's cpp wrapper),
so the swizzle / elem_type mapping does not have to be implemented from scratch.

So both paths work. Not implemented, because this is a feasibility question, not a deliverable for this round; if it were done,
what it buys is host-side TMA kernels going from "one child swap per shape" to "patched in place like any other kernel".

### 25.2 There is a third path, and it is the cheapest one (same day, addendum)

The above framed the problem as "either patch the 128 bytes or re-harvest the whole segment", and missed one path. Triton actually has **three** lowerings for descriptors:

| Path | What the kernel parameters look like | TMA hardware? | What DG has to do |
|---|---|---|---|
| Host-side descriptor (`TensorDescriptor.from_tensor` as an argument) | 128-byte CUtensorMap passed by value | Yes | Either patch those 128 bytes, or treat it as an opaque site |
| Decomposed path (forced by `TRITON_OVERRIDE_ARCH=sm80`) | Ordinary integers for the base plus per-dimension shape/stride | **No** (`cp.async.bulk.tensor` appears 0 times) | Nothing, but it amounts to turning TMA off |
| **Device-side descriptor** (kernel writes `tl.make_tensor_descriptor`) | Base pointer + i32 per dimension + one pointer to a 128-byte global scratch | **Yes** | **Nothing; it is already served today** |

In the third path, the kernel builds the descriptor itself in that global scratch with `tensormap.replace.tile.*` + `tensormap.cp_fenceproxy`,
and the shape comes in as ordinary i32 parameters -- that is,
**DG patching parameters the usual way is enough, without touching a single byte of the tensormap rules**.
This is exactly why the TMA Inductor generates itself never had problems (the opening paragraph of 23): it takes this path.

So the real trade-off is not "patch the descriptor or re-harvest", but:

- **The kernel is generated by us** (Inductor's TMA, `use_tensor_descriptor`) -> already the third path, nothing to do.
- **The kernel is user-written and takes host-side descriptors** (the common way persistent matmuls are written in vLLM / sglang) ->
  only then does patching the 128 bytes come into play, or keeping the opaque site from 24.

### 25.3 Breaking the device path down further (same day, addendum)

The rebuttal phase independently reproduced the core point: ptxas rejects the `.param` space outright
(`State space incorrect for instruction 'tensormap.replace'`), `.global` passes under
`-arch=sm_90a`, and under `-arch=sm_90` it reports unsupported. Consistent with the experiment on this machine.

But it pointed out that the device path actually has **two** approaches, and I had only thought of the hard one:

1. **Table lookup** (simple, **needs neither sm_90a nor `tensormap.replace`**): at capture time, encode the 128-byte descriptor for each shape on the host
   and put them in a device array; the planner just picks a row:
   `cudaGraphKernelNodeSetParam(handles[i], off, &table[sel], 128)`. This is the call DG already issues
   (`dynagraph.py:1123`, where the size is already a variable, not a hard-coded 8); only the size and
   the source operand change. When the set of shapes is enumerable, this is enough.
2. **Build on device** (the one validated by the experiment on this machine): needed only when the shape is truly unbacked and cannot be enumerated.

There is also an implementation cost that should not be underestimated: the host path's `dg_init` is currently **one flat byte buffer indexed by offset**,
whereas a 128-byte tensormap parameter needs its own 128-byte-aligned slot, and `nd.kp[j]` has to point there.
That breaks the invariant that the "compare dirty bits by offset" code in `dg_step` relies on. It is not "no new mechanism".
There is also an open question that cannot be checked on this machine: for a 128-byte parameter declared `.align 64`, will the driver accept a
source pointer that is not 64-byte aligned -- the headers only constrain the arguments of `cuTensorMapEncodeTiled`, not this.

### 24.2 The opaque site failed to track the input behind a descriptor (same day, correction)

After 24 was pushed, trying it on a real kernel revealed that it was wrong. The probe
`dynagraph/probes/probe_tma_real.py` runs the mxfp dequantization from OpenAI's `triton_kernels`
(`numerics_details/mxfp.py:140`, where a host-side descriptor is passed directly as an argument to `@triton.jit`):

```
rows= 128  rel diff 0.0e+00
rows= 128  rel diff 3.7e-01      <- same shape a second time: wrong
rows= 256  rel diff 0.0e+00
```

With DG off it is exact, with it on it is wrong; for the same shape, **reusing the same pair of input tensors** is correct, while **creating new ones** is wrong.
This comparison pins down the cause: the opaque child graph baked in an input address that the harvest key did not track.

The wrapper looks like this:

```
buf2 = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(arg2_1, [64, 64])
_upcast_from_mxfp_0.run(buf1, buf2, arg5_1, s6, 1, s85, 2*s32, ..., stream=raw_stream0)
```

The descriptor `buf2` is built from the input `arg2_1` on the **previous line**. `_extern_read_positions` only looks at names
that appear on the site's own line, and that line has only `buf2`, not `arg2_1` -- so the input address never entered `_hkey`,
the second call with the same shape hit the old harvest, and the child graph replayed with the previous address: no error, wrong answer.

**Fix**: `_tma_sources` parses `X = TensorDescriptor.from_tensor(SRC, ...)`
(and the old API's `create_Nd_tma_descriptor`) into "descriptor variable -> source tensor",
and the site scan follows that link; along the way every name also goes through `_view_of`, so the class
"the site reads a view of some input that was built on an earlier line" is caught as well.

The lesson is not in the code: **`probe_tma.py` is a toy kernel I wrote myself, with the descriptor built on the launch line,
so it could never hit this hole.** The real kernel exposed it on its very first run. From now on, any change of this kind that "infers structure from
wrapper text" must be run through a real third-party kernel.

## 26. Patching host-side TMA descriptors as parameters: implementation plan (2026-09-21 evening)

Xinwei set the direction: descriptor patching should be supported, and **the construction method is provided by the wrapper**; DynaGraph does not implement the encoding itself.
Looking into it, upstream already has all the parts in place; the wiring just only reaches Inductor's own kernels.

**Parts that already exist:**

- `torch/_inductor/runtime/static_triton_launcher.py:43` `make_host_tma_expander()` returns
  `expand_host_tma_descriptor(cache, pos, tensor, cacheable, shape, strides, block, meta)`,
  which directly yields `[CUtensorMap, *shape, *strides]`, **and reuses the already-encoded copy when the address has not changed**,
  so on the hot path even `cuTensorMapEncodeTiled` is skipped.
- `triton.backends.nvidia.driver.make_tensordesc_arg(desc, meta, _)` is the layer underneath it;
  it uses `fill_tma_descriptor_tiled`, and the whole swizzle / elem_type mapping stays inside Triton itself.
- `torch/csrc/inductor/static_launcher/cuda.cpp:110-143` already knows how to locate those 128 bytes inside
  Triton's `PyCUtensorMap` object (the Python side cannot reach them: `PyCUtensorMap`
  exposes no attributes at all; `tp_basicsize` 256 = 128 header + 128 descriptor).
- `inductor_meta["host_tma_descriptor_args"]` (`select_algorithm.py:953`) is already a ready-made format for
  "the wrapper declares how descriptors are constructed": one `{block_shape, shape, strides}` per descriptor.
  This is exactly the shape Xinwei wants -- **it is just that today only Inductor's own host-TMA codegen sets it**;
  user-written kernels do not.

**To do:**

1. **The kernel table understands expansion.** A `tensordesc<>` signature entry is 5 parameters in the cubin,
   so the current use of `args.index(nm)` as a cubin index is wrong. Change it to walk `arg_tys` (the expanded
   string the static launcher has already computed) and get the cubin parameter range for each signature entry.
2. **Take descriptor construction from the wrapper.** `_tma_sources` (already added in 24.2) gives
   "descriptor variable -> source tensor"; add the block shape on top. The more correct approach is to have
   `generate_tma_descriptor` (`codegen/wrapper.py:2651`, the only exit point in the Python wrapper)
   also emit a structured table along the way, so DynaGraph reads a table rather than text -- consistent with the `_capture_deps` principle.
3. **Host path.** Per shape: compute the source buffer's address and shape -> `expand_host_tma_descriptor` ->
   write the 128 bytes at that offset in the node's parameter block (the offset comes from `cuFuncGetParamInfo`; see the pitfall in 25),
   and the trailing shape/stride integers are filled in as ordinary scalars. In `dg_init`,
   `nd.kp[j] = nd.buf.data() + off`; the 128 bytes themselves fit, only alignment needs a separate slot.
4. **Device path.** Two approaches, already analyzed in 25.3: when shapes are enumerable, "the host precompiles a table and
   the planner picks a row" (does not need sm_90a); only for truly unbacked shapes, build on the device
   (`tensormap.replace` + release fence, verified working on this machine).

**Be honest about scope.** `flash_attn_3` 3.0.0 on this machine is `_C.abi3.so`, a compiled extension;
from DynaGraph's point of view it is an opaque extern call, and it rebuilds its descriptors every time in its own host code,
so it **needs no change**. What actually runs into this path is the `triton_kernels` kind of library (mxfp dequantization has been measured to hit it).
So this is worth doing, but its coverage is "OpenAI triton-kernels-style libraries", not "all modern workloads".

### 26.2 Two pitfalls that would turn into silently wrong answers (caught during the rebuttal phase)

**Pitfall 1: the kernel table is shared by both paths.** `extract_kernel_table` is the only producer of `self.kernels`;
the host patcher and `generate_planner` (the device patcher) read the same table -- `generate_planner`'s
own comment says "the two paths are consistent by construction". So if a descriptor kernel is only changed from "reject" to "patchable on the host path",
**the device path will still take it into the table and then not patch the descriptor**, and the result is a silently wrong graph.

This has to be a three-way split, not two-way: descriptor kernels must be marked "only the host path can patch this", and when
`generate_planner` encounters one it must either force `update="host"` (`:4555-4561` already has a force in the opposite direction to copy from),
or send it back to `opaque`. Until the device path has the precompiled table of 25.2 or the `tensormap.replace` of 25.3,
descriptor kernels must not take the device path.

**Pitfall 2: the upstream descriptor cache keys only on the address, not the shape.**
`expand_host_tma_descriptor` at `static_triton_launcher.py:62-75` only compares
`cached[0] == data_ptr`; shape and strides are parameters it accepts but never looks at. Measured: two views of the same storage,
`big` (64,256) and `big[:32]` (32,256), have the same `data_ptr()` and the same `pos`, and get the same expansion.

This is fatal for DynaGraph, because **buffer addresses in the arena are often frozen across shapes** (the fixed layout freezes them by construction;
under the dynamic layout slot 0 is frozen too). Copying that cache as-is would give a descriptor with a stale globalDim.
So call `make_tensordesc_arg(desc, meta, None)` directly (or `cacheable=False`),
with our own memo keyed on `(source address, resolved shape, strides, block_shape)`.

**An assertion the probe should gain along the way**: run the same address with the same shape twice but different symints,
and assert that the 128 bytes pushed down **differ** at globalDim. This is exactly what `probe_tma.py` is missing,
and it is the same kind of blind spot that let the 24.2 bug slip through.

## 27. The registration path has landed: the descriptor is now a patchable parameter (2026-09-21 evening)

Xinwei's remark is the entire design of this section: **"when fa3 and the like register, they don't only register parameters, they also have to register how the tma is built"**,
and **"triton kernels go through the registration path too"**. Done, and verified by all four probes.

### What the registry looks like

`torch/utils/_capture_tma.py`, same principle as `_capture_deps` -- **the producer declares, DynaGraph only reads**:

```python
@dataclass(frozen=True)
class TmaArg:
    param: str                      # parameter name in the kernel signature
    source: str                     # which tensor the descriptor is built from (its name in the wrapper)
    block_shape: tuple[int, ...]    # block shape; does not change when the shape changes
```

Inductor's wrapper, in `UserDefinedTritonKernel.codegen` (`ir.py`), declares every
`TMADescriptorStable` argument, and `codegen_tma_declaration` (`codegen/wrapper.py`)
writes it as one line:

```python
_capture_tma.register('_upcast_from_mxfp_0', [
    _capture_tma.TmaArg('out_desc', 'buf0', (64, 128)),
    _capture_tma.TmaArg('mx_tensor_desc', 'arg2_1', (64, 64))])
```

**This line must be written in `self.header` (module scope), not in the `call` body** -- DynaGraph parses this wrapper
before the first `call`, so writing it in the body is as good as not writing it (measured: it reports `undeclared TMA descriptors ['xd','yd']`).

Undeclared descriptors still take the old path: `extract_kernel_table` raises `Unsupported`, and the launch falls back to an opaque site.
So turning the switch on only gives "declared" descriptors an additional, better path; it cannot make anything that was servable before unservable.

Switch: `TORCHINDUCTOR_DYNAGRAPH_TMA_PATCH` / `config.triton.dynagraph_tma_patch`, **off by default** (reason at the end).

### The table has to understand "one signature entry expands into several parameters"

`tensordesc<bf16[64, 128]>` is not a single parameter in the cubin. The width depends on whether `arg_tys` carries per-descriptor meta:
with meta it is `1 + 2*rank`, without it is `3 + 4*rank`. **Greedily scanning `arg_tys` for contiguous `i`/`l` runs is wrong** --
an ordinary int immediately following a descriptor gets swallowed along with it. So go through `_desc_rank`/`_desc_width`/`_tensordesc_meta`,
compute the width per signature entry, then ask `cuFuncGetParamInfo` for each segment's byte offset (`map_off` 128 bytes + `tail_off` some scalars).
The offset is never 0: measured on this machine, 48 / 240 / 432.

The host patcher's C++ takes one more argument, `const char* const* desc`; for each descriptor it first `memcmp`s the 128 bytes,
and only if they differ does it `memcpy` and set `touched` -- if the shape did not move, the descriptor produces no driver call.

### Cache key: not the shape, but "where the source tensor is and what it looks like"

The first version cached the 128 bytes by harvest key (shape). It blew up immediately on the real mxfp kernel:

```
rows= 128  rel diff 0.0e+00
rows= 128  rel diff 3.2e-01      <- same shape appearing a second time
```

Because **a descriptor's source can be a graph input**, and input addresses may differ on every call (those copied into the store change;
those read in place are the caller's own tensors). The shape was reused, the address was not. This is the same kind of bug as the one in 24.2 that already shipped once:
**not everything the descriptor depends on went into the key**.

Changed to key on "what these bytes are a function of": each source's `(data_ptr, shape, strides)`.
For regions whose sources are all in the arena it is still one rebuild per shape (arena addresses are frozen across shapes); where the source is an input, it follows the input.
Note that the cache only saves "re-encoding those 128 bytes"; **every replay still pushes the blob to `dg_step`** --
because the exec may have been swapped for another one (a different topology combination), and the C++ side's `memcmp` compares against the copy that exec itself stores.

### Source names have to go through the rename/view table

The second measured failure: `buf2 = buf0; del buf0  # reuse`. The declaration says `buf2`, while the arena only knows
`buf0`/`buf1`/`buf6`. Pure renames go through `self.alias`; `reinterpret_tensor` also carries its own sizes/strides/offset.
So first `_view_of(src, self.alias, self.views)` gets the name that owns the storage and the element offset,
then `_desc_view_geometry` gets the view's own geometry, and `as_strided` puts it in place.

### Both pitfalls from 26.2 are plugged

- **Pitfall 1 (both paths share one table)**: in `build()`, after the update mode is chosen, a nonzero `n_desc` forces
  `self.update = "host"`, and the log says `update device -> host: the region has a TMA descriptor`.
  Until the device path has the 25.3 implementation, it does not touch descriptor kernels.
- **Pitfall 2 (upstream cache looks only at the address)**: do not use `expand_host_tma_descriptor`; call
  `make_tensordesc_arg(desc, meta, None)` directly, with our own memo keyed as above.

### The assertion added to the probe

The one 26.2 asked for has been added to `probe_tma_shape.py`: hook `_desc_blobs`, record each pushed
`(source address, shape, 128 bytes)`, then assert -- **for every descriptor whose address did not move across shapes, its 128 bytes must have changed**.
Without this, "numerics are correct" might just be because the kernel never reads globalDim from the descriptor at all. Measured:

```
descriptors: 8  address unchanged across shapes: 2  of which the 128 bytes changed with the shape: 2
```

### Measured results

| Probe | Switch off | Switch on |
|---|---|---|
| `probe_tma inductor` | pass (Inductor's own TMA is built on the device; parameters are still pointer + integers) | pass |
| `probe_tma device` | pass | pass |
| `probe_tma host` | pass, via opaque site | pass, **via parameter patching**, recording 0 |
| `probe_tma host_undeclared` (`register` replaced by an empty function) | pass, via opaque site | pass, **still via opaque site**, log says `undeclared TMA descriptors ['xd','yd']` |
| `probe_tma_real` (OpenAI `triton_kernels` mxfp dequantization) | pass, via opaque site | pass, **opaque -, recording 0, max relative diff 0.0e+00** |
| `probe_tma_shape` (address frozen, shape moving) | -- | pass, 1.2e-07, the 128 bytes really do change with it |

The real-kernel row is the point of this work: `2 kernel nodes, 0 child sites` -- the descriptor kernel
**stays in the main graph**; it no longer needs a child graph, and it is no longer re-harvested for every new shape.

The `host_undeclared` row pins down a safety property: **turning the switch on cannot break anything that was servable before**.
If the producer did not declare (a third-party wrapper, or a codegen path whose declaration is not wired up yet),
`extract_kernel_table` still raises `Unsupported`, the launch falls back to an opaque site, and numerics and recording counts are unchanged.
So this switch only adds a better path for the "declared" part.

### Which axis this step spends

What it buys is coverage and recording cost (the opaque site is gone; new shapes do not need a re-harvest). What it spends is latency:

```
DynaGraph update auto: device (gpu 13 us at the recorded shape, host patching ~61 us)
DynaGraph update device -> host: the region has a TMA descriptor
```

The region is forced from the device path back to the host path, 13 us -> 61 us. This cost is not intrinsic; it is because the device path of 25.3 has not been built yet.
**So it stays off by default for now**: today both paths can serve kernels with host-side descriptors (off = opaque site, on = parameter patching);
the switch decides "how it is served", not "whether it can be served", so it is not my place to silently trade away latency on Xinwei's behalf.
Once the device path (precompiled table / `tensormap.replace`) is done, this trade-off disappears, and defaulting it on can be discussed then.
