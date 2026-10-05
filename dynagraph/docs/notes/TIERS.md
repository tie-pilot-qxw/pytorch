# Tiering by graph property: when the planner is not needed at all, and when the host side is enough

This started from a measurement (`docs/notes/BENCH.md`): `dynagraph_planner` takes **14.3%** of every replay
(43.7 us / 315 us, 48 nodes). Today this cost is paid unconditionally, but **most graphs do not need
device-side updates at all**.

## The criterion is not "how dynamic", it is **where the value lives**

This is the key to the whole tiering. The one irreplaceable reason for a device-side planner is that **the host
cannot get the value before replay**. As long as the host can get it, patching the graph on the host is enough,
and cheaper.

| Tier | Criterion | What it needs | Coverage today |
|---|---|---|---|
| **0 static** | No symbols | Nothing to do | Goes back to upstream via the `no-symbols` fallback, already correct |
| **1 backed SymInt** | **The host knows the shape before the call** -- it is that Python int in `inputs` | Host-side `cudaGraphExecKernelNodeSetParams`, finished before replay | **Everything served today** |
| **2 unbacked SymInt** | The value exists only in device memory (`nonzero`, data-dependent decode, device-side counters) | Only a device-side planner | Not a single line written yet (hard problem #1) |

**So right now those 43.7 us pay for a capability that is not used yet.** Every path served today gets its
shapes from the Python int `inputs[sym_i]` -- all of it is tier 1.

## Moving tier 1 to the host saves more than those 43.7 us

1. **No longer requires every node to be `deviceUpdatable`.**
   `cudaGraphExecKernelNodeSetParams` works on **any** kernel node and does not need
   `cudaLaunchAttributeDeviceUpdatableKernelNode`. So the whole
   `_begin/_end_device_node_collection` handle-collection machinery, and the `handle-mismatch`
   class of fallbacks, can go away entirely.
2. **This path may lead straight to extern kernels.**
   cuBLAS/NCCL nodes **have no handles** -- they do not go through Triton's static launcher,
   and this is a knot the device-side route cannot get around (it is what the line in `docs/notes/ELASTIC.md`
   about "the handle table and the kernel table may each be missing one while the counts still match"
   refers to). Host-side node updates do not care where a node came from.
3. **The patch overlaps with the GPU work of the previous step.** The host-side calls are not in the graph and
   not on the critical path; as long as the host has slack, it comes for free.

## The cost, and tier 1's own criterion

The host side has to make one call for each of N nodes; this is **host time**. Whether it can be hidden depends
on whether the GPU has work:

* Measured on this model: GPU 275 us per step, the host side currently uses only 57 us -- plenty of slack.
* But on graphs with little GPU work, the N host calls may in turn become the bottleneck.

So tier 1 needs one more decision internally: **estimated host patch time vs the graph's GPU time**.
Both numbers are measurable (GPU time with CUDA events, host side by timing a dry-run);
measure once at build time and decide, no guessing.

There is also a qualitative difference, not an optimization question: the device-side planner is a node in the
graph, so a DynaGraph region **can be embedded in a larger captured graph**; host-side patching requires this
region to be replayed on its own. There is no such need today, but do not treat it as a pure downgrade.

## How to make the device-side planner itself cheaper (tier 2 still needs it)

The generated code currently makes calls one node at a time:

```c
cudaGraphKernelNodeSetEnabled(handles[i], 1);
cudaGraphKernelNodeSetGridDim(handles[i], dim3(...));
{ int32_t v = ...; cudaGraphKernelNodeSetParam(handles[i], 24, &v, 4); }
```

48 nodes x 2-4 each = 150-200 independent device-side runtime calls,
43.7 / 48 = **0.91 us/node**.

1. **Use the batch API.** CUDA 13.1 has
   `cudaGraphKernelNodeUpdatesApply(updates, count)`, which submits a batch in one call.
   **Our own microbench already uses it** -- `microbench/devupdate_scale.cu`;
   its README records "3000 nodes = +148 us/launch", i.e. **0.049 us/node** (2 updates batched per node).
   An 18x gap. But the two sides differ in more than batching (thread configuration and updates per node also
   differ), so **18x is a lead, not a conclusion**: first extend `devupdate_scale.cu` into a
   batched vs unbatched comparison at N=48, settle that, and only then change the codegen of `generate_planner`.
2. **Early-exit when the shape has not changed.** Keep a copy of "the last applied ctx" on the device; if it is
   identical, return right away. Consecutive steps with the same shape are common in training; this saves the
   whole cost.
3. **Patch only the nodes and parameters that actually depend on symbols.** Right now everything is patched
   unconditionally. Which grids/parameters contain symbols is known at codegen time.
4. **Skip `SetEnabled` for nodes whose grid is always positive.** That is one call per node,
   possibly a third of the total.

## Order of work

1. First measure batched vs unbatched (microbench, ten minutes); this decides how the tier 2 codegen is written.
2. Tier 1's host-side path -- it saves 14% of replay, removes the handle mechanism, and is also the entry point
   for extern calls.
3. Extern interface design (see the next document), built on top of tier 1.
4. Leave tier 2 for unbacked SymInt; at that point rewrite the planner with the batch API.

---

# The same yardstick for "which kernel to pick": declared branches (design, 2026-09-22, proposed by Xinwei)

The tiering above measures where the **shape symbols** live. The second thing the same yardstick should measure
is **which kernel to pick**. Today the approach is "grow a body when a topology change is observed"; the criterion
itself is never declared, and the consequence is that every branch pays for one capture the first time it appears.

**What declaring buys is that capture, not "moving the criterion to the device".** Where the criterion is
evaluated is a separate, independent axis, and in most cases the host is enough (Xinwei's words: "It doesn't
necessarily have to be the device; actually sometimes simply going through the host is fine too; what's left
is the re-capture time").

## How expensive the thing we are buying is

| Action | Measured | Source |
|---|---|---|
| **Re-capture + instantiate + upload** | **6.8 ms** (500 nodes) / **19.8 ms** (3000 nodes) | `docs/notes/EXTERN.md` |
| SWITCH body switch | ~9 us (whole call) | microbench |
| Swap one node's func on the exec | 0.73-0.81 us/node | `docs/notes/FEASIBILITY.md`, "2. All measured data" |
| Full-graph `cudaGraphExecUpdate` | 0.47-0.53 us/node | Same as above |
| Swap one child graph | 0.44 us/child (N=48) | `host_update.cu` |

Three to four orders of magnitude apart. **That is the entire case for declaration.**

## Axis 1: where the criterion lives

| | Examples | How it is evaluated |
|---|---|---|
| **Host** (the vast majority) | DeepGEMM tile selection, cuBLAS heuristic, `use_cascade`, `num_prefill_tokens > 0` | `_deps_key()` computes it in `__call__`, before launch, and writes `ctx[BODY0+s]` |
| **Device** (few, but not served at all today) | FA3 picking split-k from the length distribution of `seqused_k`, MoE picking a kernel from the topk histogram | Criterion compiled into the planner kernel, reading tensors declared via `_capture_scratch` |

The device row is the first real use case for tier 2 (the value exists only in device memory) -- not `nonzero`,
but **device-decided kernel selection**. For the host to read the value it would need a D2H sync, and that cost
has been measured: once a host-built TMA descriptor is forced device->host, **13 us -> 61 us**.

## Axis 2: how much the branches differ (decides which mechanism)

What can be changed in place on an **already-instantiated exec** is the hard boundary of this axis.
`cudaGraph_t` is the template, `cudaGraphExec_t` is the executable produced by instantiate;
"swapping on the exec" means changing the latter without re-instantiating.

| What differs between branches | Mechanism | Exists today? |
|---|---|---|
| Scalar parameters | `SetParam` / host shadow buffer | Yes: `dynagraph.py:1241`, `1911` |
| Pointers | Same as above (8-byte write) | Yes: four sources |
| grid | `SetGridDim` / `nd->p.gridDim*` | Yes: `917`, `1906` |
| One kernel fewer (subset) | `SetEnabled`, issued only on transitions | Yes: `913-916` (today only used to express empty tensors) |
| **Different kernel / different cluster (variants)** | **Pre-place V sibling nodes linearly, enable only one each time** | Yes, the mechanism exists, but it is not used for variants |
| **The kernel function itself** | `cuGraphExecKernelNodeSetParams` swapping func -- **supported by the driver, measured bit-exact** | No: **DynaGraph code never writes `nd->p.func`** |
| **dynamic smem** | Same as above -- **supported by the driver, measured to be changeable** | No: **`sharedMemBytes` never appears anywhere in the file** |
| **cluster dimensions** | SetParams cannot change it (node attribute); full-graph `cudaGraphExecUpdate` can, **at no extra charge** | No |
| Actually more/fewer nodes | Only a SWITCH body | Yes |

Three boundaries, re-verified 2026-09-22 on GPU 7 (`dynagraph/verification/_wf_cublas_funcswap.py`,
`_wf_v_swap.py`, `_wf_execupdate_scale.py`):

- `cuGraphExecKernelNodeSetParams` can swap **func + grid + smem + parameters**, but **cannot swap cluster**.
  fp32 addmm: the original node M=512 (cutlass 64x64, smem 98304, cluster (0,0,0)) swapped to
  M=2048 (cutlass **128x64**, smem **147456**, same cluster) -- `max|err|=0`.
  Deliberately pinning smem at the original node's 98304 gives an illegal memory access, so smem really was raised.
  Crossing clusters always crashes (CUDA 912 / 715).
- **Full-graph `cudaGraphExecUpdate` swaps the cluster too, at no extra charge**:
  a single-node graph switched to (1,8,1) in 1.65 us, to (2,1,1) in 1.66 us, same cluster 1.61-1.70 us,
  all four targets `max|err|=0` / unwritten 0 / out-of-bounds 0.
  Cost = **fixed 1.23 us + 0.085 us per node** (1/8/48/200/1000 nodes:
  1.31 / 1.86 / 4.53 / 15.22 / 86.67 us).
- `cudaGraphExecChildGraphNodeSetParams` **carries no launch attributes at all** --
  this is exactly the cause of the illegal instruction in gpu-regime on the evening of 2026-09-19.
  So putting the cluster in `_node_sig` is required, not merely conservative.

**Warning: the earlier note that "the same cluster also crashes" was wrong.** The 715 at `_wf_swap2.log:36` is a
**cascade of a sticky CUDA error**: in the same run, M=1022 crossing clusters crashed first, and after that every
call returns 715. Running M=2048 alone is bit-exact; let 1022 crash once first and the same 2048 immediately
turns into 715. **Lesson: in those logs, no line after the first failure can be used as data.**

`_node_sig` = `(node type)` or `(node type, clusterX, clusterY, clusterZ, cooperative)`.
Two kernels that differ only in func are the same topology (on purpose; the child route being cheap depends on
this); kernels that differ only in smem also count as the same -- **this was an oversight**: smem has never been
checked or set, while the driver actually supports changing it. DeepGEMM does vary smem, so this is a real hole
to fill.

## Two things the astra review overturned (2026-09-22, re-verified by measurement)

I sent this design to astra for review; it blocked it and pointed out two places where **code that was already
running was wrong**. Both were verified on GPU 7 (`microbench/cond_persist.cu`), and **both hold**.

### 1. A conditional's value does not persist across replays (a real bug in the planner, fixed)

The planner's original comment said "Set only when the shape changed, so a skipped planner
leaves the last selection standing". The CUDA documentation says: without `cudaGraphCondAssignDefault`
the value is **undefined** at the start of each execution; with that flag it is **reset to the default**.
Neither one is "left standing".

Measured:

```
[default 0] set body 2 and execute -> 300 (correct); replay without setting the condition -> 100  **back to the default branch**
[default 1] set body 2 and execute -> 300 (correct); replay without setting the condition -> 200  **back to the default branch**
```

So under `TORCHINDUCTOR_DYNAGRAPH_TOPOLOGY=switch`, **any recurring shape runs the default body instead of the
selected one**, silently. The default setting is `host`, so for the default configuration this bug was latent.
The regression suites did not catch it: they change shape on every call, so `NSYM != 0` always held, which
happened to hide the bug.

**Fixed**: both planners (the main planner and the unbacked `dynagraph_planner_u{j}`) now
**write unconditionally on every replay**, and that write was moved **before** the "early-exit when the shape
has not changed".

### 2. SWITCH is paid on every replay, and it is two orders of magnitude more expensive

The "SWITCH ~11 us fixed each" recorded at `docs/notes/FEASIBILITY.md` (section "2. All measured data") was taken by me as the **switching
cost**, so I wrote "use SWITCH when switches are rare". astra pointed out that it is a **replay cost** -- paid
whenever execution passes through a conditional node, even if the same body is selected. This qualitative
correction is right.

**But the 42.8 us/site I measured the first time was an artifact of GPU contention** (GPU 7 had a simulation job
at 100% at the time); re-measured on an idle GPU, the old record of 9-11 us is the correct one. 8 dispatch
points, 3 bodies, selection fixed and unchanged, replay-only, sweeping the number of sites
(`microbench/cond_persist.cu`):

| Sites | Bare | **SWITCH per site** | **Pre-placed 1-of-3 per site** |
|---|---|---|---|
| 1 | 1.8 us | +10.8 ~ +12.3 | +0.7 |
| 2 | 2.9 | +9.2 ~ +10.2 | +0.4 |
| 4 | 4.8 | +8.4 ~ +8.6 | +0.4 |
| 8 | 7.7 | +8.2 ~ +8.7 | +0.5 |
| 16 | 13.8 | +8.3 ~ +8.6 | +0.5 |

**SWITCH ~8.5 us/site/replay, pre-placed ~0.5 us/site/replay, a 17x gap** (not the 80x I wrote at one point).
The pre-placed number matches the independently measured 0.269 us per disabled node (2 disabled nodes per site).

A real side observation: on a fully loaded GPU the SWITCH per-site increment rose to 42.8 us (5x), while
pre-placed barely moved. **SWITCH's device-side graph dispatch is especially sensitive to GPU contention**, and
serving GPUs are fully loaded to begin with.

**So the conclusion is a division of labor, not "SWITCH is useless":**

- **Variants are all kernel nodes and V is small** -> pre-place + SetEnabled. 17x faster.
- **The branch contains non-kernel nodes (memset / copy / child graph)** -> SWITCH only.
  Device-side `cudaGraphKernelNodeSetEnabled`, as its name says, only handles kernels; see the next item.
- **V is large** -> SWITCH. The pre-placed standing cost is 0.269(V-1), SWITCH is a constant 8.5,
  **crossover at about 33 variants**.
- **`torch.cond`** -> SWITCH only, by nature: program-level control flow, and branches have arbitrary side effects.

### 3. Disabling only applies to kernel nodes; memset/copy in a branch still run

My own experiment missed this -- in `_wf_preplaced_enable.py` all three variants had only kernel nodes.
Build a branch with a memset and the problem shows:

```
3 nodes in the graph: Kernel Memset Kernel
disable only B's kernel      -> 0    **memset still runs and wipes A's result to 0**
disable kernel + memset      -> 777  (A's result preserved)
```

`cudaGraphNodeSetEnabled` (host-side) **can** disable Memset nodes (measured: "allowed");
but device-side `cudaGraphKernelNodeSetEnabled`, as its name says, only handles kernels.
So **the granularity of selection must be the whole branch subgraph** (initialization, copies, reductions, any
externally visible side effect), not "the few kernels of this branch". On the device path, either find an
alternative expression for the non-kernel nodes, or prove that running them unconditionally is harmless, or
reject the branch.

## The cheapest path: linear pre-placement + SetEnabled (measured 2026-09-22, at Xinwei's prompting)

Capture all V variants of a site **linearly into the main graph as sibling nodes**, and enable only one each time.
`docs/notes/FEASIBILITY.md` (section "2. All measured data") had long recorded "~0.85 us per extra slot, cheaper than SWITCH when V <= 13",
but two key things had not been measured; now they have been (`dynagraph/verification/_wf_preplaced_enable.py`,
`_wf_disabled_toll.py`, GPU 7, colocated):

**1. Pre-placed nodes can each carry a different cluster -- this gets around the one wall.**
Capture fp32 addmm at M=512 / 1024 / 4096 into **the same graph**: the measured clusters of the three nodes are
`(0,0,0)` / `(1,8,1)` / `(2,1,1)`, smem 98304 / 231424 / 231424, and the grids all differ too, yet they
**coexist in the same instantiated exec**. The cluster is a node attribute, fixed at instantiate time -- but each
variant is **its own node** and carries its own, so neither SWITCH nor ExecUpdate is needed.

**2. Disabled really means disabled.** `docs/notes/SOLUTION.md` (section "Probe problems found by the review (showing how easily the criteria themselves run idle)") noted that the probe at the time had not proven
this ("what was actually proven is only that SetEnabled(0)/(1) are consistent with each other"). This time all
three output buffers were filled with sentinels and only one node was enabled at a time: every time
`max|err|=0`, unwritten 0, **other nodes written 0**.

**Cost (measured):**

| | Paid on switch | Standing cost per replay |
|---|---|---|
| **Linear pre-placement + `SetEnabled`** | **2.13 us** (two calls, median) | **0.269 us x (V-1)** |
| SWITCH body switch | **9-11 us** constant | **0** (variants are not in the main graph) |

The standing number was fitted by sweeping the node count (N=1/2/4/8/16/24, enabling only the 0th:
2.03 / 2.26 / 3.15 / 3.60 / 6.30 / 8.23 us/replay, measured interleaved, median == minimum).
**A disabled node costs about a third of an enabled node (0.78-0.80 us/node)**, not zero.

**(The following passage was written assuming "SWITCH is a switching cost"; it has been overturned by item 2
above and is kept for comparison)**

**So which one to choose depends on how often switches happen:**

- **V small, frequent switches** (decode seqlen changes every step, the 3 clusters of cuBLAS/DeepGEMM)
  -> **pre-place**. At V=3 it is a 2.13 us switch + 0.54 us/step standing, clearly beating SWITCH's 9-11 us.
- **V large, rare switches** -> SWITCH. The standing cost is 0, and the 10 us switch is amortized.
- Rough crossover: when switching every step, `2.13 + 0.269(V-1) < 9`, i.e. **V < 26**; when never switching,
  SWITCH always wins. (The "V <= 13" in `FEASIBILITY.md` is of the same order, but it booked the two costs as one.)

**On "whether disabled nodes need to be patched": the planner is our own code, and it already pays almost nothing
for them.** When it computes an empty grid in the grid section, it does `SetEnabled(0)` and then **`return`s
immediately** -- the subsequent `PARAMS` / arena `PTRS` / `EXTPTRS` / `VIEWPTRS` are all skipped.
The only part still paid is the `INPTRS` section, which was **deliberately placed before the grid section**, with
the comment "so a node disabled at this shape still takes the address for when it is enabled again",
and it only runs when `ctx[INDIRTY] != 0` (some input address moved).

**This section can also be brought to zero**, using state the planner already has: it stores each node's enabled
state in `ctx[STATE0 + i]` (precisely so that it "issues SetEnabled only on transitions"), so the
disabled -> enabled transition is known. Move `INPTRS` after the grid section and, at the moment of the
transition, force one refill of that node's input pointers -- in steady state the number of runtime calls is
unchanged.

**So the standing overhead of the pre-placement route is only that 0.269 us/node/replay, which is the driver's cost
of walking the graph, not ours.**

**What really needs to change is something else**: today the **only trigger for `SetEnabled(0)` is "the computed
grid is empty"** (empty tensor); the planner has no notion of "variants". To use it for variant selection, the
planner must enable nodes by **the variant number selected for the site**, not by whether the grid is empty. And
that slot **already exists** -- it is `ctx[BODY0 + s]`, used by SWITCH. So the change is: one more mapping,
"node i belongs to variant v of site s", and the enable condition becomes `v == ctx[BODY0 + s]`. That mapping is
what the declaration has to provide.

## ExecUpdate is not a substitute for SWITCH: it wipes out every patch

First, a retracted conclusion for the record: at one point I computed "SWITCH 9 us constant vs ExecUpdate 1.23+0.085N,
crossover at 91 nodes". **That was a wrong comparison**, because the two do not operate at the same level at all.

`dynagraph/verification/_wf_execupdate_clobber.py` (2026-09-22, GPU 7): capture a graph of `y = x + 1`,
instantiate it, use `SetParams` to redirect the output pointer to `z` (simulating DynaGraph's per-shape patch),
replay and confirm it wrote to `z`; then **ExecUpdate back to the same template graph**:

```
1) replay as captured                y[0]=2.0  z[0]=0.0
2) SetParams redirects output to z   y[0]=0.0  z[0]=2.0   <- patch takes effect
   ExecUpdate(same template graph) rc=0
3) replay after ExecUpdate           y[0]=2.0  z[0]=0.0   <- patch is gone
```

**ExecUpdate re-parameterizes every node from the template**, so the arena pointers, symbolic grids and scalars
that DynaGraph patched in per shape are all overwritten back to the values from template capture time. Every branch
switch would require re-patching the whole graph -- and we did exactly the opposite optimization (early exit when the
shape has not changed, 42.7 -> 0.9 us), which ExecUpdate would undo outright.

| Mechanism | Can change cluster | Clobbers patches? | Cost |
|---|---|---|---|
| `SetParams` (single node) | no | no | 0.73-0.81 us/node |
| `ChildGraphNodeSetParams` | no | no (only swaps that child graph) | 0.44 us/child |
| **SWITCH selecting a body** | yes (different bodies can have different clusters) | **no**, each body keeps its own patches | **~9 us constant** |
| `ExecUpdate` (full graph) | yes | **yes, wipes everything** | 1.23+0.085N **+ re-patching the full graph** |

**So for cluster changes, SWITCH bodies are the only path.** That actually makes the design simpler: no need to choose
between two mechanisms, and no need to bucket by node count. ExecUpdate's one advantage, "the only thing that can change
the cluster", is cancelled out completely by "wipes out every patch".

(Its cost data is still kept, because it answers a different question: **changing the cluster is not expensive in the
driver by itself** -- a single-node graph switched to (1,8,1) costs 1.65 us and to (2,1,1) 1.66 us, no different from
the same cluster. So switching across clusters between SWITCH bodies has no hidden cost either.)

## Where DeepGEMM lands (Xinwei's example, checked)

`get_best_config<ArchSpec>(const GemmDesc&)`,
`/opt/vllm/.deps/deepgemm-src/csrc/jit_kernels/heuristics/common.hpp:14` --
**pure host C++, does not read a single byte of device memory** (nowhere under `heuristics/` is there `item()` / `cudaMemcpy` /
`data_ptr`); the inputs are just the fields of `GemmDesc` (m/n/k/num_groups/dtype/majors/num_sms/...)
plus three process-global switches. It enumerates candidates + takes the argmin of an analytic cost model; there is no tuning table.

What changes between two configs:

| Changes | Does not change |
|---|---|
| **kernel function** (every config field is a C++ template parameter, baked into the JIT source; the hash is the cubin cache key) | **grid** -- the main GEMMs are all persistent-CTA, `gridDim = (num_sms,1,1)` |
| **dynamic smem** | **number of kernels** (on SM120, split_k>1 adds one reduce kernel; that is the exception) |
| **cluster dimensions** (and the **number** of launch attributes changes too -- the cluster attribute is only appended when cluster>1) | |
| TMA descriptor bytes in the parameter buffer | |

So the correct decomposition for DeepGEMM is: **one SWITCH body per cluster shape; inside a body, swap func +
smem with SetParams on the exec.** Neither the grid nor the node count needs to change.

**And the number of bodies is far smaller than the number of kernels** -- the dense cuBLAS sweep is a ready-made reference:
over M=1..4096, **94 kernels, 357 changes between adjacent M, but only 3 cluster shapes**.
**3 bodies, not 94.** This number is what makes "pre-capture every branch" feasible.

## What a declaration must carry

1. **Branch list.** Not an optimization but a **correctness requirement**: if a body that was never captured gets selected,
   nobody can fall back. The count is the number of cluster shapes, not the number of kernels.
2. **Criterion**: either a pure host-side function (the `_capture_deps` resolver has exactly this shape),
   or a device-side expression that can be lowered to C++.
3. **Operands** (only needed for device-side criteria): which device tensors to read -- `_capture_scratch` already solves this.

Three constraints: all branches are **shape-equivalent** (the host still does the arena layout, so the criterion must not
affect output shapes), **workspace is allocated for the max branch**, and the criterion is **cheap and pure**.

## Two numbers now filled in (formerly "two things not yet measured")

1. **Full-graph `cudaGraphExecUpdate`, which can change the cluster, now has a cost**: 1.23 us fixed + 0.085 us/node;
   changing the cluster costs nothing extra. See above.
   (The 0.47-0.53 us/node at `docs/notes/FEASIBILITY.md` (section "2. All measured data") comes from an earlier synthetic bench, 6x slower than this one;
   the two were never reconciled; this measurement takes precedence.)
2. **"Same-cluster func swap has an unexplained failure" -- explained: we misread our own log**:
   a cascade of sticky CUDA errors. See above. In reality, same-cluster func swap + tile swap + larger smem are all bit-exact,
   **meaning the DeepGEMM scenario itself holds**.

## On mixing host / device

Today `dynagraph_topology` is a global `host | switch` (`config.py:2217`).
Xinwei pointed out that "some parts updated by the host and some by the device" would get complicated -- agreed, but
**declarations actually reduce the need for mixing**: a site that has declared its criterion is self-contained (all branches
pre-captured, the criterion carries its own operands), so there is no requirement for each site to pick its own mode.
The actual tiering is still "does this graph need the planner"; only the criterion widens from "are there unbacked symbols"
to "are there device-side branch declarations".

---

# Which node types can be disabled -- the real boundary of the pre-placement route (measured 2026-09-22)

`microbench/enable_node_types.cu`, with four node types in one graph: Kernel / Memset / Memcpy / ChildGraph:

| Node type | Host `cudaGraphNodeSetEnabled` | Device `cudaGraphKernelNodeSetEnabled` |
|---|---|---|
| Kernel | **allowed** (and can be re-enabled) | allowed |
| Memset | **allowed** | **cannot get a handle** -- `cudaGraphDeviceNode_t` can only be obtained from |
| Memcpy | **allowed** | `cudaLaunchAttributeDeviceUpdatableKernelNode`; |
| **ChildGraph** | **`invalid argument`** | non-kernel nodes cannot be named at all |

**ChildGraph cannot be disabled**, and this one is critical for us: DynaGraph's extern sites **are child graph nodes**.
So the route "pre-place a set of variants, each a captured child graph, and enable only one" **does not work** --
selecting a child graph variant can only be done with SWITCH. Pre-placement applies only when the variants can be flattened
into kernel / memset / memcpy nodes.

This is also the real reason SWITCH has not been replaced by pre-placement, not "it is cheaper when V is large":
**it is the only mechanism that can select or skip a child graph as a whole.**

## Coverage on real serving

`_enable_selectable` turns this into a check, run on Qwen3-0.6B / vLLM (504 sites):

| Site | Number of sites | Selectable via enable? |
|---|---|---|
| `mm` | 336 | **enable:both** x336 |
| `unified_kv_cache_update` | 84 | **enable:both** x84 |
| `unified_attention_with_output` | 84 | enable:both x28, **enable:host-only** x56 |

**448/504 (89%) can be selected via enable on both the host and the device path; 56 (11%) only on the host**
(the body contains memset/memcpy); **not a single one is enable:no**.
So the pre-placement route covers all sites in host mode, and 89% in device mode.

# Today cuBLAS does not go through SWITCH at all (so end2end shows no difference)

`dynagraph_topology` defaults to `"host"`, and none of the vLLM probes ever changed it. In host mode:

- extern sites are **plain child graph nodes**, the conditional handle is 0, and the SWITCH loop in the planner spins idle;
- topology changes are handled by **one complete instantiated graph per combination** (`self.execs`), chosen by the host per shape;
- child graphs are swapped with `cudaGraphExecChildGraphNodeSetParams`, 0.44 us/child.

So that 8.5 us/site/replay has **never been paid even once** -- no wonder end2end serving shows no difference;
the ~173 us measured on 14B/bs=8 is entirely host-side node updates, 0 on the GPU.

**The cost is paid elsewhere: in the number of graphs.** A single 0.6B run already has 364 `new topology` lines.
This is exactly where the acceptance metric Xinwei set (DynaGraph 4048 graphs vs vLLM 354 graphs) comes from --
host mode trades graph count for per-step time.

**The whole branch design is about making this trade more favorable**: push the graph count down to 1,
and **only at that point start paying SWITCH's 8.5 us/site**. This is exactly where the value of the pre-placement route lies:
drop the graph count without paying the 8.5 us -- at the price of only being usable on node types that can be disabled.

# How the astra review was handled (2026-09-22)

The three items above have been re-checked and addressed. Below are the remaining items it raised; **note that two of them
are already done** -- this list has been cited before, do not copy it as "all pending".

1. **One branch id is not enough to execute that branch.** Declaring "I have {plain, cascade, DCP}" does not say how to
   **capture** each one: it needs a construction/capture recipe + usable inputs, an applicability predicate, the complete node
   sequence, parameter update rules, resource requirements, and the version relationship between the selector and these artifacts.
   FlashInfer's "prefill and decode run in the same call" in particular shows that a variant describes
   **an entire execution plan**, not one mutually exclusive kernel.
2. ~~**`_capture_scratch` has an existing bug**~~ **Fixed (commit 63a6c27190)**:
   changed to per-call-scoped collection (`scope(key)` + record into the innermost scope + same name with a different object
   is renamed and kept); `probe_capture_scratch` covers three kinds of misattribution. The original problem: one global pending
   value per op key. Two calls of the same op overwrote each other; two declarations within one call overwrote each other;
   a call that declared nothing inherited the previous declaration that had not been consumed. `pop()` guards against double
   consumption, not misattribution. It had to become a **per-call-scoped** collector, with explicit begin/end + accumulation.
3. ~~**A resolver returning `None` conflates four states into one**~~ **Fixed (commit 63a6c27190)**:
   `_capture_deps` now has `Known` / `Unavailable` / `Unsupported`;
   on the consumer side, `Unavailable` falls back for this call only and is not memoized, and `Unsupported` goes into
   `skip_keys`. The original problem: a legitimate profile/no-op, context not installed yet, backend unsupported, and a bug in
   the resolver all looked the same. It needed typed results
   (`Known(snapshot)` / `Unavailable(reason)` / `Unsupported(reason)`);
   "unknown" must never mean "reuse the previous body".
4. **Shape equivalence is far from enough**, especially for training: the same sizes can come with different strides, storage
   offset, alignment, padding, aliasing, write coverage, in-place semantics and saved-state layout.
   If one branch returns a view of its input and another returns new storage, a single downstream in-place write tells them apart.
   In training, two forward kernels can produce equivalent outputs yet need different backward intermediates / LSE layout /
   RNG state. This needs a **boundary contract**, not a one-line "shape-equivalent".
5. **An incomplete branch space needs an error protocol, not a bounds check.** An out-of-range SWITCH condition
   **raises no error and executes no body**; downstream simply reads the output of the previous replay. The symmetric failure for
   pre-placed nodes is "everything is disabled, the output is stale". It must be possible to: initialize this call's
   selection/state, validate the id **and the applicability predicate**, record the error (which call / which site / which
   selector / what was violated), **block downstream computation and writes**, and have the host observe it before accepting the
   result. An error flag alone cannot stop a KV cache write or an optimizer write.
6. **DCP / multi-GPU**: if one rank rejects a branch while other ranks enter that branch's collective, they will hang.
   Branch-dependent collective ordering needs a cross-rank protocol.
7. **The pre-placement route has a structural prerequisite that is not on astra's list but blocks just as much**: extern sites
   today **are child graph nodes**, and child graph nodes **cannot be disabled** (measured: `invalid argument`). So
   "pre-placed variants + enable" first requires **not capturing extern variants as child graphs** -- that is a by-product of
   item 2 (declaration -> update code); the two have to be done together, pre-placement cannot come first.
8. **`cond-topology` is a hard limit**: a site that is already inside a conditional body may not have a second
   topology; it goes straight to `_fallback`. Nested variants (cascade further split into trtllm/wrapper) will run right into it.
9. **Measurement basis**: `2.13 us` is the latency of the **host-side Python call**, not the planner's device-side transition
   cost, and it does not include work deferred to the next launch; two SetEnabled calls only cover single-node variants;
   switching between multi-node bodies means disabling one batch and enabling another, so the relevant dimension is **the number
   of changed nodes**, not V; the 0.269 us slope for disabled nodes comes from a **serial chain**, does not necessarily hold for
   sibling child graphs with a join, and is a difference between endpoints, not a regression fit.
10. **`ExecUpdate` is already forbidden in device-planner mode**: CUDA disallows it whenever either of the two graphs contains a
   device-updatable kernel node. So the mode in which the clobber experiment ran
   does not apply to our device path anyway (the conclusion stands, but the reason has to change).
11. **Other**: `357 changes between adjacent M` is not the real switch rate in serving; that needs a trace of real request shapes;
   `504 sites, 0 unaccounted for` proves attribution of **the captures that were observed**, not branch coverage,
   not resource lifetime, not content validity.

The adversarial test list it gave (not run yet):

| Test | Failure it targets |
|---|---|
| Shape and pointers unchanged, only content changes so that the branch changes | selector blocked by the shape early-exit |
| Repeatedly select the same non-default body, poison the output before each replay | conditional not persistent |
| A -> B -> arena relocation -> A, poison the workspace | stale parameters / invalid workspace state |
| Select a variant that "has no work" | empty-grid predicate dropped |
| An inactive branch contains memset/copy | kernel-only disabling leaves side effects |
| Unknown selector + downstream KV/optimizer writes | error reported but not safely suppressed |
| Context absent at profile time, real context installed later | `None` conflates unknown with a valid capture |
| Alternate between variants in training, then run backward | incompatible saved-state interfaces |
| Two calls share the same global workspace/context | missing ownership and execution order |

## Tier 2 (library calls), first cut: capture once per signature, clone the rest and rewrite pointers (2026-09-23, commit 0d90e1d4e0)

Flag `TORCHINDUCTOR_DYNAGRAPH_CLONE_SITES=1`, off by default. It only handles `extern_kernels.*` calls that take `out=`
(cuBLAS/cuDNN); `ops:` custom ops do not take this path and are left for tier 3.

- **Signature** `_call_signature`: each tensor's shape/stride/dtype/address alignment (capped at 256B), plus which operands overlap and by how much.
  Same signature = the recorded kernels, grids and scalars are all identical; only the operand addresses in the parameters differ.
- **Representative**: the first time a signature appears, it is warmed up + captured as before; `_clone_plan` records every word in the parameters that points into an operand (operand index + offset).
  A representative that has non-kernel nodes, or that has a word landing within 4KB past the end of an operand (an end pointer, which would not move along), is never cloned.
- **Members**: neither executed nor captured. `cudaGraphClone` the representative's graph and rewrite the kernel parameters in the clone according to the plan (both the kernelParams and the extra forms are supported).
  Non-operand pointers (the cuBLAS workspace in the harvest pool) are kept as is; the clone holds a reference to the representative to keep it alive.
  Not executing is safe: the numerical values from harvest are never read; the answer for this shape comes from the later graph replay.

Measured (Qwen3-0.6B, GPU 2, `dynagraph/verification/_wf_clone_sites.py`):
- Per new shape: of the 112 mm sites -> 4 captured, 108 cloned; the other 56 are kv_update/attention, captured as before.
- Greedy decoding at batch 1/3/5/8/13/16: the 1104 tokens are identical with the switch on and off; `_regress_quick.sh` passes 9/9 with both settings.
- **Harvest time essentially unchanged** (12 runs ~580 ms vs 540-600 ms, within noise). One capture takes only ~17 us
  (the 0.7 ms/call reported by cProfile is an attribution artifact; direct timing is the right number), one clone ~50 us of Python. Capture was never the bulk of the cost.

Direct timing, breaking down where each harvest (~50 ms) goes:

| Chunk | Per harvest | Notes |
|---|---|---|
| `_arena_views` | ~11 ms | Pure-Python recursive evaluation of layout expressions (`go` called 6.5e4 times over 12 harvests). Compiling it into a single function should make this disappear |
| attention + kv_update | ~14 ms | Warm-up + capture of the custom ops. What tier 3 is meant to eliminate |
| 108 clones | ~6.5 ms | ctypes per-node Clone/FindInClone/SetParams. Parameter templates could be cached per representative |
| 4 mm representatives | ~5 ms | cuBLAS first-time cost for a new M, unavoidable (unless switching to a Triton GEMM) |
| Main graph re-capture (new topology) | 15 ms x 5/12 | Caused by attention topology changes |

So the next cuts, ranked by payoff: compiling the layout expressions (host-only, zero risk) > tier 3 for kv_update/attention > faster cloning.
