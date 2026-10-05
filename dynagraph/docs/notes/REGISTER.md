# What the tier-3 register should look like (2026-09-23)

Goal (Xinwei): any kernel whose source we control should be able to enter the dynamic cudagraph just by registering it; registration should also take care of initialization problems such as
JIT / autotune / workspace (following SGLang, which folds warmup into the cudagraph lifecycle).

Method: inventory the open-source operators used by vLLM along four lines (vLLM's own csrc operators, attention libraries, GEMM/MoE libraries,
and vLLM's hand-written Triton kernels), look at "what determines the launches of one call" for each operator, and derive the interface from that.
The conclusions of each line's raw report are summarized in Section 2; the file:line references are in the reports (inside the container at `/opt/vllm`).

## 1. Conclusion: register = variant key + prepare + plan + bind

For every operator inventoried, the launch splits into two levels: **structure that changes with "bucket-level" host scalars**, and **values that change on every call**.

| Part | When it runs | What it computes | Examples |
|---|---|---|---|
| `variant(call) -> key` | Every call; must be cheap | The host inputs that select the kernel / number of launches / cluster | DeepGEMM's M bucket -> block_m, cluster; FA3's max_seqlen_q -> one_mma_wg/pack_gqa, num_splits; fused_moe's JSON M buckets; Triton's int==1 / 16-alignment / None specialization; rms_norm's `data_ptr%16` |
| `prepare(call)` | Once per variant, outside capture | JIT compile, autotune, obtain function handles, `cudaFuncSetAttribute`, allocate workspace | DeepGEMM JIT (vLLM's deep_gemm_warmup is exactly this); Triton warmup/autotune; FA4 cute.compile; FlashInfer's first plan loading the module |
| `plan(call) -> [LaunchSpec]` | Once per variant | Per node: kind (kernel/memset/memcpy), function, block, smem, cluster, attributes such as PDL, argument template, workspace spec | All |
| `bind(call) -> argument values` | Every time the key changes (shape/pointer/context) | What to fill into each slot of the template: pointers (which tensor + offset), scalars (shape expressions), TMA descriptors (re-encoded from ptr/dims/strides/box/swizzle) | All |

The consumer (DynaGraph) divides the work along this split:
- `bind` changed -> `cuGraphExecKernelNodeSetParams` patches the node (func, grid, smem, and arguments can all be swapped), on the order of microseconds.
- `variant` changed but cluster / node count unchanged -> still SetParams (swap func + smem).
- cluster changed (topology unchanged) -> ExecUpdate can do it (it requires identical topology and wipes out patches; see the ENGINE/EXTERN notes).
- Node count changed -> **ExecUpdate cannot do it** (different topology fails outright). The only options left: pre-place nodes for every variant + select with SetEnabled
  (measured: 2.13 us per switch, 0.27 us per disabled node per replay), switch to another existing exec, or **build the graph again from the plan and instantiate it** (no re-capture needed).
  So the set of variants should ideally be **enumerable** (DeepGEMM's warmup and vLLM's M buckets already enumerate them).

Why the key is split into two levels, variant and bind: the same operator across 28 layers has the same variant and differs only in pointers.
`plan` runs once per variant, `bind` runs once per layer per shape -- this is how the earlier measurement of "every layer is the same" (mm 112 -> 4,
attention 28 -> 1) is expressed in the interface.

### Mapping to SGLang
SGLang requires an attention backend to implement `init_cuda_graph_state` (resident buffers),
`init_forward_metadata_capture_cuda_graph` (write metadata at capture time), `init_forward_metadata_replay_cuda_graph`
(update metadata before replay), plus `post_warmup_hook`. Mapped onto this design: resident buffers = workspace spec;
at capture time = plan; before replay = bind; warmup = prepare. The difference is that SGLang warms up twice and captures one graph
for every capture size, whereas here prepare runs once per variant and plan/bind do not run on the GPU.

### Corrections after review (2026-09-23; review forwarded by Xinwei, each point checked)

1. **describe gives argument bytes, not a binding table.** With only bytes, the consumer does not know which field comes from Q's address or which block is a
   TMA descriptor that must be re-encoded -- so every new key on every layer runs the entire host dispatch again. Measured: the FA3 describe route costs 969 us per site
   (`fwd_describe` C++ 379 us + vLLM attention Python and sink parsing ~590 us), slower than the 353 us of "record once + rebind".
   **Who implements bind must be explicit; two options are allowed:**
   - Declarative binding: the operator gives the source/expression of each field, and the consumer compiles and executes it (simple kernels, Triton).
   - Native binder: the operator provides C++ `bind(plan, call, resources) -> argument bytes`, reusing the library's own argument construction (FA3, CUTLASS, DeepGEMM).
     FA3 goes this way first: the plan stores a `Flash_fwd_params` template + the argument-construction function of the template instance; bind only fills pointer/shape fields and then calls
     `to_underlying_arguments` (TMA is encoded here). We do not first make the compiler understand CUTLASS Params just for the sake of a unified interface.
2. **workspace is not "allocated once per variant at prepare".** Within the same variant, a larger shape also requires growing the workspace; 28 layers sharing a plan
   does not mean sharing a workspace, and neither can concurrent lanes. Split it into: immutable, shareable plan/compiled artifacts; workspace held by the consumer per (site, lane)
   (capacity, alignment, initialization requirements, lifetime); **initialization that must happen before every execution (zeroing semaphores) is a node in the graph**.
   The scope is limited to the registered operator's own temporary resources.
3. **PDL is an edge, not an attribute.** In a graph, PDL is an edge type/port between nodes (CUDA Graph edge data); `plan -> [LaunchSpec]`
   either supports only plain serial chains or expresses dependency edges. First version: support serial chains only, keep the edges from capture (SetParams does not touch edges);
   express edges when building the graph from the plan. What is missing is not just four fields but also **a contract for dependencies and resource lifetimes**.
4. **Order: first a minimal FA3 closed loop, then fill in the general interface.** Fix one path and do native bind -> same variant with changed shape/address, zero capture, and correct results ->
   hook up the workspace/memset/TMA that this path needs -> measure host overhead for new shapes and for replay hits -> then extend to PDL, other variants, other operators.
5. Trimmed `_fa3d_C` (SM90a, BF16, hdim128): CMake 11.5s, ninja compile + link 146s (`build/.ninja_log`).

## 2. Inventory results (classified by what determines the launch)

| Class | Representatives | Argument form | Structure changes with shape? | How it can be registered now |
|---|---|---|---|---|
| A. Simple CUDA kernels | reshape_and_cache_flash, rotary, activation, most cache/quant (about 50 in csrc) | Pointers + scalars | Mostly unchanged (T==0 -> 0 launches) | **Hand-written `launches`** (done: kv_update, byte-for-byte check passed) |
| B. CUDA kernels with dispatch heuristics | rms_norm (width chosen by alignment), qk_norm_rope, per_token_group_quant (block depends on shape), topk (1 <-> 2 launches), selective_scan (by-value struct) (about 40) | Pointers + scalars, some by-value structs | Kernel / launch count changes with shape | Hand-written `launches` + variant key; by-value structs need a layout (or a C++ packer) |
| C. Hand-written Triton kernels | 589 `@triton.jit` in vLLM (attention ops, fused_moe, LoRA, mamba, FLA, ...) | Triton packed arguments + 2 scratch pointers | constexpr/specialization changes with shape | **`triton_launch(fn, grid, *args)`** (done: derived from Triton's own binder/cache, byte-for-byte check passed); autotune goes into prepare |
| D. Libraries with large structs + TMA | FA3, FlashMLA, CUTLASS scaled_mm, DeepGEMM, Machete | By-value Params containing CUtensorMap | Both variant and cluster change with M | For now **`recorded(op, template_key, sources)`** (record once per structure, rebind pointers for the other layers; done); the proper fix is for the library to provide describe |
| E. Launch inside JIT host code | FA4/CuTe DSL, fmha_sm100 | Compiled host function called via TVM-FFI | Same as above | Can only be recorded; the proper fix is for cute's `__call__` to have a describe mode and trace once more |
| F. Distributed/IPC | custom all-reduce, lamport, symmetric memory | Pointers into IPC registration slots | Must be registered across ranks after capture | Out of tier-3 scope (post-capture registration is its own protocol) |
| G. Launch inside another library | moe_permute (cub), gptq (cuBLAS), awq (ATen sum) | Not visible | - | Can only be recorded; cuBLAS to be discussed separately |

Four common requirements the interface must be able to express (items the current implementation lacks are marked [missing]):
1. **Non-kernel nodes**: memset (FA3 semaphore, dynamic fp8 quant, persistent_topk), memcpy (all-reduce). [missing]
2. **Variable number of launches**: T==0, an extra combine when splitting, topk 1 <-> 2, Marlin tiling by M. Returning a list is already supported; when the count changes it must take the structural-change path (pre-place / switch exec / rebuild from plan, not ExecUpdate). [missing] (currently falls back)
3. **Launch attributes and dependency edges**: cluster (changes with M in most libraries), the function attribute for >48KB smem (done in prepare); PDL is an edge type between nodes (see correction 3 above). Cluster: checked; PDL edges: [missing]
4. **TMA descriptors as an argument kind**: DeepGEMM/FA3/FlashMLA/CUTLASS/triton_kernels all have them, and each must be re-encoded whenever M changes.
   In the interface this should be `TmaDesc(tensor, box, swizzle, ...)`, encoded by the consumer with `cuTensorMapEncodeTiled`. [missing]

Two more points:
- **Temporary allocations inside the operator** (cub sort workspace, topk workspace, FA3's lse/out_accum, Triton scratch):
  these should be declared as a workspace spec (size = f(shape)), with the consumer providing a resident buffer. Currently `Launch.owner` holds the memory allocated at recording time.
- **Hidden inputs**: env (`VLLM_BATCH_INVARIANT`, `DG_*`), device properties (SM count, occupancy), pointer alignment, string arguments (kv_cache_dtype)
  -- all of these must go into the variant key, so `variant` must receive the real arguments, not just shapes.

## 3. The interface each library should provide (the "source-controllable" half)

The inventory shows that the libraries are all missing the same thing: **"compute one launch without launching it"**. Several libraries already do half of it:
- FlashInfer: `plan()` (host computes the partition, writes device workspace) / `run()` -- the closest.
- FA3: `get_scheduler_metadata` already makes the host decisions once, but returns only device data and drops decisions such as num_splits/pack_gqa, which `fwd` then recomputes.
- DeepGEMM: `get_best_config` + `make_tma_*` are pure host C++, just not bound to Python; every launch goes through a single place, `LaunchRuntime::launch`.
- CUTLASS: `to_underlying_arguments` + `get_grid_shape` are describe; every launch goes through two places, `kernel_launch` / `launch_kernel_on_cluster`.

So the minimal change is to add a describe sink at these **single launch exits**: when a sink is set, write
`(CUfunction, grid, block, smem, cluster, attrs, argument bytes)` into it and return without launching.
FA3 needs changes in 4 places (the three launch points main, combine, prepare + the semaphore memset),
CUTLASS needs 2 changes to cover all CUTLASS operators, and DeepGEMM needs 1.
In addition, FA3's `mha_fwd` allocates out/lse/accum itself; describe mode needs the caller to pass them in (= workspace spec).

This turns class D from "record" into "call the library's describe": it still runs the library's host code (tens of us), but no longer needs side-stream capture,
and it gets the structure (variant) and bindings directly instead of guessing them.

## 4. Current implementation (commit 1c860b1acf)

Interface `torch/utils/_capture_launch.py`:
- `register(qualname, launches, key=, variant=, prepare=, exact=)`; `Launch(kernel, grid, block, smem, args, cluster, owner)`.
- Lifecycle (SGLang's warmup -> capture metadata -> replay update, turned into per-operator, per-variant):
  `prepare` runs once per (operator, variant) and is guaranteed to be outside capture (at build time before main graph capture; at replay time before the first launch of a new variant);
  `launches` gives the structure at capture time, where it is checked byte for byte, and gives new arguments on new shapes / `key` changes.
- Three ways to write `launches`: hand-written (classes A/B), `triton_launch` (class C, reuses Triton's binder/cache, autotune in prepare),
  `recorded(op, template_key, sources)` (transitional for classes D/E/G: record once per structure, rebind for the rest).
- `launches_of` lets an operator delegate to the operators it calls; `record/check` verify the declaration (by-value structs with `exact=False` only check the kernel name and launch shape).

DynaGraph: registered sites do not go into child graphs and are not harvested; at capture time they are inlined and checked; per step, `(hkey, key())` decides whether to SetParams;
operands are compiled once from the call text, and views are built only for the buffers that are used; when all extern calls are registered, the graph is no longer rejected by `extern-launch`.

Verification (`_regress_quick.sh` 12/12):
- `probe_register.py`: three kinds of source-controllable kernels in one model -- hand-written CUDA (load_inline), Triton with `@triton.autotune`, DeepGEMM JIT GEMM --
  **using registration only**: all shapes served by DynaGraph, 0 harvests, 0 upstream recordings, every shape bitwise identical to eager, 13 variants each prepared once and all outside capture.
- vLLM Qwen3-0.6B (GEMM via Inductor Triton, kv_update hand-written, attention via the FA3 describe copy): 122/122 served, checked against eager on every call,
  tokens identical to those with DynaGraph disabled; new shapes ~18ms (~26ms under high load), hits ~0.11ms. The profiling region without metadata still falls back
  (vLLM issues one kernel of its own outside FA3).

## 5. Next steps (in the order given by the review)

1. Minimal FA3 closed loop (native plan/bind): first measure what the 379 us of `fwd_describe` consists of, then split `mha_fwd` into
   plan (heuristics/variant/workspace spec, once per variant) + bind (fill pointers/shapes + `to_underlying_arguments`).
   Verification: same variant with changed shape/address, zero capture, correct results; measure host overhead for new shapes and for replay hits.
2. Hook up the workspace this path actually needs (out_accum/lse_accum held per site x lane, growable), the semaphore-zeroing node, and TMA.
3. Then extend: PDL edges, other variants (node-count changes go through pre-place / switch exec / rebuild from plan), other operators (rms_norm, Triton attention, DeepGEMM).
