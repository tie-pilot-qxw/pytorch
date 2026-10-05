# DynaGraph implementation progress

The motivation is wrapped up; see `FINDINGS.md`. This note records **implementation** only.

## Verified (2026-09-18)

### 1. Core mechanism closes the loop -- `microbench/dynagraph_proto.cu`

First time the scattered primitives were chained into a complete loop: **one capture covers a whole shape range, with zero re-recording at run time**.

Four modes run head to head on the same sequence:
- A eager (N launches per step, exact shape)
- B recapture (record one graph per new shape -- what PyTorch does today)
- C pad2max (one static graph, always runs at n_max)
- D dynagraph (record once, a planner on the device changes each node's grid)

**Element-wise correctness comparison passes everywhere** (n = 4096 / 34816 / 65536 / 4097 / 65529,
including sizes that are not a whole number of blocks), max relative error **0.000e+00**:

```
n=4096     max relative error 0.000e+00  match
n=65529    max relative error 0.000e+00  match
All correctness checks passed: one capture covered the whole range.
```

Pitfalls hit and already handled:
- **grid=0 is illegal** (`cudaErrorInvalidArgument`). When n=0 the node must be disabled with
  `cudaGraphKernelNodeSetEnabled(handle, 0)`; the grid cannot be set to 0.
- **A device-updatable graph can be instantiated only once** (FEASIBILITY.md, sections "2. All measured data" and "But the kernel function itself can be swapped").
- **The comparison must reset the inputs every time.** The four modes share buffers, and chained multiplication
  means the next mode does not start from the same point as the previous one -- the first version was written
  exactly this way and reported a false "mismatch".

**Timing numbers are not usable yet**: the GPU is shared with other people. Also, 50 nodes at this size are
GPU-bound to begin with; to see a benefit we have to get into the launch-bound regime (many nodes, small kernels),
which requires an exclusive GPU.

### 2. The planner's inputs really are obtainable -- `survey/dump_grid_exprs.py`

Design section 5 assumes "Inductor's grid is a sympy expression in symints and can be mechanically codegen'd into the planner".
Verified to hold on real generated code:

```python
triton_poi_fused_addmm_relu_0_xnumel      = 512*s77
triton_poi_fused_add_addmm_relu_1_xnumel  = 256*s77
s77 = arg2_1                     # the symint enters the graph as an ordinary int argument
```

- Each kernel's numel is a **closed-form arithmetic expression** in symints
- The symint itself is an ordinary integer input of the graph -- exactly the value that `cudagraph_trees.py:447` uses
  as the `fn_cache` key
- grid = `ceil(xnumel / XBLOCK)`, computed in the launcher

**So the planner kernel can be generated mechanically.** This path works.

### 3. What can be done without a GPU (an important engineering constraint)

- **nvcc compilation does not need a GPU** (measured: 13.1 works, `cudaGraphKernelNodeUpdate` is present)
- **Inductor's CUDA codegen needs a real device** -- without a GPU it reports
  `hasPrimaryContext expects a valid device index`, and fake tensors do not help
  (fake saves the memory allocation, not the existence of the device)
- So: **correctness and codegen can be done on a shared GPU; only timing needs an exclusive one**

### 4. Planner code generation -- `dynagraph/probes/planner_codegen.py`

Converts Inductor's sympy size expressions into device-side CUDA. **The generated code has been verified to compile and link.**

The design does not use switch-case: 3000 nodes written as a switch would produce a huge jump table.
The vast majority of expressions are **affine** (`512*s77`), so they go through a compact table:

```c
struct NodeDesc { int64_t coef, konst; int32_t sym, block, param_off, special; };
// numel = coef * ctx[sym] + konst, one thread handles one node
```

Only **non-affine** expressions (products of several symbols, FloorDiv, Max/Min, etc.) fall into the generated `switch`.
So the planner's code size does not depend on the number of nodes, only on "how many kinds of non-affine expressions there are".

sympy->C covers the operators Inductor actually produces (Add/Mul/FloorDiv/CeilDiv/ModularIndexing/
Mod/Max/Min/integer powers). **On an unrecognized structure it raises instead of guessing** -- a wrong guess gives a silently wrong grid,
and in turn silently wrong results, which is the hardest kind of bug to track down.

**The planner has to be compiled separately with nvcc** (Triton-generated kernels go through Triton's own
PTX/cubin pipeline, not through nvcc), so it is a separately compiled module loaded at run time.

> **Warning, 2026-09-18 correction: `-rdc=true` is not needed.** I originally thought the device-side graph APIs
> (`cudaGraphKernelNodeSetGridDim` etc.) required relocatable device code,
> and therefore the `-rdc=true` + `-dlink` + `lcudadevrt` runtime-linking route.
> **Measured: not needed**:
>
> ```
> nvcc -arch=sm_90a -cubin -o planner.cubin planner.cu   # emits a cubin directly, 5424 bytes
> ```
>
> So the planner can be compiled into an ordinary cubin; at run time `cuModuleLoad` + `cuModuleGetFunction`
> gets the CUfunction, and `cuLaunchKernel` issues it on the capture stream --
> which removes the whole `cuLinkCreate/AddData/Complete` runtime-linking sequence.

## Locating the PyTorch integration points (2026-09-18, three parallel investigations + adversarial verification, every anchor checked)

### A. Parameter offsets: cannot be computed, must be asked from the driver

**PyTorch never builds a flat parameter buffer.** Every launch path ends up in the pointer-array form
`cuLaunchKernel(..., void** kernelArgs, nullptr)`
(`torch/csrc/inductor/static_launcher/cuda.cpp:304`), so the host side never knows the byte offsets.

The line "we allocate 8 bytes per argument" at `cuda.cpp:35` **is a trap** --
that is the slot array holding **pointers**, not the parameter buffer; the 8-byte stride never reaches the kernel.

The real offsets live in the cubin and follow the PTX/CUDA ABI: **natural C alignment; an int32 is not padded to 8 bytes**.
Measured with `cuFuncGetParamInfo` on GPU 5 against a real Inductor cubin:

```
triton_red_fused_sum_2    in_ptr0@0(8) out_ptr0@8(8) xnumel@16(4) r0_numel@20(4)
                          <scratch#0>@24(8) <scratch#1>@32(8)
triton_poi_fused_..._3    ptr@0 ptr@8 xnumel@16(4)  <scratch>@24(8)   <- 4 bytes of padding in between
```

**So a formula like `8*N` is wrong**: two i32s are packed back to back (16, 20), while an i32 followed by an i64 gets 4 bytes of pad.
The driver **must be asked per kernel**. In addition, constexprs (XBLOCK) do not go into the buffer at all,
and Triton 3.8 **appends two 8-byte scratch pointers that are not in the signature**.

**-> The hand-filled `param_offset` idea in my `planner_codegen.py` is right, but the values must come from `cuFuncGetParamInfo`.**

### B. Two unexpected pieces of good news

1. **smem is not a problem.** The shared memory of Inductor's Triton kernels is a compile-time constant frozen at
   `make_launcher()` time; measured across batch 37/41/97/193 and reduction 512/777/1024/2048, it never changes, with zero recompilation.
   The item "no device-side smem setter" in FEASIBILITY.md (section "Measured: the same GEMM, what changes when M changes") **is a cuBLAS finding and does not affect Triton kernels**.
2. **`GridExpr` already has `mode='cpp'`**, which emits C expressions directly (ceildiv becomes `((n+(b-1))/(b))`).
   **We should not write a second sympy->C translator**; we should reuse `GridExpr.from_meta(meta, cfg, mode='cpp')`.

### C. The route changed: it must be "toggle after capture", not "toggle at launch"

`microbench/dynagraph_proto.cu` uses `cudaLaunchKernelEx` + a launch attribute to mark the node
device-updatable **at launch time**. **But PyTorch's Triton path never calls
`cuLaunchKernelEx`** -- `triton_heuristics.py:2159` still carries
`TODO: When the AOTI C++ launch path gains cuLaunchKernelEx support`.
Following the original plan would mean rewriting the launchers on both the PyTorch side and the Triton side.

**It can be worked around** (measured and passing on GPU 5): capture normally, then on the **still-alive template** call
`cudaGraphKernelNodeSetAttribute(node, cudaLaunchAttributeDeviceUpdatableKernelNode, {1})`,
and read `devNode` back with `cudaGraphKernelNodeGetAttribute` (the setter takes a const pointer,
so the handle can only be obtained through the getter). **Not a single line of the launcher needs to change.**

Three accompanying constraints:
- The hook must be `register_graph_capture_end_hook` (`graphs.py:220-227`),
  **not** `register_graph_instantiate_hook` -- the latter fires only **after** `instantiate()`,
  whereas device-updatable must be set **before** instantiate, and such a graph can only be instantiated once;
  miss the window and there is no way back.
- `keep_graph=True` at `cudagraph_trees.py:1149` goes from "convenient" to **required** --
  with `keep_graph=False`, `capture_end_post` destroys the template (`CUDAGraph.cpp:262-270`).
- **Cost**: the order returned by `cudaGraphGetNodes` is unspecified, and launching the same Triton kernel twice with different numel
  gives two nodes with the same `func` pointer -- **the func pointer cannot tell them apart**.
  We would need edge information about capture order from the launcher, or `cudaGraphGetEdges` + a topological sort.
  PyTorch has neither. (The toggle-at-launch route does not have this problem: the handle is returned directly by the launch call
  -- this is the real trade-off between the two routes.)

### D. Concerns ruled out by measurement

- `cudaGraphInstantiateWithFlags(AutoFreeOnLaunch | UseNodePriority)` returns `cudaSuccess` on a graph containing
  device-updatable nodes, and a grid+param patch within the same launch gives correct results.
  **`CUDAGraph.cpp:297-305` does not need to change.**
- A second instantiate of the same graph returns `operation not supported`; **"can only be instantiated once" is confirmed again**.

### E. To do: `cudaGraphUpload`

`driver_types.h:4008-4011` states: a graph whose device-updatable nodes are updated from within the graph
**must be `cuGraphUpload`ed before launch**, and must be re-uploaded after a host-side exec update.
In the whole tree it appears only in the hipify name table; **`CUDAGraph` has no binding for it**.
Measured: it also runs without an explicit upload (the first launch uploads implicitly), so it is a latent hazard rather than immediately fatal, but it needs to be added.

### F. The hardest roadblock: **cuBLAS**

`extern_kernels.mm` / `addmm` appear in almost every real model with a dynamic batch dimension.
It is a kernel node, but **m/n/k and the algorithm choice are decided on the host at capture time**.
**Changing the grid on the device -- the only mechanism this project has proven so far -- has no effect on it at all**,
and the sympy->CUDA planner cannot reach it either. Record-at-max does not save it either:
running a cuBLAS node baked for max on a small shape in the range computes the wrong extent.

Everything else (output metadata becoming a function of symints, dropping `cached_tensor_outputs`,
narrowing input copies, range policy) is **hard but bounded** engineering; **this one currently has no known path**.
FEASIBILITY.md section 3 already flagged GEMM as a real problem ("cuBLAS 61 variants"); now it has become the top risk.

## PyTorch source changes made so far (branch `dynagraph`)

Following Xinwei's view, we go with the **direct `cuLaunchKernelEx` rework** rather than the toggle-after-capture workaround --
the workaround relies on topological sorting to identify nodes, and launching the same Triton kernel twice gives two nodes with the same `func` pointer,
which simply cannot be told apart; changing the launcher is at least deterministic.

The change is smaller than expected: **three files, about sixty lines**:

| File | What changed |
|---|---|
| `aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.h` | Added `_(cuLaunchKernelEx)` to `AT_FORALL_NVRTC_EXTENDED`. This macro is already under `#if defined(CUDA_VERSION)`, so ROCm never hits it |
| `aten/src/ATen/cuda/detail/LazyNVRTC.cpp` | Added a lazy-load stub modeled on `cuLaunchKernel` (it belongs to the "Irregularly shaped functions" category and has to be written by hand) |
| `torch/csrc/inductor/static_launcher/cuda.cpp` | `launchKernel` gets a trailing parameter `CUgraphDeviceNode* out_dev_node`; when non-null it goes through `cuLaunchKernelEx` + `CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE` and returns the `devNode`. The four existing call sites explicitly pass `nullptr` |

The ROCm branch **raises an explicit `TORCH_CHECK` error** rather than silently ignoring it -- it has no counterpart to `cuLaunchKernelEx`,
and a silent degradation would become "we think the node is marked device-updatable but it is not", with the consequence that run-time patches silently stop working.

The driver API names were checked against the CUDA 13.1 headers:
`CUgraphDeviceNode` (`cuda.h:307`),
`CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13` (`:2277`),
`cuLaunchKernelEx(const CUlaunchConfig*, CUfunction, void**, void**)` (`:18058`).

## The cuBLAS roadblock: it can be bypassed, but not for free (branch `dynagraph-gemm`)

`extern_kernels.mm` is only the **default choice**, not the only choice:

```
utils.py:3412  use_aten_gemm_kernels() -> not (max_autotune or max_autotune_gemm)
                                          or _use_autotune_backend("ATEN")
utils.py:2305  use_triton_template()   -> (max_autotune or max_autotune_gemm) and ...
```

Measured (`dynagraph/probes/gemm_route_probe.py`, dynamic=True):

| Config | extern_kernels | Triton template |
|---|---|---|
| default | `['mm']` | none |
| `max_autotune_gemm=True` | `['mm']` | none |
| `+ max_autotune_gemm_backends="TRITON"` | **none** | `triton_tem_fused_addmm_relu_t_0` |

**Turning on `max_autotune_gemm` alone is not enough** -- autotune picks cuBLAS, because it really is faster.
ATEN has to be explicitly kicked out of the candidates before it lands on the Triton template.
Once it lands there, the GEMM is an ordinary Triton kernel: the grid is a closed-form function of symints, the planner can change it,
and the template also fuses the epilogue (relu) in for free.

**The cost, recorded honestly**: in the same autotune run, cuBLAS `mm` 0.0100 ms vs the best Triton 0.0304 ms,
**about 3x slower** (fp32 with `ALLOW_TF32=False`, which is cuBLAS's best case; the GPU is shared,
so absolute values are not trustworthy, but both were measured in the same round under the same contention, so the ratio is roughly usable).
So this is not "cuBLAS is not a problem", but "cuBLAS went from **no known path** down to **a performance trade-off**".

**A newly surfaced open question**: even with `dynamic=True`, autotune picks BLOCK_M/N/K on **one concrete shape**
(64x512 in the probe). The Triton template handles arbitrary M via masking,
so **correctness** across the range is fine; but **how this config performs across the whole range has not been measured**.
This is exactly one of the questions DynaGraph has to answer: when one graph covers a range, how should the tile config be chosen.

### CUTLASS: performance on par with cuBLAS, and **patchable**

The Triton route works but is 3x slower. CUTLASS is a better answer.

Measured (bf16, `max_autotune_gemm_backends="CUTLASS"`): 60 candidates, **0 Triton**, and the winner is

```
cutlass_cd923be8  0.0100 ms
cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_tnt_align8
  _warpspecialized_cooperative_epi_tma
```

A genuine SM90 tensorop warp-specialized kernel, and compiled from source that Inductor generates itself,
not an opaque library call.

**Structural difference from cuBLAS**: what gets generated is a host function, with M/N/K passed in as ordinary int arguments
(`s77` is M), and only the internal `gemm_op.run(stream)` launches. So at capture time it is likewise
"the host computes the grid, then launches". **But the grid computation is written in source we can see**:

```cpp
// tile_scheduler_params.h:246-263
launch_grid.x = possibly_truncate(cta_per_device, problem_blocks_total);  // = min(...)
```

**grid = min(hardware capacity, total number of problem tiles).** On H100 `sm_count=132`; with tile 128x128 and N=512,
`problem_blocks_total = ceil(M/128)*4`, **so it saturates at M >~ 4200 and the grid becomes a hardware constant independent of M**;
for smaller M it is a closed-form function of M. The planner can compute both cases.

**But changing M alone is not enough.** `PersistentTileSchedulerSm90Params` holds quantities derived from M:

```cpp
uint32_t      problem_tiles_m_;            // ceil(M/TileM)
FastDivmodU64 divmod_batch_;               // FastDivmodU64(blocks_m * blocks_n)
FastDivmodU64 divmod_cluster_blk_major_;
int32_t       log_swizzle_size_;
```

`FastDivmod` is magic-number + shift division-to-multiplication; the multiplier has to be computed at construction.
**All of this is pure integer arithmetic, and the planner can recompute it on the device** -- the scheduler's
`initialize()` section has to be moved into the planner codegen. It is bounded work; the source is right there in
`tile_scheduler_params.h`. The good news is that `divmod_cluster_shape_major_/minor_`
depend only on the cluster shape (a compile-time constant) and do not need to be touched.

**Conclusion: the cuBLAS roadblock is solved.** The route is CUTLASS rather than Triton:
on-par performance, at the cost of the planner recomputing one extra block of scheduler parameters.

### Availability of each GEMM backend on this machine

| Backend | This machine | Notes |
|---|---|---|
| ATEN (cuBLAS) | available | not patchable; this is the one to bypass |
| TRITON | available | patchable, but measured 3x slower |
| **CUTLASS** | **available** | **patchable, on-par performance <- pick this** |
| NVGEMM | available after installing two packages | see `SETUP.md` pitfall 10; not tested yet |
| CUTEDSL | not available | needs Blackwell; H100 is Hopper |
| CK / CKTILE | not available | ROCm only |

### End-to-end validation: handle collection works on real Inductor output

`dynagraph/probes/test_handle_collection.py`. The test deliberately uses a model that produces several kernels
(a reduction breaks pointwise fusion, softmax and cumsum each form their own segment),
because with only one kernel "in order, all distinct" holds trivially and verifies nothing.

**The result connects the two lines of work:**

| | handles | kernels | |
|---|---|---|---|
| GEMM not routed | 2 | 4 | the difference is exactly the two `extern_kernels.addmm` |
| `max_autotune_gemm_backends="TRITON"` | **4** | **4** | aligned, handles all distinct |

`extern_kernels` (cuBLAS) are not Triton kernels and **do not go through the static launcher**, so no handle can be obtained for them,
and the planner therefore cannot change them.

**-> The launcher rework and GEMM routing must be done together; without either one we cannot get all the handles.**
These are not two independent optimizations; they are two halves of the same thing.

A precondition confirmed along the way: `use_static_cuda_launcher` defaults to True in OSS
(`config.py:71-83`), so the change is on the default path.
But `can_statically_launch` (`triton_heuristics.py:3030`) bypasses some kernels
back to Triton's own launcher -- those also yield no handle. This test exists to find such gaps:
**if the handle count does not match the kernel count, some node was missed**, and this must be a hard assertion rather than a soft warning.

### The planner's input table can now be extracted automatically -- `dynagraph/probes/extract_kernel_table.py`

For a compiled model, it extracts for each kernel that enters the graph the triple
(numel symbolic expression, XBLOCK, parameter byte offset), which feeds directly into `planner_codegen.Node`.

**Two things we only got right after falling into pitfalls:**

1. **Do not hook `CachingAutotuner.__init__`.** That mostly captures autotune **candidates** --
   measured: a two-layer MLP captured 42, all of them mm template variants, with `kernel_name` still stuck at
   `Placeholder.DESCRIPTIVE_NAME`, and not one of the pointwise/reduction kernels that actually enter the graph was captured.
   The right way is to scan the globals of the **generated wrapper module** (identified by defining `call()`);
   candidate modules also go through `PyCodeCache` but have no `call`. After filtering it drops from 45 to 3, exactly the ones that enter the graph.
2. **The CUfunction is not on the launcher**; it is at `at.compile_results[i].kernel.function`.
   The launcher is a closure function produced by exec and only carries `config`/`n_regs`/`shared`.

**Measured offsets, proving that the `8*N` formula really does go wrong:**

```
triton_per_fused_..._1  [(0,8) (8,8) (16,4) (20,4) (24,8) (32,8)]
                         ptr    ptr   xnumel r0numel scratch scratch
triton_per_fused_..._2  [(0,8) (8,8) (16,8) (24,4) (28,4) (32,8) (40,8)]
                                             xnumel r0numel
```

`xnumel` happens to line up (16 and 24 are both multiples of 8), but **`r0_numel` is at 20 and 28**;
`8*N` would compute 24 and 32, **both wrong**. In the mm template, `ks0@24(4)` is followed directly by `scratch@32`,
with 4 bytes of padding in between. The two trailing scratch pointers (the global/profile scratch appended by Triton 3.8) are also confirmed.

**-> Offsets can only be obtained from `cuFuncGetParamInfo`, not computed.**

**New to-do: the mm template is `grid_type=FixedGrid`, not `Grid1D`**;
its grid does not follow `ceil(numel/XBLOCK)`, so the planner has to handle this class separately.
(Persistent reductions have `XBLOCK=1`, `grid_0 = xnumel`, a degenerate case of Grid1D.)

### The extraction stage of the end-to-end pipeline is complete -- `dynagraph/probes/end_to_end.py`

On real `torch.compile` output, it automatically produces all the inputs the planner needs:

```
[0] triton_tem_fused_addmm_t_0            FixedGrid
    to patch: ks0    = 's77'   offset=24 size=4
    grid (explicit) = ['8*((31 + s77) // 32)', '1', '1']
[1] triton_per_fused__softmax_..._1       Grid1D  XBLOCK=1
    to patch: xnumel = 's77'   offset=24 size=4
```

**Three key realizations, all found by trial and error:**

1. **numel is not a named variable; it is a positional argument inlined at the call site.**
   I originally searched for named variables like `triton_..._0_xnumel = 512*s77` and found none.
   The real form is:
   ```python
   triton_per_..._1.run(buf4, arg1_1, buf0, s77, 256, stream=raw_stream0)
   #                    in_out  in_ptr0 in_ptr1 xnumel r0_numel
   ```
   The expressions can only be obtained by aligning positions with the signature.

2. **The grid of `FixedGrid` is passed explicitly, and is also a closed-form function of the symbols.**
   ```python
   triton_tem_..._0.run(arg3_1, arg0_1, buf0, s77, 8*((31 + s77) // 32), 1, 1, ...)
   #                                              ^^^^^^^^^^^^^^^^^^^^ grid_0
   ```
   So the mm template is likewise computable and patchable; only the source of the grid differs (it does not follow `ceil(numel/XBLOCK)`).

3. **It is not only numel that needs patching -- every argument that contains a symbol does.**
   The mm template's `ks0 = s77` is at offset 24, and its name does not contain numel. A rule of "only handle numel" **would miss it**,
   and the consequence of missing it is that mm computes with the old M and produces **silently wrong results**.
   The current rule is: every argument whose expression contains `s\d+` must have an offset; if even one is missing it is a hard failure, not a warning.

**Not done yet**: generating the planner cubin, injecting the planner node at capture time, replaying per shape and comparing.
The extraction step works; what remains is wiring up `planner_codegen.py`.

## Milestone: First successful DynaGraph execution inside PyTorch (2026-09-18)

`dynagraph/probes/end_to_end.py`: compile -> extract -> generate and compile the planner -> capture (collect handles)
-> for each different shape, write ctx and then replay -> compare against the reference. Fully automatic, no manual intervention.

```
    M=1024   vs compiled 0.00e+00 OK   (compiled vs eager 1.73e-05)   negative control 0.00e+00 OK
    M=333    vs compiled 0.00e+00 OK   (compiled vs eager 2.56e-05)   negative control 1.94e+00 OK
    M=7      vs compiled 0.00e+00 OK   (compiled vs eager 1.37e-04)   negative control 9.52e+01 OK
    M=1      vs compiled 8.00e-08 OK   (compiled vs eager 8.00e-07)   negative control 3.60e+02 OK
```

11 shapes, 10 of them **bit-identical**. **One capture, zero re-recordings.**

### Three places where the result almost turned out to be false

**1. With a row-wise model, a planner that does nothing still passes.**
The first version of the model ended with `.sum(dim=-1)` (row-wise), so output row i depends only on input row i.
So even if the planner does nothing and the graph keeps computing with the capture-time M, taking the first Mi rows is still correct --
**and at the time it really did report "success"**. After switching to `.sum(dim=0)` (a cross-row reduction) and adding a **negative control**
(deliberately not updating ctx, and requiring that the result must change), it immediately exposed that all 4 shapes were wrong.
-> **Any validation of "one graph covers many shapes" must come with a negative control**; otherwise it cannot prove the mechanism is doing anything.

**2. numel appears in two forms side by side; recognizing only one of them misses cases.**
Call sites contain both inline expressions and named variables:
```python
triton_red_..._sum_2_r0_numel = s77
triton_red_..._sum_2.run(..., ks0, xnumel, triton_red_..._sum_2_r0_numel, ...)
```
Parsing only the call site yields the variable's **name**, which contains no `s\d+`, so the reduction range stays stuck at the capture-time M forever.
Named variables must be expanded into their defining expressions.

**3. During expansion, the symbols themselves must be treated as terminals.**
The wrapper contains `s77 = arg2_1` (how the symbol is pulled out of an input tensor).
Expanding indiscriminately turns `s77` into `arg2_1`; the symbol is lost, and the argument is then no longer judged to need a patch --
that is exactly how `ks0` and `xnumel` disappeared after fixing item 2.

### Two more lessons

- **Only patch integer scalar arguments.** Pointers (`*fp32`) have fixed addresses in the graph's private memory pool,
  but after named variables are expanded they become `empty_strided_cuda(...)` and look like they "contain symbols". Filter by signature type.
- **The reference must be the compiled version at the same shape, not eager.** Eager uses cuBLAS + aten softmax,
  while the graph uses Triton mm + online softmax -- **different algorithms**, so differences of 1e-5 to 1.4e-4 are algorithmic differences.
  Using eager as the baseline conflates "is it correct" with "is it the same numerical path" --
  at one point I thought M=7 had a bug because of this; in fact the wrong reference had been chosen.

### Not done yet

- Capture must be done at the **maximum** of the range (outputs and intermediate buffers are allocated by the graph's private pool at the sizes in effect at that time).
  Currently `MMAX` is specified by hand; a real system needs the range policy of `fn_cache` to decide it.
- Inputs/outputs go through hand-fixed buffers; a real system needs to hook into `cudagraph_trees`'s
  `_copy_inputs_and_remove_from_src` and `reconstruct_outputs`.
- The affine lookup-table optimization in `planner_codegen.py` has not been merged in yet (everything currently goes through a switch;
  with more nodes the instruction cache will get tight).

## Integrating into PyTorch: the module has landed (branch `dynagraph`, commit `adcd5dcd99`)

`torch/_inductor/dynagraph.py` (437 lines) + `config.triton.dynagraph` (off by default).
The machinery validated in the prototype was ported into a proper module, and **validated on its own before hooking into `cudagraph_trees`**,
so that "porting drift" and "integration bugs" would not have to be debugged together:

```
  extracted 5 kernels, symbols ['s77']
    bucket_of(1000) = bucket (10,), recorded at 1024
    bucket_of(513)  = bucket (10,), recorded at 1024    <- same bucket as 1000, reuses the same graph
    bucket_of(512)  = bucket (9,) , recorded at 512
    M=1024  vs compiled 0.00e+00 OK   negative control 0.00e+00 OK
    M=333   vs compiled 0.00e+00 OK   negative control 1.56e+00 OK
    M=7     vs compiled 0.00e+00 OK   negative control 9.03e+00 OK
```

Module API: `extract_kernel_table` / `generate_planner` / `compile_planner` /
`launch_planner` / `bucket_of`. Geometric bucketing; `dynagraph_bucket_ratio` defaults to 2.0
(a graph recorded at N serves downward to N/2).

### The output-shape hurdle: lighter than originally estimated

During the localization phase, `reconstruct_outputs` was marked as "the hardest piece" (outputs are rebuilt from metadata frozen at recording time).
Measurements say this judgment needs correcting.

**The first test version had a blind spot**: the model returned `sum(dim=0)`, whose output shape `(D,)` does not change with M,
so **it never hit this hurdle at all**. Retesting with a model whose output shape is `(M, D)`:

> After manually slicing `out_buf[:Mi]`, the computed results are **still bit-identical**.

In other words, **the planner computes correctly, and the first M rows of the output storage are the correct answer**;
the only thing missing is one piece of metadata: "the tensor's declared shape".
`reconstruct_outputs` is a **bookkeeping problem, not a correctness cliff** --
what is needed is for the `size`/`stride` in `outputs_metadata` to be recomputed from the current symints at replay time;
the data itself does not need to move.

### Three remaining pieces

1. **Switch `fn_cache` to bucket hits** (`cudagraph_trees.py:444-491`),
   and record at the bucket's upper bound rather than at the current shape.
2. **Inject the planner in `_record` and collect handles** (`:1521`),
   using the already-added `_begin/_end_device_node_collection`.
3. **Make `reconstruct_outputs` recompute size from the symints** (`:1321`);
   along the way `cached_tensor_outputs` has to be disabled (it bakes the shape in).

## Where record-at-max fails, and arena relayout (2026-09-18)

**Counterexample pointed out by Xinwei**: the total is fixed but each sample differs -- variable-length sequences packed by a token budget,
where the total of 4096 tokens is constant, but it might be 2 sequences x 2048 or 64 sequences x 64. In that case

- buffers shaped `[num_seqs, ...]` get **larger** as the number of sequences grows
- buffers shaped `[max_seq_len, ...]` get **smaller** as the number of sequences grows

**The two buffers move in opposite directions while the total stays the same**, so there is no single "maximum shape" that covers both at once.
`buffers_are_monotonic` correctly detects the violation and falls back to per-shape recording --
but in this kind of workload every batch is a new shape, which amounts to getting no cudagraph at all.
**The safe fallback is useless exactly in the scenario that needs it most**, and that is precisely the RL rollout / variable-length training
that the motivation side identified as the most promising. So this one cannot be sidestepped.

### What is needed is not a general caching allocator

Key simplification: **Inductor fixes the buffer allocation order and lifetime-based reuse at compile time**;
the only thing that changes at runtime is the size of each block. So what the planner has to do is
"re-lay out the same compile-time plan using sizes computed at runtime" --
**the order is fixed, the reuse relations are fixed, only each block's size changes**. That is a device-side bump allocation,
not a general allocator, and it needs neither a free list nor defragmentation.

### Prerequisite verified: pointer arguments can be changed on the device side

`microbench/devpatch_ptr.cu`. Previously the planner only changed scalars; a pointer is 8 bytes,
and the driver might do extra validation, so it had to be verified separately:

```
capture wrote to bufA
  points to bufA (control)       bufA={1000,1001,1002,1003}  bufB={0,0,0,0}
  redirected to bufB             bufA={0,0,0,0}              bufB={1000,1001,1002,1003}
  redirected to bufB and n=64    bufA={0,0,0,0}              bufB={1000,1001,1002,1003}
```

**The planner redirected the entire output within the same launch.** The mechanism is the same
`cudaGraphKernelNodeSetParam` as for scalars, only 8 bytes wide. And the offsets of every argument had already
been obtained from `cuFuncGetParamInfo` (when fetching them for the scalars, all of them were fetched along the way).

### Symbolic sizes are not available in the memory-planning layer (measured, 2026-09-18)

`dynagraph/probes/check_symbolic_sizes.py`. Validating this step first was the right call; the result is **negative**:

```
compute_size_for_scheduler_buffer returns:
  buf0  size_alloc=1048576   concrete integer     <- 1024x256x4 bytes, M already substituted
  buf4  size_alloc=8192      concrete integer
```

The signature at `memory.py:138` is annotated `dict[str, tuple[int, int]]`;
although the values come from `get_allocation_size`, whose signature is `-> Sequence[Expr]`,
by this layer they have **already been concretized with hint values**. So the planner cannot take sizes from here.

**But the lifetime information from the same call is good, and it is shape-independent**:

```
  buf0  live steps [0, 4]      buf4  live steps [2, 3]
  buf1  live steps [1, 4]      buf6  live steps [4, -1]
```

### Hence the two data sources for arena relayout

| What is needed | Where it comes from | Status |
|---|---|---|
| **Which buffers can share a slot** (lifetimes) | `BufferInfo.start_step/end_step` from `memory.compute_memory_timeline`, **shape-independent, fixed at compile time** | OK, verified obtainable |
| **The symbolic size of each buffer** | the size/stride expressions of `empty_strided_cuda((s77, 256), (256, 1), ...)` in the wrapper | OK, `_find_allocations` already implemented |

The two sides are **joined by buffer name (`buf0`/`buf1`...)**, and the naming is consistent.
Note that the timeline contains more buffers than there are `empty_strided_cuda` calls --
some are inplace reuses that are not allocated separately; only the buffers that are actually allocated need slots.

**The conclusion is unchanged, but the path changed**: sizes are not taken from the memory-planning layer; instead it is
"lifetimes from the planning layer + symbolic sizes from the wrapper source". Both pieces are already obtainable.

### Four remaining steps for arena relayout

1. **arena**: allocated once at capture time, with capacity set by the "upper bound on the total" rather than "the sum of the per-block upper bounds".
   In this kind of workload the total is bounded (the token budget is the upper bound).
2. **Argument -> buffer mapping**: already exists. An extracted pointer-argument expression is either
   `buf0` or `empty_strided_cuda(...)`, so it is clear which block it refers to.
3. **Lifetimes**: taken from `compute_memory_timeline` (verified obtainable, and shape-independent).
   At compile time, run one interval-graph coloring on them to decide "which buffers share a slot";
   at runtime the planner only needs to compute each slot's size = the max of the sizes of all buffers on that slot,
   and offset = prefix sum of the slot sizes. **It is a single O(number of slots) scan, not a general allocator.**
4. **Planner extension**: compute each block's size -> accumulate the offsets -> rewrite the pointer arguments of every kernel
   that uses the block to `arena + offset`.

**Impact on ratio=inf**: once arena relayout is in place, the "record at the maximum" constraint reduces to just
"the total does not exceed the arena", and the monotonicity requirement can be dropped entirely -- that is when a single graph truly covers everything.

## Milestone: Arena relayout succeeds end to end (2026-09-18, GPU 6 used exclusively)

`dynagraph/probes/test_arena_e2e.py`. Model `(B, L, D) -> (sum over L, sum over B)`,
with `B*L` always equal to 4096, four splits:

```
B=2    L=2048  total slot bytes=526336   out0 0.00e+00  out1 0.00e+00 OK
B=8    L=512   total slot bytes=139264   out0 0.00e+00  out1 0.00e+00 OK
B=64   L=64    total slot bytes=81920    out0 0.00e+00  out1 0.00e+00 OK
B=512  L=8     total slot bytes=526336   out0 0.00e+00  out1 0.00e+00 OK
```

**One capture (at B=8) served the whole space, and both outputs are bit-identical.**
The middle column shows that the layout really is recomputed on every replay: the two extremes each need 526336 bytes,
while B=64/L=64 in the middle needs only 81920.

5 buffers were colored into 2 slots, and 8 pointer arguments were rewritten.

### Why this case matters

It is **the very scenario the record-at-max proof cannot cover**. On the same source code,
`buffer_dominates` rejects both extremes:

```
recorded at B=512: buf4 needs 2048 elements, only 8 would be allocated
recorded at B=2  : buf2 needs 8 elements, only 2 would be allocated
```

No single recording shape is feasible; only re-laying out on every replay works.

### Two false alarms that were investigated

- **In the first version out1 was entirely correct and out0 entirely wrong**, which looked like "the pointer rewrite only half worked".
  In fact the test code guessed the output buffer from the last character of the name and picked `buf3` (an intermediate).
  The graph's outputs are the buffers whose lifetime has `end_step == -1` (they live through the whole schedule);
  identifying them that way is correct. **out1 was already 0.00e+00 at the time; the mechanism had been fine all along.**
- The diagonal blind spot of the guard is described in the previous section; it was also forced into the open by this case.

## Milestone: Hooked into cudagraph_trees: `TORCHINDUCTOR_DYNAGRAPH=1` works end to end (2026-09-18)

`dynagraph/probes/test_flag.py`; the model goes through `torch.compile(dynamic=True, mode="reduce-overhead")`,
with five shapes (512, 256, 333, 64, 7):

```
  dynagraph=False   recorded 5 times
  dynagraph=True    recorded 0 times
  every shape bit-identical to the control group (0.00e+00)
```

**One graph served all five shapes, without a single re-recording.**

### Why the reference cannot be eager

GEMM is routed to a Triton template (`extern_kernels.mm` does not go through the static launcher,
so no handle can be obtained for it), and that already differs from eager's cuBLAS by 3e-4. Using eager as the reference would misread
algorithmic differences as correctness problems -- this pitfall was already hit once on `end_to_end.py`.
The reference is **the same compile path, with dynagraph turned off, at the same shape**.

Also, the two sides are not guaranteed to be bit-identical either: the control group re-records one graph per shape, and each graph can pick its own
autotune config; DynaGraph has only one graph, so its config is pinned. With a different XBLOCK the reduction's
summation order differs, so a one-ULP difference is correct. So the criterion is written as "no farther from eager than the control group".

### Four bugs, all of them silent

The four bugs caught in this round have one thing in common: **no error is raised at all; the output is a tensor with the correct shape and
plausible-looking values, but the tail is wrong**. They are recorded one by one here, because they all belong to the same class.

**1. Parsed the wrong function.** When graph partitioning is enabled, `compile_fx` uses
`recursively_apply_fns` to cudagraphify **each `partition_N` separately**,
so the runtime argument order is the partition's, not that of `Runner.call` -- and the two really do differ:

```python
def partition_0(args):
    arg3_1, arg0_1, arg1_1, s77 = args     # (activation, weight, bias, symbol)

def call(self, args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args  # (weight, bias, symbol, activation)
    s77 = arg2_1
```

Moreover, inside the partition the symbol **appears directly in the unpacking list** and does not go through `s77 = arg2_1`.
Reading the `call` line gives `s77 -> index 2`; at runtime `inputs[2]` is a `Parameter`,
so `env` comes out empty and the whole thing falls back. This one is the mild case; at least it fell back.

**2. Parsed the wrong buffer name.** The call site uses the name after Inductor's renaming:

```python
buf2 = buf0; del buf0  # reuse
triton_per_...run(buf2, arg1_1, s77, 128, ...)
```

`buf2` is not in the slot table, and the original code **skipped** it, treating it as "an input or a graph output, not managed by the arena".
That is not a skip -- that node keeps its capture-time address, so the arena only holds the mm result written by the first kernel,
and the graph returns an intermediate as the final output.
The same renaming also requires **extending the underlying buffer's lifetime until its last alias dies**;
otherwise the slot would be handed to another buffer while it is still being read.

**3. Used the wrong launch config.** This is the most insidious one. `CachingAutotuner.run` narrows `launchers` down to one only on the
**first real call** (`autotune_to_one_config`, around `triton_heuristics.py:2518`).
Before that, a reduction carries three candidates at once, XBLOCK 1, 8 and 32:

```
triton_per_fused_addmm_mean_relu_sub_1 launchers: [{'XBLOCK': 1}, {'XBLOCK': 8}, {'XBLOCK': 32}]
```

The original code iterated over `launchers` in `DynaGraphRunner.__init__` and took the **last one** (32),
while what actually ran at capture time was XBLOCK=8. So the grid was computed as `ceil(512/32)=16`,
16 blocks x 8 rows per block = **only 128 rows computed**; the remaining 384 rows kept the raw mm output.

The debugging process is worth recording: first confirm that row 0 is completely correct (which shows the pointer patch is fine),
then print the distribution of wrong rows -- `0..127 correct, 128..511 wrong`, a clean prefix;
then compare the tail against the "only mm ran" result: the difference is 8e-4 (exactly the Triton vs torch mm difference),
which confirms the reduction never touched those rows. 128 = 512/4, which is exactly 32/8.

Fix: **warmup must move before planner codegen**, not just before capture --
warmup is precisely the step that makes every autotuner settle; kernels that have not settled are rejected outright.

**4. Unmodeled grid shapes.** `Grid1D/2D/3D` are per-axis ceil-divides and can be modeled;
`CooperativeReductionGrid`, `SplitScanGrid`, `ComboKernelGrid`,
`PrecomputedGrid` and others have their own algorithms and now explicitly raise an exception instead of being approximated.

### Safety net: an on-the-spot self-check right after capture

All four bugs were silent, so "fixing each one correctly" is not enough on its own. Now:

- `generate_pointer_patches` **raises `Unsupported`** when it encounters a `bufN` whose ownership cannot be resolved,
  instead of skipping it. Skipping means leaving the node with its capture-time address.
- At the end of `build()`, **replay once at the recording shape and compare against eager running the same partition**;
  if the results are not bit-identical, the whole thing falls back.

The second point is the key: it turns the entire class of "the planner formula is wrong" problems into fallbacks.
The cost is one replay plus one eager forward pass, which is negligible relative to compile time.
Of the four bugs above, 2 and 3 would each be caught by this check alone.

### Fixed along the way

- `launch_planner` was still the pre-arena 2-argument version, while the planner now takes 4 arguments
  (handles, ctx, arena, slot_off). Argument-count mismatch -> `CUDA_ERROR_INVALID_VALUE`,
  and the failure happened midway through capture, after which the process segfaulted. Everything now goes through `_launch`,
  with `slot_of=None` meaning "no arena" (as distinct from "there is an arena but nothing belongs to it").
- `test_handle_collection.py` now routes GEMM to Triton by default.
  `extern_kernels` can never yield handles, so the routing is a prerequisite, not an option;
  `DYNAGRAPH_ROUTE_GEMM=0` still reproduces the branch where the handle count does not match.
- Added stubs for `_begin/_end_device_node_collection` to `torch/_C/__init__.pyi.in`.

### Test status

All six tests pass (GPU 6):

```
OK test_module.py          OK test_arena.py       OK end_to_end.py
OK test_handle_collection.py  OK test_arena_e2e.py   OK test_flag.py
```

Committed and pushed to `fork/dynagraph` (`0db6208ceb`).
The local `dynagraph` branch has an older base (to match the already-built `.so`),
so pushing goes through cherry-picks onto a `dynagraph-push` branch; the DynaGraph-related files are byte-for-byte identical on both sides.

## Runtime cross-shape verification, and the input-headroom problem it forced out (2026-09-18 evening, GPU 3 colocated)

**Note: GPU 3 is shared with someone else; there are no timing numbers anywhere in this section.**

### The self-check only covers the recording shape, and that is a real gap

The self-check added at the end of `build()` in the previous section compares against eager only on the **recording shape**. The XBLOCK bug caught this round
happened to be wrong on the recording shape as well, which is the only reason it was caught -- a formula that is "right on the recording shape, wrong on other shapes"
cannot be covered by the self-check. So `__call__` now also runs an eager comparison for the **first few new shapes** (3 by default,
`TORCHINDUCTOR_DYNAGRAPH_VERIFY_SHAPES`):

- eager uses `inputs`, with the real shapes; replay reads the fixed-size copies. The two sides are computed independently.
- On a mismatch it reports `runtime-mismatch`, this region is **retired**, and afterwards normal re-recording resumes.
- Retirement works by `__call__` returning `None`; at that point `inputs` has not been cleared yet, so the caller takes over directly.
- Zero overhead in steady state: once enough shapes have been verified, eager is no longer run.

### Negative control: `dynagraph/probes/test_verify.py`

"No error reported" can mean there really is no problem, or that the verification is not running at all. So the
`cudaGraphKernelNodeSetGridDim` call in the planner is removed, and the grid stays at the value recorded at capture time:
when the shape shrinks, the extra blocks are masked off and the result is still correct; **when the shape grows there are not enough blocks, and nobody computes the tail**.

Three cases, all pass:

```
  intact planner                 shapes=(64, 96)   0 recordings, no fallback, numerics correct
  grid patch removed             shapes=(64, 96)   runtime-mismatch, recordings >= 1, numerics still correct
  shape exceeds input headroom   shapes=(64, 512)  input-too-large, recordings >= 1, numerics still correct
```

The third row is a **real bug** hit while writing this test, and it only shows up when the shapes are ordered **ascending**:

```
RuntimeError: The size of tensor a (8192) must match the size of tensor b (65536)
```

The static input buffers were allocated for the **recording shape**, and the recording shape is just "the first shape to arrive".
The arena handles intermediate buffers; inputs are not in the arena. Now the input buffers are over-allocated by `headroom`
(2x by default), the address does not change (kernels only patch sizes, not pointers),
and exceeding the headroom retires the region with `input-too-large`.

**Key point: retirement is a fallback, not a free pass for the result.** In all three cases the result handed to the caller must be correct,
and the test checks this separately. For the same reason, "the output changed" cannot be used as evidence that the sabotage took effect --
once it is caught the output is supposed to be correct anyway; the evidence is the tag and the re-record count.

### Fallback reasons are now countable

Every rejection goes through the same `_fallback(tag, detail)`, which prints one line
`DynaGraph fallback [tag]`. `usable()` was replaced by `unusable_reason()`,
which says which of the four conditions it is. Current tags:

```
no-wrapper-source  no-kernels  no-symbols  no-arena-outputs  no-symbol-args
symbol-not-an-argument  unsettled-config  unmodelled  planner-build
unevaluable-size  capture-failed  handle-mismatch  selfcheck-mismatch
runtime-mismatch  input-too-large  input-not-contiguous  exception
```

`dynagraph/probes/applicability.py` uses these tags to count hit rate and fallback distribution on real models
(each model runs in its own subprocess -- when DynaGraph goes wrong it is a segfault, which the parent process cannot catch with try).

## The handle-count check is structurally blind to extern kernels (2026-09-18 evening)

This one was dug up by an agent sent out to write a probe, and it matters more than the thing the agent was originally sent to measure.

`_capture` has a check: if the handle count != the kernel count, the whole graph is rejected, and the comment says
"a count mismatch means some kernel did not go through the static launcher". That direction holds; **the converse does not**:

- `extern_kernels.mm` is not a `CachingAutotuner` -- it never enters the kernel table;
- it also does not go through `StaticCudaLauncher` -- it never enters the handle table.

**Both sides are short by one at the same time**, so `len(handles) == len(kernels)` still holds and `_capture` happily returns True.
Measured by the probe: under two topologies the handle-count check **never fired once**; what actually blocked it was the
**self-check replay** on the recording shape (the tag is `selfcheck-mismatch`, not `handle-mismatch`) --
re-laying out the arena moved the Triton kernels' pointers away from the address cuBLAS writes to, and the fallback happened only because the numerics did not match.

In other words, extern kernels had so far been blocked by an **empirical numerical comparison**, not by a structural check.
The probe did not observe any wrong data, but the nature of the protection differs from what the code comment claims --
and "a node stuck at the recording shape" is exactly the kind of error this machinery should reject structurally above all.

Added `unreachable_launch(src)`: it scans the entry function and rejects with `extern-launch` on seeing `extern_kernels.` /
`torch.ops.` / `aten.` / `.item()`.
The comment in `_capture` was also extended with "the converse does not hold".

### This restriction determines the applicability

Today DynaGraph requires **every kernel in the graph to be Triton**. Consequences:

- GEMMs must be routed via `max_autotune_gemm_backends="TRITON"`, otherwise every model with a GEMM is blocked.
- **Conv nets are essentially excluded**: `max_autotune_conv_backends` defaults to `ATEN`, and PyTorch itself
  notes at `config.py:715` "Triton conv templates show wins on ROCm; on CUDA,
  profiling shows no gains on H100". That is, convolutions can only be served if they are routed to Triton,
  but that by itself is slower than cuDNN -- the speedup gained may not make up for the loss incurred.

So the 61 timm + 27 torchvision models in the list are all `extern-launch` under the default config.
This is not a DynaGraph bug; it is its current boundary. The real way out is to let the extern kernels' nodes
get device-updatable handles too, which is a separate engineering project.

## Six stress probes + three defects found by adversarial review (2026-09-18 evening)

Sent out 6 agents, each writing a probe that exercises an untested path, then sent one adversarial reviewer per probe
("if DynaGraph were replaced by an empty shell that does nothing, would this probe still print a pass?").
**Of the 6 probes, the review found real problems in 4** -- this layer is worth it.

### Three implementation defects, fixed

**1. The arena total was never checked.** The generated layout kernel clearly writes
`slot_off[n_slots] = acc;   // total, for the caller to check against the arena`,
but `__call__` only read the offsets and **never compared acc against the arena size**.
The arena is allocated once at build time for "the first shape to arrive", times headroom;
if some buffer grows faster than the input it writes out of bounds -- exactly the kind of silent error this machinery should reject above all.

The first version of the fix was in the wrong place (checked after replay, by which point the out-of-bounds write had already happened).
Now, **before** replay, the same layout kernel is run once on its own, the total is read back and checked,
and on failure the region is retired with `arena-too-small`. There is no need to re-implement the formulas in Python,
and no extra sync -- reading the offsets needs a sync anyway; it was just moved before replay.

**2. Argument index misalignment caused by specialized constexprs.** In `extract_kernel_table` two lists
are zipped by index, but with different filtering rules:

```
args = [k for k, v in sig.items() if v != "constexpr"]   # filters out all constexprs
pos  = run_args.get(gname)                               # the actual arguments of .run(...)
```

**Scalars that Inductor specialized into constants** are marked constexpr in the signature,
but **the call site still passes them** (only autotuned block sizes are not passed). An observed example:

```
signature ... ks0:i64, xnumel:constexpr, r0_numel:i32 ...   constants={'xnumel': 1}
.run(arg1_1, buf0, buf2, s77, 1, ..._r0_numel, stream=...)
```

`args` has length 5 and `pos` has length 6, so `r0_numel` read the literal `1` from `pos[4]`,
`_is_symbolic("1")` is false -> no SetParam is issued -> on replay the reduction length stays at the recorded value.
That case was luckily blocked by `Unsupported("no xnumel/XBLOCK")`, by chance, not by design;
**FixedGrid template kernels would be worse** -- `grid = pos[len(args):len(args)+3]`
slides by one the same way, directly picks the wrong three grid values, and nobody downstream would notice.

Now it is split into two lists: `args` is still the cubin parameter order (the index base for `cuFuncGetParamInfo`;
constexprs are not parameters), and `call_order` is the sequence the call site actually passes (non-constexprs
plus specialized constants, minus autotuned blocks). If the two disagree, the whole graph is rejected.

**3. The multi-partition fallback was filed under someone else's tag.** `_entry_source` returned
None on seeing >1 partition, so `_input_symbol_map` returned {}, and what finally got reported was `no-symbol-args` --
the reason does not match, so the statistics get skewed. First a dedicated `multi-partition` tag was added, ordered before the symbol checks;
**the restriction itself was removed on 2026-09-19**, see the next section.

### Probe problems found by the review (showing how easily the criteria themselves run idle)

- **The y axis of `probe_grid2d` is actually a constant.** What is actually generated is
  `{'XBLOCK': 4, 'YBLOCK': 64}` and ynumel is statically 64, so `gy` is a constant.
  `grid_type` is indeed `Grid2D` (I checked the dump myself), but **a symbol-driven y extent is still untested**.
  The `XBLOCK=32/YBLOCK=32` cited in the report does not match the file on disk; this is evidence drift.
- **`probe_multi_partition_fallback` also passes when idle.** Its three criteria
  (not served / recordings 8->8 / numerics correct) hold equally for "DynaGraph rejected every graph for some other reason";
  there is no positive control. A positive control was added later; later still the probe was flipped entirely
  and renamed `probe_multi_partition`, see the next section.
- **The disable path claimed by `probe_empty_and_tiny` was not proven.** The review pointed out that the kernels
  run sequentially on the same stream, so it cannot detect the failure mode "if it cannot be disabled it scribbles into the arena";
  what it actually proves is only that SetEnabled(0) and SetEnabled(1) are mutually consistent.
- **The target_hit of `probe_extern_kernel_fallback` was overstated** -- the "handle count mismatch" branch
  it was sent to verify can never fire. This was already fixed into a structural check in the previous section.

## Multi-partition: the restriction is removed (2026-09-19)

The reason for rejecting multiple partitions was wrong. The docstring of `_entry_source` said "the collected kernels
belong to two different cudagraphs, and this layer cannot tell them apart", but `compile_fx`, via
`recursively_apply_fns`, calls cudagraphify **once per partition separately**, and the `__name__` of the callable handed
to the hook is exactly `partition_0` / `partition_1`
(measured by `dynagraph/_probe_partition_id.py`). So each partition already has its own
runner, arena and graph; all that is needed is to read the right section by name.

This matters more than it sounds: **any graph break produces partitions** (`.item()`, `.cpu()`
round trips, data-dependent branches), so multiple partitions are the norm, not a corner case.

### But "reading the right section" is only a necessary condition

After taking the section by name, passing data between partitions exposed two real problems, both fixed:

**1. Inputs appear in the return value.** `partition_0` is `return (buf0, buf3, arg1_1)` --
`arg1_1` is passed through unchanged to the next partition. `_graph_outputs` used to pick only `bufN`,
so it returned a tuple shorter than what the caller unpacks. Now it returns the whole sequence, and the runner uses `out_order`
to record whether each position is an arena buffer or a passed-through input, and reassembles them in the caller's order.

**2. Buffers come in as arguments and are written in place.** `partition_1` is
`arg1_1, buf0, buf6, s77 = args`, then `buf1 = buf0; buf8 = buf1`,
and the kernel writes `buf8` as `in_out_ptr0`. It does not allocate a single buffer of its own. Three things had to change:

* allow `bufN` in the unpack line (`_input_symbol_map` used to require everything to look like `argN_1|sN`);
* `generate_pointer_patches` skips "buffers that come in as arguments" exactly like `argN_1`
  -- their address is given by the caller and is not in the arena;
* **write back**. The kernel writes to the copy held here, so the caller's tensor would hold stale data.
  Read/write intent is inferred from the kernel parameter names (`in_out_ptr*` / `out_ptr*`), and after replay the result is
  `copy_`'d back into the caller's tensor. This is the only place that tells read/write intent.

In addition, the `no-arena-outputs` criterion itself was wrong: a partition can perfectly well return only buffers
others gave it and own no allocations of its own, and that is still a graph worth serving (grid and scalars still need
patching). It was changed to `no-outputs`, which checks `out_order` instead of `outputs`.
The layout kernel's `int64_t sz[N]` does not compile when N=0 (zero-sized variable);
it was changed to `max(N, 1)`.

### Measured

`dynagraph/probes/probe_multi_partition.py` (the original `probe_multi_partition_fallback.py`
flipped entirely): both partitions are fully served, **recordings 8 -> 0**, and on four shapes the results are **bit-identical** to the control group
(same compile path, dynagraph turned off). 14/14 pass.

## extern kernel: the split fallback is implemented (2026-09-19)

`triton.dynagraph_partition_extern` (env `TORCHINDUCTOR_DYNAGRAPH_PARTITION_EXTERN=1`):
`scheduler.should_partition` returns a partition reason for any `ir.ExternKernel`. The upstream
`custom_should_partition_ops` **only applies to `FallbackKernel`**, while mm/addmm/bmm go through
`ExternKernelOut` (`ir.py:8269`), which is a different branch of `ExternKernel` and out of its reach -- the most important
category happens to be outside its coverage, so this had to be added.

`dynagraph/probes/probe_partition_extern.py`, **Inductor default config** (max_autotune_gemm off,
GEMMs go through cuBLAS), a 4-layer GEMM model: without splitting the whole thing falls back with `extern-launch`; after splitting, 1 -> 8
partitions, **all 8 sections served**, not a single extern call in the served sections, recordings 4 -> 0,
numerics bit-identical to the control group.

Pitfall hit: the first version of the probe called each shape only once, the control group's recording count was 0, and the criterion was useless --
on the first call cudagraph_trees only does an eager warmup; it records on the second call. Each shape has to be run twice.

This is the floor for coverage, not the end point: after splitting, a GEMM-heavy model may leave only the epilogues to Triton;
the graph is fragmented and the host overhead of the extern calls remains. The cost of fragmentation has not been measured yet. Stacked on top is the
tiering in `docs/notes/EXTERN.md` (SDPA -> GEMM -> conv); a failure at any tier falls back to here.

## extern kernel: staying in the graph (child-graph route) is implemented (2026-09-19 late night)

The split fallback measured 15x slower (`docs/notes/BENCH.md`), so this route keeps the cuBLAS calls **in the same graph**:
each `extern_kernels.<name>(` call site becomes a child-graph node during the main capture; the first time a new shape
appears, only those few extern kernels are captured into a small graph (the arena is already laid out for that shape, so the pointers are naturally correct),
and afterwards `cudaGraphExecChildGraphNodeSetParams` swaps the node. Switch: `triton.dynagraph_extern_child`.
Details and pitfalls are in sections five and six of `docs/notes/EXTERN.md`.

`dynagraph/probes/probe_extern_child.py`, Inductor default config, 8 addmm call sites: of 4 shapes, 3 are served by
**one graph**, bit-identical; M=64 gets an extra splitKreduce node from cuBLAS fp32 (the topology changed), so
**only that shape** is handed back upstream to record its own graph (`SKIP_SHAPE`), and the region is not retired. 17 regression tests are running.

Two design trade-offs, recorded as they are:
* **Topology changes are still a wall.** The current handling is to hand those shapes back upstream, which means those shapes go back to "one graph per shape".
  SWITCH (one body per topology, with the planner setting the condition) is the designed proper solution; not implemented.
* **Harvest is per shape.** Under a long-tail distribution every new shape needs one small capture (milliseconds, containing only the extern kernels);
  much cheaper than upstream re-recording the whole model, but not zero. The harvest table is capped at 1024.

### Not fixed yet

**An empty shape arriving first retires the region permanently.** When the first shape has a 0 dimension, kernels with an empty extent
are never launched during capture at all, so the handle count comes up short, and `handle-mismatch`
permanently rejects this region. This is a fallback, not wrong data, and low priority, but it is noted here.

## Two things the performance measurements exposed (2026-09-19, GPU 6 exclusive)

After the machine rebooted, GPU 6 was empty (0 MiB / 0% / 71W idle), so the four-way comparison was finally possible.
The shape stream comes from **real data**: the UniProt human proteome, 132016 sequences (lengths 32..1024,
median 314); 32 distinct lengths sampled at their real frequencies, 128 steps. `bench.py` checks on its own whether
the GPU is exclusive, and refuses to time if it is not clean.

### 1. Repeated launches: the table must be built per call site, not per kernel object

The first run was rejected outright by the positive control:

```
DynaGraph fallback [handle-mismatch]: 48 handles for 5 kernels
```

12 blocks reuse the same 5 kernels, but the graph has **48 nodes**.
`_parse_run_calls` built a `dict[kernel name -> actual args]`, so when the same kernel was launched multiple times
**only the last call site was kept**. The table therefore had 5 entries against 48 handles, and the whole graph was rejected.

**This explains why every earlier test passed while every real model was blocked** --
the test models were all single-block or had no repeated kernels, which happened to avoid this path.
Anything with repeated structure (every transformer, every deep network) gets stopped here.

The correct model is **one `.run(` = one graph node = one handle**. After switching the table to be built per call site,
the 12-block model was served normally (0 recordings, numerics vs re-record 2.7e-07), and as a side effect it is naturally ordered by launch
order, so sorting via `source_code.find()` is no longer needed.

### 2. One sync per step gives back everything cudagraph saved

The first real numbers measured after that fix were a **negative result**:

```
Hot (steady state, 128 steps):  A re-record 47.3 ms   C pad2max 31.3 ms   D DynaGraph 99.5 ms
```

**In steady state it is 2.1x slower than re-recording.** The cause: every `__call__` does
`self.slot_off.tolist()` to read the arena slot offsets back to the host -- **one device-to-host sync per step**.
The launch-bound tier measures exactly the launch overhead, and a single sync gives back precisely what cudagraph saved.

Changed to compute the prefix sum on the host (same formula as the layout kernel); the device copy is kept for the planner,
and the two are reconciled once at build time, rejecting with `layout-mismatch` if they disagree.
**Zero syncs per step**, and the arena out-of-bounds check becomes genuinely preventive as a side effect --
checking only after the layout node in the graph has run means the out-of-bounds kernel has already written.

This is a **deliberate duplication of the formula**, which is why that reconciliation is mandatory; you cannot rely on "it looks the same".

### 3. Still retired under natural arrival order

```
DynaGraph fallback [input-too-large]: arg 0: 117248 > 71680
```

D is reported in two variants: `max-first` (largest shape first) is served; **natural order retires**.
Today DynaGraph needs the largest shape to arrive first -- the static input buffer is sized at headroom
times the first arriving shape, while the real arrival order is random. This is not an experimental setup that can be worked around; it is the cost of the
limitation "inputs are not in the arena", and it should be measured rather than avoided. The way out is to put inputs in the
arena too, which requires the planner to also patch input pointers; that is the next piece of engineering.

### All timing numbers are void: host CPU load 93

After the caching change, rerunning changed **every** number, including the baselines I had not touched at all:

```
        this round    previous round
A re-record  330.8 ms     47.2 ms
C pad2max     66.6 ms     31.9 ms
D            350.8 ms    124.8 ms
```

The same untouched baseline differs by 7x. Checking `uptime`:

```
load average: 93.53, 84.77, 64.82
```

**The launch-bound tier measures CPU launch overhead, and the CPU is shared across the whole machine** --
other people's jobs run on other GPUs, but their host processes still compete for CPU. `gpu_is_idle()` only checked the GPU,
not the CPU; that is a methodological hole.

**So every timing number before this section is void, including the two optimization attributions based on them:**

1. The first concluded that the bottleneck was the per-step `slot_off.tolist()` sync -- switching to a host-side prefix sum
   was **actually slower** (99.5 -> 124.8 ms).
2. The second found that `_eval_int` runs `ast.parse` every time, dozens of buffers x every step --
   two levels of caching were added (AST + per-shape layout cache).

The second change is **correct in itself** (dozens of `ast.parse` calls per step are objectively wasteful),
but **no claim can be made about how much it gains, because it cannot be measured**. Both times were searching for signal in noise.

Lesson: **"removing something that looks expensive" is not the same as finding the bottleneck.** On a machine with interference,
first confirm the measurement itself is reproducible before talking about optimization; if it is not reproducible, reach for a profiler instead of continuing to guess.

`bench.py` has been changed: each variant repeats N times and **takes the minimum** (interference can only make a run slower, never faster),
reports the max/min spread, prints a warning if it exceeds 1.3x, and writes the load average into the results.
Rerun once the machine is idle.

### The timing methodology needs to change once more

The current "cold" number is swamped by **compile time** (A cold 10007 ms / hot 47 ms, a difference of more than 200x),
and compilation is the same fixed cost for A/C/D. The right split is to warm up on the largest shape first to get compilation out of the way,
then measure: first pass over the stream = recording cost (A records one graph per new shape, D records zero), second pass = steady state.

## Four coverage gaps, all showing up as "safe fallback" (2026-09-19)

The probes that were sent out hit four paths that had never been exercised. What they have in common: **none of them raises an error**;
they just make it look like "DynaGraph does not take effect on my model".

| Gap | What blocked it | Consequence |
|---|---|---|
| Table deduplicated by kernel name | `handle-mismatch: 48 handles for 5 kernels` | **Every model with repeated structure** |
| `Grid2DWithYZOverflow` not modelled | `unmodelled: grid type ... is not modelled` | Every 2D-tiled pointwise |
| numel specialized to a constant treated as "no numel" | `unmodelled: no xnumel/XBLOCK to size a grid` | A batch of normal kernels |
| 0-dim scalar tensor treated as "parse failure" | `unmodelled: buf3, which no allocation owns` | Graphs that use scalar intermediates |

**1. One `.run(` = one graph node = one handle.**
`_parse_run_calls` used to build `dict[kernel name -> actual args]`; when the same kernel was launched multiple times, only the last
call site was kept. 12 blocks reusing 5 kernels -> table with 5 entries, 48 handles.
This explains why every earlier test passed while every real model was blocked --
the test models were all single-block or had no repeated kernels, which happened to avoid it.

**2. What Inductor actually emits is `Grid2DWithYZOverflow`, not plain `Grid2D`.**
When the number of blocks in y exceeds 65535 it is folded into z:

```
raw = ceil(ynumel/YBLOCK);  div = ceil(raw/65535)
gy  = div ? ceil(raw/div) : 0;   gz = div
```

My `_GRID_AXES` only had the idealized `Grid2D`, so the **actually common form** of 2D tiling
was always a blank. `probe_grid2d` "passed" only because its ynumel happened to be a static
64, making `gy` a constant -- a nominal hit only. Now `probe_grid2d_symbolic` specifically requires
that **the y-range expression actually references ctx** for it to count.

**3. A grid with constant numel is constant to begin with; it does not need patching.**
A scalar specialized to a constant is not in the argument table, yet I treated it as "no numel" and rejected outright.
Now `consts` records its literal value at the call site, and the grid can still be determined.

**4. `empty_strided_cuda((), (), torch.float32)` is a legitimate 0-dim tensor.**
`tup()` returned `[]` both for a genuine `()` and for "could not parse", and both were dropped together by a single
`if not sizes: continue`. Now `None` means not understood, and `[]` means 0-dim.

### One methodological point

All four of these were exposed only because **probes hit uncovered paths**, not by reasoning from reading the code.
And they were all masked by a correct "safe fallback" -- without fallback-tag statistics,
we would not even know that "this many things are being blocked".

## Next steps
## Next steps

1. ~~Automatic planner generation~~ Done.
2. ~~Hook into cudagraph_trees~~ Done (see the previous section).
3. **End-to-end gain measurement**: what needs measuring now is the speedup **through the public switch**,
   four-way comparison recapture / eager / pad2max / DynaGraph. **Must have the GPU exclusively.**
4. **Applicability**: on a set of real models, collect DynaGraph hit rate and each fallback reason.
   Fallback reasons all have explicit `log.info` now and can be collected directly.
5. **Hard problem #1, unbacked SymInt**: sizes like `nonzero`/`x[x>0]` are not pure functions of symints;
   the planner has to read them from device memory. This is the real technical core and has not been touched.
6. **Unverified risk: buffers whose stride depends on M.** A kernel's stride argument is patched to a new value,
   but the buffer is laid out with the old stride -- the current `buffer_layouts` recomputes strides from env,
   but no test case has covered it.
7. **The affine lookup-table optimization in planner_codegen.py** has not been merged into the module yet (the module is currently all switch).
8. **CUTLASS scheduler parameters** (`FastDivmodU64`, derived from M) are not yet recomputed in the planner.

## Repository status

- The fork is at `tie-pilot-qxw/pytorch`, working branch `dynagraph` (**do not push to main**).
- The local `dynagraph` base is older than the fork, to match the already compiled `.so`;
  pushes go through `dynagraph-push` (same base as `fork/dynagraph`) via cherry-pick.
- There are already forks of `sglang` and `vllm-omni`, which will be useful for end-to-end validation.

## Fixed slots: pointers written only once (2026-09-19, late night)

Result of slicing up the real planner in `docs/notes/BENCH.md`: of the 43 us when the shape changes every step, the 95
pointer SetParams account for half. Pointers have to be patched every time because the arena is **re-laid out** for each shape --
the layout kernel computes each slot's size from the current symbols and then does a prefix sum, so slot offsets move with the shape, and the buffer
addresses baked into kernel nodes have to move with them.

The change: offsets are fixed at build time. Each slot is sized at the largest buffer it holds at the recording shape x `headroom`
(`fixed_slot_offsets`), and the layout kernel degenerates into writing constants (it is kept so the host-side
prefix sum still has something to reconcile against, and the flag protocol stays unchanged). ctx gains one more cell, "pointers dirty": it is 1 after capture,
and the planner runs the pointer-patch section of the `switch` only when it is 1; after the first replay the host clears it to zero,
and from then on the planner only patches grids and scalars.

Cost: memory goes from "the maximum over shapes of the total" to "the sum of per-slot maxima" (under the same headroom the totals
are similar, but shapes with a different distribution -- e.g. B grows while L shrinks -- can blow out one particular slot while the total is still sufficient).
A shape that blows out a slot reports `arena-too-small` (it now says which slot, how much is needed, and how much there is); the region retires
and upstream records per shape. The "fixed total, varying split" space in `test_arena_e2e.py` is the worst case:
each slot is sized at its maximum across the whole space, and the total is twice the original.

What did not change: harvest on the child-graph route runs on arena views, and the views now use fixed offsets, so the small graphs
harvested for each shape bake in the same set of base addresses, which conveniently removes the "different shape, different address" concern.

**Results (same night)**: 17 regression items pass (among them, the 3e-5 in `probe_partition_extern` was traced to the probe using
a different compilation as its reference, where runtime autotune on the shared GPU picked a different reduction config; see
`docs/notes/BENCH.md`); the newly added `probe_update_persistence.py` shows that device-side updates persist across launches, so
the premise of "pointers written only once" holds. How many us this saves has not yet been measured on an exclusive GPU.
