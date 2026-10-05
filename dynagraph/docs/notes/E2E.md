# End-to-end workloads (from 2026-09-23)

Xinwei, 09-23: "We now have to run end-to-end workloads, instead of wasting time on fake workloads we synthesized ourselves."
The targets are the kind that the earliest survey found to have large launch overhead and especially many shapes (FINDINGS.md "workload survey"):
GNN neighbor sampling, molecular / MD, point-cloud sparse convolution, recommendation.

## 1. Environment

Installed and importable in the torch 2.15 (fork) venv: torch_geometric 2.8, torch_scatter / sparse / cluster /
spline_conv, pyg-lib (built from source, `/workspace/_deps/src`, ninja, see `_deps/install/build_pyg.sh`),
e3nn 0.6 (MACE uses the e3nn 0.4.4 in `/workspace/_deps/mace_site` layered on top), mace-torch 0.3.16, nequip 0.19.1,
schnetpack 2.2, torchmd-net (from source), spconv-cu126 2.3.8 + cumm (wheel, does not depend on the torch ABI), ogb, ase.

Data (`/workspace/_deps/data`): ogbn-arxiv, QM9, MD17 aspirin, MD22 (double-walled nanotube with 370 atoms,
buckyball-catcher). KITTI drive 0093 is being downloaded (`data/kitti`).

## 2. harness (`dynagraph/e2e/harness.py`)

Modes: `eager` / `compile` (dynamic=True) / `trees` (reduce-overhead, upstream records one graph per shape) / `dg` /
`pad` (pad everything to the global max, dynamic=False + reduce-overhead, one static graph -- the K=1 baseline defined in FINDINGS).

**Each mode runs three segments**; this is the most important change this round:

| Segment | Contents | Why |
|---|---|---|
| warm | the first N batches | compilation and first-time recording all happen here |
| new | the remaining batches, all shapes unseen | real training resamples every epoch, MD gets a new neighbor list every frame; **this is the steady state** |
| replay | the new segment run again as-is | all shapes seen; only a reference for the hit case, never happens in practice |

The very first version was "run the same set of batches three times"; the second and third passes were all hits, and DG looked as fast as pad -- that was an illusion.

Switches: `AMP=1` (wrap forward in bf16 autocast), `GEMM=deepgemm` (`e2e/dgemm.py`) / `GEMM=triton`
(Inductor Triton templates), `DGLOG=file` (DynaGraph log), `DGTIME=method,...` (accumulate wall-clock time per method,
printed per segment), `DGPROF=new|replay` (cProfile one segment), `DGHKEY=1` (on each harvest, print which part of the hkey changed).
`vs compile` in the report is the loss difference against the same set of kernels run without cudagraph; it shows the error DG itself introduces.

## 3. Status of each workload

| workload | shapes | result (dg mode) |
|---|---|---|
| GraphSAGE + NeighborLoader, ogbn-arxiv training (`e2e/sage.py`) | 80 batches, 80 distinct | **all served by DG**, 0 fallbacks, 0 recordings; with AMP + DeepGEMM all GEMMs are in tier 3 |
| SchNet, QM9 training (`e2e/schnet.py`) | 60 batches, 60 distinct | **all served by DG**, 0 fallbacks; all GEMMs are DeepGEMM (including the Gaussian basis with K=50, stride padded) |
| MACE energy+force inference, MD17 aspirin (`e2e/mace_md.py`) | 60 frames, only 5 distinct | all served. Too few shapes, trees also records only 5 graphs; not the target scenario |
| MACE, MD22 double-walled nanotube, 370 atoms | 60 frames, 53 distinct; 57/59 adjacent frame pairs differ | **all served**; with `GEMM=triton` 0 extern calls, new shapes ~= replay (7.9 vs 7.9 ms, shared GPU) |
| Point-cloud sparse convolution inference, SECOND VoxelBackBone8x, KITTI drive 0093 (`e2e/pointcloud.py`) | 80 frames, 80 distinct 4-level tuples; per-level max/mean 1.23-1.26 | **all served**, 0 fallbacks. Dynamo cannot trace into the spconv ops, so it is written following the torchsparse dataflow: the kernel map is precomputed on the real frames with spconv `get_indice_pairs`, then per layer gather -> DeepGEMM `m_grouped_bf16_gemm_nt_contiguous` (27 groups, each padded to 128 rows, rows with layout -1 scattered to a discard row) -> scatter-add -> BN -> ReLU. Grouped GEMM requires K%64, so all channels are padded to 64 |
| ESM-2 35M MLM training, UniProt human proteome (`e2e/esm.py`) | token budget 8192, length bucketing: 40 batches, 34 distinct (B, L), B 8..123, L 66..1024 | 638/639 served (forward failed the check on the first shape and fell back once, to be investigated); attention uses flex (Triton, tier 1), linear layers use DeepGEMM |
| Recommendation | -- | not done |

trees baseline: SchNet backward segfaults under upstream trees (see Section 5); it runs after the workaround.

## 4. DynaGraph / registration-layer changes this round (uncommitted)

As "what changed -> why":

1. **The launch structure of inline sites takes part in graph selection** (`_apply_inline` / `_inline_exec` / `_combo`).
   For some M, DeepGEMM picks a 2-CTA cluster (TMA multicast), and `SetParams` cannot change the cluster. Xinwei pointed out
   that "different clusters should be treated as different variants" -- before that I had tried changing the SM count / modifying the library to turn off multicast, which was the wrong direction. Now each inline
   site's (number of kernels, cluster of each) goes into the host-mode exec combination key, just like child-graph topology, and a new structure is captured once at that shape.
   SAGE: only 1 extra graph. In SWITCH (device) mode it still falls back; not done.
2. **`recorded(..., allocates=False)` uses raw stream capture** (`_capture_launch._record_raw`): it bypasses torch
   CUDAGraph / the private pool, 81 us vs 292 us (one DeepGEMM call).
3. **DeepGEMM lowering** (`e2e/dgemm.py`, outside the repo): bf16 `aten.mm/addmm` -> `dgemm::mm_out` (writes into a given output);
   the a/b major-ness for `bf16_gemm_nt` is decided from the strides, so one function covers all four transpose combinations. Measured: the only requirement is that non-unit strides are 16B-aligned;
   dimensions are arbitrary -- misaligned operands are copied by Inductor into a layout whose stride is padded to a multiple of 8, and outputs are allocated with the padded stride.
   - `compiled_dims` lists only the static dims: the default `"nk"` bakes dW's K (= number of nodes) into the kernel, **one nvcc
     run per new node count (~5 s/step)**. This is information only the compiler has, so it is exactly what the lowering should pass.
   - Process-level initialization: DeepGEMM's `DeviceRuntime` allocates resident tensors on its first call; if that lands in the trees pool it reports
     "tensor in pool not tracked"; `install()` calls it once up front.
4. **Check tolerance** (`_same_values`):
   - Regions that contain tier-2 child-graph sites are no longer compared bit-for-bit: the cuBLAS inside the graph and the eager cuBLAS can pick different algorithms depending on alignment / workspace.
   - The tolerance is set by the **lowest precision** among the region's input and output tensors (fp32 gradients also carry the rounding of bf16 activations), 8 eps x max|ref|,
     with a floor of 1e-3; alternatively, an overall norm relative error <= the same bound. Planner errors (stale sizes / pointers) are off by the magnitude of the values themselves, so they are still caught.
     MACE's per-edge scalars (a sum over 64 channels with cancellation) differ by 1e-3 of the max in fp32; the elementwise criterion falsely rejects them, the norm criterion does not.
5. **Geometric growth of the arena / input copies** (`dynagraph_grow` default 1.0 -> 1.25): when growing by the exact size every time, each arena relocation invalidates all
   harvests of that lane; on MD22, over 40 frames it grew 24 times, already-seen shapes were re-harvested too, and replay took 51 ms per step. After the change it grows 1 time and replay takes 10.5 ms.
6. Logging: print the names of extern sites at capture time.

Regression `_regress_quick.sh`: 12/12 pass.

## 5. Upstream issues / workarounds

- With `specialize_float=False` (the default under dynamic=True), the Python float `coeff` in SchNet's GaussianSmearing becomes a 0-dim input;
  forward hands over a CUDA tensor, but backward is compiled for CPU (`rand_strided(..., device='cpu')`), so under reduce-overhead
  the backward C++ kernel dereferences a device pointer on the host -> segfault (same for trees and DG). The harness sets
  `specialize_float=True` to work around it. Worth reporting upstream separately.
- A pure inference loop must call `torch.compiler.cudagraph_mark_step_begin()` every step; otherwise DG opens a new lane for every call (it grew to 15 in MACE),
  and every lane rewrites all pointers. trees uses the same convention.

## 6. Timing (preliminary, **contaminated**)

GPU 4 was idle when the run started (util 0%); halfway through SAGE, yichen's `bench/pal` started on it, and the run was stopped once this was noticed. Only the order of magnitude is meaningful:

| Median ms per step | eager | compile | trees new shapes | **dg new shapes** | dg replay | pad |
|---|---|---|---|---|---|---|
| SAGE (AMP+DeepGEMM) | 2.79 | 1.37 | 4.23 | **4.37** | 1.38 | 0.97 |
| SchNet (AMP+DeepGEMM) | 11.39 | 7.60 | 12.78 | **14.90** | 7.56 | 7.34 |

MACE MD22 (`GEMM=triton`, shared GPU, not comparable across runs): compile 11.7, dg new shapes 7.9, replay 7.9.

**Reading: on hits DG is in the same range as compile/pad; in the real setting every step is a new shape, and there DG measured slower than compile in this run.**
In a new-shape SAGE step, DG itself spends ~2.2 ms extra (`DGTIME`):

| Item | ms/step |
|---|---|
| `_packed_launches` (DeepGEMM raw capture + bind plan for each new shape, ~5 times/step) | 1.38 |
| `_site_operands` (building operands of inline sites) | 0.24 |
| `_make_plan` (arena layout) | 0.19 |
| The rest (apply, host_step...) | ~0.4 |

For comparison: a whole pad step is 0.97 ms. When MACE uses Triton GEMM (no extern calls), new shapes cost almost nothing extra, which shows that the planner itself is not the bottleneck;
**the bottleneck is "asking the library for its launch once per new shape"**.

### 6.1 Clean timing (2026-09-23 21:47, GPU 0 / GPU 7, 0 processes from anyone else throughout)

DeepGEMM has been switched to describe. Two rounds per workload, the second round with the modes in reverse order; median ms per step in the new-shape segment (two rounds):

| | eager | compile | trees new shapes* | **dg new shapes** | dg replay | pad |
|---|---|---|---|---|---|---|
| SAGE (88 batches, 88 distinct) | 3.01 / 2.95 | 1.63 / 1.34 | 5.41 / 4.16 | **2.09 / 2.54** | 1.14 / 1.22 | 0.94 / 0.83 |
| SchNet (80 distinct) | 10.47 / 10.12 | 7.22 / 11.56 | 10.42 / 11.68 | **8.38 / 8.33** | 5.24 / 5.16 | 2.89 / 2.92 |
| MACE MD22 (100 frames, 90 distinct) | 18.08 / 25.66 | 4.16 / 4.13 | 5.99 / 5.41 | **5.59 / 5.00** | 4.14 / 3.88 | -- |

\* The first time a shape appears, trees only does a warmup (eager); it records on the second occurrence. Its "replay" column (SAGE 5.9, SchNet 22-23, MACE 12.4-12.8) is the actual recording cost.
compile's SchNet first-round mean is 52 ms: compile mode has no prepare step, so when a new shape hits a new DeepGEMM config it JITs inside the step.

How to read it:
- DG replay is the fastest of the dynamic modes; but **on new shapes DG is still slower than compile** (SAGE +0.5 to 1.2 ms, MACE +0.9 to 1.4 ms); for SchNet the two compile rounds differ too much to tell.
- **pad-to-max is the fastest across the board**. SchNet pad 2.9 ms vs DG replay 5.2 ms -- about 2x faster even at the same shapes, which shows the gap is not only about cudagraphs:
  dynamic=False kernels (specialized to static shapes) are themselves much faster. For this kind of load max/mean is only ~1.07, so pad wastes very little.
- So to beat pad, two things must be solved together: the per-new-shape host overhead (DG's own Python, ~1 ms) and the quality gap of dynamic-shape kernels.

### 6.2 Point cloud + how much there is to gain (GPU 2, a simulation job running, util ~2%, magnitudes only)

| Median ms per step | eager | compile | trees new shapes | dg new shapes | dg replay | pad |
|---|---|---|---|---|---|---|
| Point cloud (80 frames) | 2.80 | **0.60** | 2.35 | 1.21 | 0.68 | 1.28 |

Per step in compile mode (`e2e/gputime.py`):

| | wall clock | host dispatch | GPU kernel total | upper bound |
|---|---|---|---|---|
| Point cloud | 0.57 | 0.56 | 0.41 | 1.4x |
| SAGE | 1.78 | 1.76 | 0.41 | 4.3x |
| SchNet | 6.31 | 6.28 | 2.50 | 2.5x |

After compile, the point cloud is basically no longer launch-bound (49 kernels, GPU 0.41 ms); SAGE and SchNet are still heavily host-bound,
pad captures most of the gain (0.9 / 2.9 ms), and DG captures it only on replay.

**DG's new-shape overhead grows linearly with the number of opaque sites** (`DGTIME`, per new shape):

| | sites/step | describe | build operands | packing + SetParams etc. | `_make_plan` | total extra over replay |
|---|---|---|---|---|---|---|
| Point cloud | 11 | 0.18 | 0.17 | ~0.3 | 0.04 | ~0.55 |
| SAGE | 10 | 0.25 | 0.21 | ~0.45 | 0.16 | ~1.3 |
| SchNet | 89 | 1.85 | 1.33 | ~1.7 | 0.95 | ~5.7 |

Each site costs ~60 us, of which the library's own describe is only ~20 us; the rest is our Python (Launch objects, ctypes parameter blocks,
cuda-python's SetParams, operand views). The GEMM shapes of SchNet's 6 interaction layers are identical from layer to layer, so describe could have been shared by geometry.

Also: when the arena grew, the operand / parameter caches of inline sites were not invalidated (the key does not contain the arena base address) -- the same shape coming again would get a launch for the old arena.
Fixed (`_grow_arena` clears `_inline_ops/_inline_prepared/_inline_templates` and each exec's `inline_applied`).

### 6.3 ESM-2 variable-length training (GPU 7, no one else's processes throughout)

Padding to the global (123, 1024): the token count is 15.8x the average and the attention work (B*L^2) is 26.6x -- this is the "superlinear x wide distribution" cell.

| Median ms per step | eager | compile | trees new shapes | **dg new shapes** | dg replay | pad |
|---|---|---|---|---|---|---|
| ESM-2 35M | 41-43 | 20.4 | 51.8 (139.9 while recording) | **37.5** | 19.1 | 146 |

Per step in compile mode: wall clock 23.0, GPU kernels 14.7 ms (812 kernels) -> upper bound 1.57x.

- pad is 7x slower, and trees records once per new shape (234 times): neither the static scheme nor per-shape recording is usable here; **the only competitor left is compile**.
- DG replay 19.1 ~= compile; new shapes 37.5 -- each step has ~216 DeepGEMM inline sites (fwd 72 + bwd 144), and the ~60 us of
  Python overhead per site (Section 6.2) eats the entire gain and more. The 12 layers have exactly the same shapes, so describe by geometry would need only ~8 calls per new shape.
- To beat compile: bring the per-new-shape host overhead under ~5 ms (216 sites -> ~20 us/site, or share per layer + patch parameters in C++);
  in a real training loop that does not sync every step, the host overlaps with the previous step's GPU work, so the target ~= max(host, 14.7 ms).
- Note: the DG OOM in the GPU 0 round was caused by bojin's sglang filling the card midway, not by DG; DG peak memory is 21.9G (compile 2.5G)
  -- the cost of the arena / lanes / per-exec state, which also needs a look.

### 6.4 Minimal prototype: how much the DG new-shape path can gain if taken all the way (ESM, 2026-09-23)

Counts per new-shape step (`e2e/proto_counts.py`): 9 region calls, 222 inline sites, 558 Triton nodes, 494 output tensors.

Unit costs (`e2e/proto_units.py`, C++ / the library itself, CPU time):

| What must be done | Unit cost |
|---|---|
| DeepGEMM describe (Python call, real ESM fwd/dX/dW shapes) | 8.7 us |
| C++ `cudaGraphExecKernelNodeSetParams` | 0.57 us/node |
| C++ building an output tensor on the arena and handing it back to Python | 1.3 us |
| `cudaGraphLaunch` | ~2 us |

-> DG new-shape lower bound ~= 1.8 ms (describe shared per layer) to 3.3 ms (not shared). Currently it is 27-29 ms.

Host breakdown of one compile-mode step (`e2e/proto_phases.py`, GPU 2 shared, two runs): `call` of the Inductor-compiled regions (the part DG replaces)
17.1 / 16.6 ms; the rest (HF's eager code between the 8 graph breaks, AOT/autograd, optimizer) 8.5 / 17.5 ms.

Estimate: with DG at the lower bound, per step ~= max(rest of host + 2-3 ms, GPU 14.7 ms) ~= 15-21 ms, **1.3-1.7x faster than compile** (upper bound 1.57x),
which pad / trees cannot reach here. Assumptions: the Python glue outside DG stays unchanged, and DG does not fall back.

**Blockers (ahead of performance)** (both fixed, see Section 6.5):
1. The ESM forward main region (145 kernels, 72 sites) fails the check on the first new shape: several outputs are entirely 0 (each layer's bf16 weight copy (480,480),
   q/k (B,20,L,24), etc.) -> a real bug, and the region falls back to upstream. So most of DG's forward in Section 6.3 actually ran through the trees warmup; those numbers must be re-measured.
2. Sporadic illegal instruction (once each in SchNet and ESM; does not reproduce with a sync after every call); the kernel was not located.

### 6.5 Both blockers fixed (2026-09-24, GPU 0 idle throughout)

Both are really the same kind of problem: DG mistook things that were not its own for its own.

1. **harvest fed uncomputed captured outputs to later eager kernels** (the indices of `aten.sort` -> out-of-bounds index_put, which also corrupts other memory).
   Fix: after capture, `_harvest` backfills them with the values from the warm-up run via `_copy_values` before continuing.
2. **Stale kernel function handles** (root cause of the forward outputs being all 0 and of illegal instruction 715): the kernel table's `funcs` were collected from the autotuner's `compile_results`
   **before** warmup and included configs that were not selected; after autotuning, `_release_static_launchers_except`
   unloads those, and the driver later hands the same CUfunction address to a kernel loaded by the DeepGEMM JIT. `dg_init` matches nodes by function,
   so the position of the second `_to_copy_1` in each layer got paired with a DeepGEMM node (grid 132, params `(nullptr, M=1623, ...)`):
   in recorded mode nobody patched the real `_to_copy_1` node -> output all 0; in describe mode `dg_step` wrote Triton params into the DeepGEMM
   node -> 715. Address reuse depends on allocation order, hence intermittent (1-2 out of 3 runs).
   Fix: after warmup, re-fetch `funcs` from the variants the autotuner still holds (`kernel_funcs(obj)`).
   Lesson: **a CUfunction handle is not an identity**; it is meaningful only while its module is alive.

After the fix: small config (2 layers), 3/3 runs with 156/156 served and 0 fallbacks, loss delta vs eager 2.0e-4 (compile 2.4e-4); `_regress_quick.sh` 12/12.

Full ESM-2 35M (100 batches, 87 distinct (B,L), GEMM=deepgemm, GPU 0):

| Median ms per step | eager | compile | dg new shapes | dg replay |
|---|---|---|---|---|
| ESM-2 35M | 49.8 | **19.9** | 46.8 | 25.5 |

DG 1635/1635 served, loss delta vs compile 2.4e-4 -- **correct now, but 2.4x slower than compile on new shapes**; peak memory 22.9G (compile 2.5G).
The 37.5 in Section 6.3 was measured with the forward fallen back; it does not count.

cProfile (new-shape segment, inflated by the profiler) per step: `_apply_inline` 45 ms (of which `_prepare_inline` is 20 ms: describe ~6,
`_packed_launches` packing ~12; self ~15 ms, mostly cuda-python per-node SetParams + building parameter structs), `_make_plan` 17 ms
(layout expressions `_eval_int`/`go`), `_site_operands` 8 ms. ~353 launches are re-prepared per step. No re-capture for new inline structures (all 15 captures are builds).
-> Consistent with the finding of Section 6.4: the bottleneck is entirely in the Python host path, lower bound 2-3 ms.

### 6.6 New-shape host overhead optimization (2026-09-24, GPU 7 shared, host under load, commit 677ef29e48)

Step by step (median per step of DG `__call__`, full ESM-2 35M, DGTIME taken as the per-step median; `harness.py`'s DGTIME now reports both the mean and the median;
the mean is pulled up by kernel loading / JIT the first time DeepGEMM sees a given M, so do not use the mean):

| Change | New shapes | Replay |
|---|---|---|
| Starting point (after the correctness fixes) | 40.6 | 19.7 |
| Inline cache bounded by shape (previously 4096 entries < 87 shapes x 216 sites, so it was cleared every time) + expression compilation covers min/max/floor/comparisons | 35.6 | 8.7 |
| Group by signature: layers sharing the same GEMM are built only once, the rest shift addresses | 31.2 | -- |
| One C call patches all nodes (`dg_inline_set`) | 25.0 | 9.3 |
| Signature recipe (static part computed only once) + batched output creation in C++ (5.5 -> 2.4 us each, 916 per step) | 27.3* (noise) | 7.6 |
| Same-hkey fast path (whole batch of arrays cached per exec x hkey) | 23.4 | 4.2 |
| Layout evaluated once (`_make_plan` 4.4 -> 1.9) | **20.7** | **4.1** |

Full step (compile run first in the same process): new shapes DG 33.9 vs compile 26.0; **seen shapes DG 18.9 vs compile 23.6**. Numerics: vs compile 2.1e-4, 1635/1635.

Pitfalls hit while grouping: backward's operands are activations saved by forward -- different offsets into the same storage, so putting the storage offset in the signature means no group can form at all;
the same geometry can be written in many ways (an offset of `(0)+(0)`, a reshape equal to the original layout), so comparing strings does not group them either -> compare by the **numerically effective geometry** at the current shape.
Address ownership is decided by each tensor's own range (with shared storage, using the storage range attributes addresses to the wrong tensor); an address that falls inside a storage but belongs to no name is not shared.

The remaining 20.7 ms for new shapes: `_apply_inline` 11 (`_packed_launches` 2.1 for representative-site describe, `_site_signature` 1.9, `_apply_wanted` 1.8,
members 0.9, operands 1.0, self ~3), `_harvest` 2.4 (sort site), `_make_plan` 1.9, `_host_step` 1.2, the rest of `__call__` ~4.

### 6.7 Per-call work moved to a C++ runtime (2026-09-24, commit 82f1e4bb3e)

`torch/_inductor/dynagraph_rt.cpp` (load_inline, compiled once per machine) + C generated per region (layout / geometry of outputs and site operands)
+ the library's C describe (ABI v1, `_capture_launch.py`; DeepGEMM's is in `_deps/patch_dgd_c.py`, byte-for-byte identical to the Python describe).
Python is left with only build, the cross-check for the first `verify_shapes` shapes, per-structure capture, and harvest. The dgemm lowering now targets `dgemm::nt_out` (= bf16_gemm_nt as is, with b as an (N,K) view).

Full ESM-2 35M (GPU 7 shared, compile run first in the same process):

| Median ms per step | New shapes | Seen shapes |
|---|---|---|
| compile | 26.7 | 26.0 |
| **DG** | **23.0** | **18.1** |
| DG `__call__` | 10.1 (previous version 20.7) | 2.9 |

loss delta vs compile 2.5e-4, 1635/1635. GraphSAGE: DG 1.61 vs compile 2.17 ms (new shapes), loss delta vs eager 2.5e-3 (compile 2.4e-3).

Not yet in C++: small device-update regions (1-2 kernels, go through Python), sort regions with harvest, capture of new cluster structures.

### 6.8 Back to Triton GEMM (no changes to third-party libraries; 2026-09-24, commit f0e7639038)

Xinwei: we cannot modify other people's operators (DeepGEMM's describe sink / C entry point are both local patches), and Triton GEMM is more convenient. Pure-Triton regions
also get a C++ program; when the runtime is present, small regions switch to host updates. Results (full ESM-2 35M, GPU 7):

| Median ms per step | New shapes | Seen shapes |
|---|---|---|
| compile + Triton GEMM | 17.4 | 17.2 |
| DG + Triton GEMM | 20.4 | 18.9 |
| (reference) compile + DeepGEMM | 26.7 | 26.0 |

- The earlier "DG beats compile" result came mainly from the host overhead of the DeepGEMM custom-op path (one Python dispatch + the library's heuristics per GEMM, 216 times per step), not from DG itself.
- Seen-shape breakdown (gputime.py): compile wall clock 17.8 / host dispatch 16.2 / GPU 14.6 (864 kernels); DG wall clock 19.7 / host 13.3 / GPU 15.1 (901 kernels).
  DG spends less host time but has a longer wall clock -> suspected cause: each region first finishes patching on the host and then launches the full graph (+ re-upload of execs whose parameters changed), and the GPU sits idle during that time;
  compile issues kernels one at a time, pipelined with execution. The extra 37 kernels / 0.5 ms of GPU time may be copies of inputs into the store and write-backs. To be confirmed with nsys.
- Host time outside DG: ~10 ms (HF's eager code between the 8 graph breaks, dynamo/AOT wrappers, autograd, AdamW), which DG cannot touch.

## 7. Next steps

1. Get the DeepGEMM launch without capturing: within the same variant, changing M only changes 1 integer parameter + the dims bytes of 3 TMA descriptors
   (`_wf_dgemm_record_cost.py`: bytes 36-37, 56, 65 of parameters 4/5/6; the address is in plaintext at byte 0). Re-encode the descriptors on the host
   for the new M (`cuTensorMapEncodeTiled`, us-scale); the variant (config / cluster) is bucketed by M, and the library is queried only once at a bucket boundary.
   Alternatively, modify `/opt/vllm/.deps/deepgemm-src` (single TU, header-only JIT) to add a describe sink.
2. Make `_make_plan` / `_site_operands` fast (0.4 ms per new shape).
3. Point clouds (real KITTI frames, grouped GEMM), recommendation.
4. When a truly idle GPU is available, do interleaved timing (see `docs/METHODOLOGY.md`).
