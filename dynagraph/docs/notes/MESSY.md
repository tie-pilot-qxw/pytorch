# Messy-code scan: coding patterns from AI paper repos, and how much of each DynaGraph absorbs (2026-09-19, late night)

The fourth axis of the goal. `dynagraph/probes/probe_messy_code.py`: 18 typical "messy" coding patterns, each run once with
dynagraph off / on (child route, extern calls kept in the graph), shape stream `[64, 200, 128, 333] x 2 passes`,
autotune pinned on both sides (the lesson from `docs/notes/BENCH.md`). No timing, coverage only. GPU 4 (shared, correctness).

| case | pattern | dynamo graphs | breaks | regions asked/served | recordings off->on | fallback tags | on-off diff |
|---|---|---|---|---|---|---|---|
| item_branch | branch on `.item()` | 5 | 2 | 3/3 | 6->1 | extern-topology | 7e-7 |
| int_from_tensor | slice with `int(mask.sum())` | 4 | 2 | 4/4 | 6->**0** | - | 0 |
| int_from_tensor_unbacked | same as above + `capture_scalar_outputs` | - | - | - | - | **upstream NameError** (crashes even with DG off) | - |
| inplace_input | `x.mul_(); x[:,0]=1` | 2 | 0 | 1/0 | 0->0 | selfcheck-mismatch | 0 |
| cat_loop | append in a loop, then cat | 2 | 0 | 1/1 | 3->1 | extern-topology | 0 |
| bool_mask | `x[x[:,0]>0]` | 3 | 2 | 3/3 | 7->3 | **input-too-large** | 2e-6 |
| bool_mask_unbacked | same as above + `capture_dynamic_output_shape_ops` | 2 | 0 | 3/1 | 6->3 | no-symbol-args, no-symbols | 0 |
| cpu_roundtrip | `.cpu().numpy()` round trip | 2 | 0 | 1/1 | 3->**0** | - | 0 |
| pad_to_max | pad to the max length | 2 | 0 | 1/1 | 3->**0** | - | 0 |
| arange_pos | `arange(L)` positions | 2 | 0 | 1/1 | 3->**0** | - | 0 |
| transpose_across_break | receives a transposed input after the break | 4 | 2 | 2/1 | 7->4 | **input-not-contiguous** | 0 |
| dropout_train | dropout in train mode | 2 | 0 | 1/0 | 0->0 | no-symbol-args | random |
| size_arith | scalar arithmetic `/ x.shape[0]` | 2 | 0 | 1/1 | 3->1 | extern-topology | 1e-7 |
| einsum | `einsum` | 2 | 0 | 1/1 | 3->1 | extern-topology | 1e-6 |
| tolist_loop | loop over `tolist()` | 4 | 2 | 1/1 | 3->**0** | - | 0 |
| tril_mask | `tril(ones(L,L))`, buffer quadratic in L | 2 | 0 | 1/1 | 3->1 | **arena-too-small** | 0 |
| step_counter | `self.calls += 1` | **8** | 0 | 7/7 | 0->0 | - | 2e-6 |
| stash_on_self | `self.cache = h` | 2 | 0 | 1/1 | 3->1 | extern-topology | 0 |

18 cases: fully served with 0 recordings **6**, partially served 9, fully rejected 2 (both are ones upstream does not cudagraph either: in-place modification of
inputs, train-mode RNG), upstream itself crashes 1.

**Rescan after doing item 1 (on out-of-bounds, fall back only once and rebuild on the larger shape)**: bool_mask recordings 3 -> **0**
(asked 4 = rebuilt once), tril_mask 1 -> **0** (asked 2); fully served with 0 recordings **6 -> 8**, partially
served 9 -> 7. The "exceeds input headroom" case in `test_verify` was changed accordingly to expect 0 recordings.

## How to read it

* **A graph break by itself is not a problem.** `.item()`, `tolist()`, `.cpu()` cut the model into several segments; each segment
  is an independent region and is served on its own (item_branch 3/3, int_from_tensor 4/4). The cost of a break is
  on the host side; DynaGraph cannot do anything about it, and does not make it worse either.
* **Numerics:** in the non-random cases the on/off difference is 0 or 1e-7~1e-6. The non-zero ones are all cases where some shape was handed back to upstream recording
  (extern-topology / input-too-large); for that one shape, cuBLAS may pick a different algorithm when captured inside the graph than when captured in the small child
  graph, which is what [docs/METHODOLOGY.md](../METHODOLOGY.md) means by "a separate compile is not a numerics reference".
* **extern-topology appears 6 times**, and it is always the same thing: cuBLAS fp32 addmm has an extra
  splitKreduce node at M=64, the child graph cannot be swapped in, and that shape is handed back to upstream. The SWITCH route (`docs/notes/EXTERN.md`)
  is meant exactly for this; once it is done, all 6 of these rows will become 0 recordings.

## What this scan says needs fixing (ordered by how common it is in "messy code")

1. **A first shape that is too small retires the region permanently** (done, see above) (bool_mask's `input-too-large`, tril_mask's
   `arena-too-small`). The input buffers and the arena are both allocated at "first arriving shape x headroom"; data-dependent
   lengths and quadratically sized buffers easily exceed that, and once it is exceeded the region retires and every shape is recorded from then on. Fix:
   on out-of-bounds, fall back only this once -- record this shape the upstream way, then **rebuild** the runner on it
   (buffers sized for the new, larger shape), a bounded number of times (3). The cost equals one upstream recording, after which it can serve again.
2. **Non-contiguous input** (transpose_across_break). After the break, the next segment receives a `.transpose` view,
   which today is rejected outright. The kernel is specialized for that stride, so copying the input into a large buffer with the same stride would do.
3. **Recompile storm** (step_counter: a Python counter makes dynamo recompile every time, 8 graphs, 7 builds).
   Every build runs nvcc once for the planner, yet the wrapper source is identical -- planner compilation is now
   cached by source (`_module_cache`), so 7 compilations become 1.
4. **unbacked** (bool_mask_unbacked: `no-symbol-args`). There is no backed symbol among the kernel arguments;
   the length is an unbacked symbol such as `u0`, which the planner cannot read. This was the top-ranked hard problem on the open-problems list at the time,
   not something to fix in passing this time.
5. **Upstream bug**: with `capture_scalar_outputs=True` + `dynamic=True` + `reduce-overhead`,
   the wrapper Inductor generates has `buf2` undefined; it crashes the same way with dynagraph off; with `mode="default"`
   it is fine (both (30,)/(101,) are correct), so the problem is in how the cudagraph-partition codegen handles unbacked symbols.
   Noted; not work for here.

## Rescan after SWITCH landed (2026-09-19, late night)

| case | recordings off->on | fallback tags | notes |
|---|---|---|---|
| item_branch | 6->**0** | - | was extern-topology, SWITCH absorbs it |
| cat_loop | 3->**0** | - | same as above |
| size_arith | 3->**0** | - | same as above |
| einsum | 3->**0** | - | same as above |
| bool_mask | 7->**0** | input-too-large | fully served after one rebuild |
| stash_on_self | - | **illegal memory access** | was extern-topology; crashes after re-capture, under investigation |
| tril_mask | 3->0 | arena-too-small, **runtime-mismatch** | retired after the rebuild + re-capture combination, under investigation |
| rest | unchanged | | |

Fully served with 0 recordings **8 -> 13**, partially served 2, fully rejected 2, upstream crashes 1. Both new problems are on the path where "rebuild
(`REBUILD`) and re-capture (SWITCH) happen at the same time"; conv with 12 batches also crashes in the same place
(`docs/notes/EXTERN.md` section 8).

## Rescan after `cudaGraphUpload` landed (2026-09-19, exclusive GPU 0)

| case | recordings off->on | fallback tags | notes |
|---|---|---|---|
| stash_on_self | 3->**0** | - | the illegal memory access from the previous round is gone (root cause in `docs/notes/EXTERN.md` section 8.2) |
| tril_mask | 3->**0** | arena-too-small | re-captured after one rebuild, all 8 calls match bitwise; the previous round's runtime-mismatch is gone |
| item_branch / int_from_tensor / cat_loop | 6/6/3->**0** | - | in the previous round they were pushed into OOM on shared GPU 4 by someone else's memory use; they pass on an exclusive GPU |
| rest | unchanged | | |

Fully served with 0 recordings **13 / 18** (item_branch, int_from_tensor, cat_loop, bool_mask, cpu_roundtrip,
pad_to_max, arange_pos, size_arith, einsum, tolist_loop, tril_mask, step_counter, stash_on_self),
partially served 2 (bool_mask_unbacked: no-symbol-args; transpose_across_break: input-not-contiguous),
fully rejected 2 (inplace_input: runtime-mismatch, the input is modified in place; dropout_train: no-symbol-args),
upstream crashes 1 (int_from_tensor_unbacked: Inductor's own `NameError: buf2`, unrelated to DynaGraph).
The two "under investigation" items in the previous round's table are not separate problems; both are the same thing: the upload at first launch overwriting device-side updates.

The remaining 5, by difficulty: input-not-contiguous (input after a transpose; either copy it or put the stride into symbols),
inplace_input (the output aliases the input; the check does not match because the eager run already modified the input -- a copy must be saved before verify),
the two unbacked ones (no-symbol-args: unbacked sizes do not enter the wrapper arguments), dropout's seed (the philox
offset is a scalar that changes on every call; it could be fed in as a symbol).

## In-place writes: not just messy code, BatchNorm is hit too (2026-09-19, exclusive GPU 7)

Tracing `inplace_input`'s runtime-mismatch down, the root cause is not in the check but in the **extra eager runs** the runner does:
warmup x3 at build, one self-check, one check per new shape, one per harvest, and each of them actually executes the wrapper.
If the region writes its inputs in place or writes static buffers, the state gets advanced a few extra steps. `dynagraph/probes/probe_inplace.py`
has three cases; off / on each run the same stream of 4 lengths x 2 passes (8 calls), comparing outputs, the inputs after the run, and the buffers:

| case | before | now |
|---|---|---|
| `x.mul_(0.5); x[:, 0] = 1` then Linear (eval, writes its input in place) | served 0, selfcheck-mismatch (the self-check's eager run modified the input and then the replay ran, so the two sides saw different data) | served 1, 0 recordings, inputs match bitwise |
| Linear -> **BatchNorm1d(train)** -> Linear (running stats are static inputs and are written in place) | **served 1, 0 recordings, no fallback tag at all, but `num_batches_tracked` 8 -> 20, running_mean off by 1.2e+01** | served 1, 0 recordings, running stats match bitwise |
| `self.n.add_(1)` buffer counter | served 0, and n = 15 instead of 8 (the extra runs before retiring had already written it) | served 1, 0 recordings, n = 8 |

The second row is a silent error in standard training code: upstream cudagraph_trees allows static inputs to be written in place (that is how BN
gets into the graph); we absorbed the region and the check passed too (the check compares outputs, and the outputs are correct; what is wrong is the buffers), with no
signal at all. Fix: the extra eager runs fall into two kinds -- those that only need a reference value (warmup, self-check, check) run on **copies of the
written tensors** (`_eager_args`); those that must see the real storage (harvest has to capture addresses) **put the written tensors back** after running
(`_unwritten`). "Which tensors get written" = Inductor's own `mutated_input_idxs` (which sees in-place extern
calls) union the kernel-table entries whose `in_out_ptr`/`out_ptr` point at inputs (the fallback for the partition path).

The cost is only on the non-steady-state path: one extra clone (of the written tensors) per check, and one extra clone + copy-back per harvest.
Steady-state replay is unchanged. On coverage, `inplace_input` goes from fully rejected to fully served -- upstream does **not put** in-place writes to non-static inputs
**into a cudagraph** (recordings off = 0), so this class consists of regions that only DynaGraph serves.

Full-table rescan after the in-place fix (exclusive GPU 6): fully served with 0 recordings **14 / 18** (inplace_input added), partially served 2
(bool_mask_unbacked, transpose_across_break), fully rejected 1 (dropout_train: no-symbol-args -- tracing it down, it comes from the
training wrapper's argument naming, see the next section), upstream crashes 1 (int_from_tensor_unbacked).

## Rescan after training naming + random ops + views + non-contiguous (2026-09-19, exclusive GPU 6)

| case | recordings off->on | fallback tags | notes |
|---|---|---|---|
| dropout_train | 0->**0** | - | was no-symbol-args; training wrapper naming + seed op run on the host + RNG state put back (`docs/notes/EXTERN.md` section 11) |
| transpose_across_break | 7->1 | - | was input-not-contiguous; both regions with symbols are served, the remaining 1 is a symbol-less static region recorded by upstream (section 12) |
| bool_mask_unbacked | 6->3 | no-symbols, unevaluable-size | was no-symbol-args: after the naming was relaxed it got as far as size evaluation; the unbacked `u0` is not in env, so it is still partially served |
| rest | unchanged | | |

**Fully served with 0 recordings 15 / 18**, partially served 2 (the two unbacked / static-region cases), **fully rejected 0**, upstream crashes 1.
From 8/18 (before SWITCH) -> 13 (SWITCH + upload) -> 14 (in-place writes) -> 15 (training naming / random / views / non-contiguous).
All that remains is unbacked (`bool_mask_unbacked`, `int_from_tensor_unbacked`, the latter being Inductor's own
NameError).

## Rescan on the afternoon of 2026-09-21 (exclusive GPU 6)

**Fully served with 0 recordings 17 / 18**, partially served 1 (`transpose_across_break`: both regions with symbols are served,
the remaining 1 recording is a symbol-less static region, recorded by upstream), **fully rejected 0**, did not enter cudagraphify 0.
The 15 in the previous entry is an older number; the two unbacked cases (`bool_mask_unbacked`, `int_from_tensor_unbacked`)
both came in after unbacked tier 2 (symints stay in GPU memory, the planner reads them inside the graph).

The same afternoon's "kernels whose table cannot be read are demoted to opaque sites" (EXTERN.md section 24) had **no effect** on this table:
it is identical row by row before and after the change. None of the cases here failed because a kernel table could not be read; what trips them is the Python-side coding patterns.
The coverage that change actually bought is in `probe_tma.py host`: regions with host-side TMA descriptors went from "rejected whole" to "served".

The next step along the same line, "patch host-side TMA descriptors as parameters" (EXTERN.md section 27), likewise had no effect on this table:
none of these 18 cases writes a `TensorDescriptor`; they all use ordinary PyTorch operators.
The coverage that change bought is in `probe_tma_real.py` (MXFP dequantization in OpenAI's `triton_kernels`):
from "becomes an opaque site, re-harvested once per new shape" to "stays in the main graph and is patched via kernel-node parameters".
