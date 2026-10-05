# P0 coverage survey harness

Answers one question: **run `torch.compile` over a pile of real models -- how many fit into cudagraph as a whole graph, and for the ones that don't, which rule do they get stuck on?**
This is the basis on which DynaGraph was started; the methodology mirrors Table 1 of PyTorch 2 (ASPLOS'24).

## Running

```bash
# use_main.sh sets up the venv, NVIDIA's preset version variables and LD_LIBRARY_PATH; source it first
docker exec -d dynagraph bash -lc '
  source /workspace/pytorch-main/dynagraph/setup/use_main.sh && cd /workspace/pytorch-main/dynagraph/survey &&
  mkdir -p ${DG_OUT:-/tmp/dynagraph_out} &&
  CUDA_VISIBLE_DEVICES=5 python runner.py \
    --list models_all.txt --out ${DG_OUT:-/tmp/dynagraph_out}/results_capture.jsonl --jobs 6 --fake \
    > ${DG_OUT:-/tmp/dynagraph_out}/survey_capture.log 2>&1'
```

**Use `docker exec -d`**: when the session drops, the child processes in the container get killed, and a full run takes more than half an hour.
Generate the list first with `python make_lists.py --out models_all.txt`, and view the results with
`python report.py ${DG_OUT:-/tmp/dynagraph_out}/results_capture.jsonl`.
Results and logs go under `$DG_OUT` (default `/tmp/dynagraph_out`); `run_all.sh`, `run_both.sh`, `run_real.sh`, `finalize.sh` and `rerun_failed.sh` do the same.

| Flag | Effect |
|---|---|
| `--fake` | Fake tensors throughout. Zero tensor memory; only a ~520 MiB CUDA context per process. **Needs the self-built main**; the container's bundled 2.11 reports a fake mode mismatch |
| `--fast` | Stop at codegen and skip cubin compilation; ~4.1 s -> 2.4 s per model. **The cost is losing the layer-3 data** (the whole graph giving up cudagraph happens after `compile_to_module`) |
| `--jobs N` | Number of concurrent subprocesses. Estimate GPU memory as N x 520 MiB |
| `--timeout S` | Per-model timeout. The paper used 30 minutes; a timeout counts as a failure |
| `--one SPEC` | Run only one, for debugging |

First run `python validate_lists.py models_all.txt` to pass the list through on CPU;
models that cannot be built are written to `failed_models.txt`, so you don't discover them only during the real run.

## Files

| File | What it does |
|---|---|
| `runner.py` | Process isolation + the four required compile settings. One subprocess per model |
| `instrument.py` | Four hooks that record the decisions and their reasons |
| `models.py` | model spec -> `(model, args, kwargs)`. Supports `tv:` `tvdet:` `timm:` `hf:` `builtin:` |
| `make_lists.py` | Generates the spec list from the lists in `benchmarks/dynamo` |
| `validate_lists.py` | Trial construction on CPU; filters out models that cannot be built |
| `report.py` | jsonl -> coverage table |
| `lists/` | Copies of the paper-era (v2.1.0) lists, more complete than the local checkout |
| `models_extra.py` | Eight data-dependence probes (GNN aggregation/sampling, molecular dynamics, NMS, sparse voxel, varlen packing, MoE, LLM decode) |
| `figures.py` | Figure 1 (partition distribution), Figure 2 (reason breakdown), output as PDF |
| `fake_vs_real.py` | Checks whether fake and real tensors give the same conclusions (result: only good for scouting) |
| `acceptance.py` | Five acceptance checks once the self-built torch is installed |
| `repro_relu6.py` | Self-contained minimal repro of the `nn.ReLU6` bug, can be handed upstream as is |
| `batch_attribution.py` | Batch attribution: runs every model with `cpu_ops` twice, splitting fixable from structural |
| `filter_by_size.py` | Filters the list by parameter count (GPU memory is a hard constraint in real-tensor mode) |
| `run_real.sh` / `finalize.sh` | The two full runs / final tables, figures and attribution |
| `rerun_failed.sh` | Reruns the crashed and timed-out models with real tensors |

Scripts starting with an underscore are one-off diagnostics, kept because they record how the conclusions were reached:

| Script | What it answered |
|---|---|
| `_cpuops_probe.py` | Which source ops the `cpu_ops` nodes come from |
| `_minrepro.py` / `_relu6.py` / `_relu6_dyn.py` | Narrowed it down to the three trigger conditions of `nn.ReLU6` |
| `_attribution.py` | How many remain after swapping out ReLU6 (MobileNet goes to zero, SSD300 loses none) |
| `_purity2.py` | Whether the CPU computation chains read the GPU (the 91-vs-2 split) |
| `_dyn_ablation.py` | Whether turning dynamic off brings it to zero, separating fixable from structural |
| `_batch_effect.py` | Batch size does not affect the conclusions, but it exposed that backward graph compilation is sporadic |
| `_extra_probe.py` | Partitioning of the eight data-dependence probes |

`builtin:` is five synthetic probes used to calibrate the harness itself:
`plain` (no data dependence) / `datadep` (boolean indexing) / `nonzero` / `cpuop` (drops to CPU midway) / `item`.
After changing the harness, run these first; if the numbers don't match, the harness is broken.

## Inference and training are two scenarios; run them separately

By default `runner.py` **runs forward only**. When the backward graph gets compiled depends entirely on AOTAutograd's laziness;
the same model may or may not compile it with a different batch (measured: batch 2/32 have a backward, batch 256 does not).
So the data from the default round is the **inference scenario**.

The training scenario needs `--train` (in `runner_next.py`, to be swapped in once the current round finishes),
which reduces the output to a scalar and calls `backward()`, forcing the backward graph out as well.
The paper reports inference and training separately, and the "messy open-source training code" the user cares about is mostly a training scenario,
so both rounds must be run, with separate tables.

## Two settings that never change

| Setting | What happens without it |
|---|---|
| `force_disable_caches = True` | The second compile of the same model hits the cache and never goes through the scheduler; counts silently drop to 0 |
| `mode="reduce-overhead"` | Without cudagraph enabled, `should_partition` semantics flip (every node "stays out of the graph"); the counts are meaningless |

## Two more settings to run both ways

Data dependence has two checkpoints, and a single survey run only sees one of them, so **run both configurations**:

```bash
OUT=${DG_OUT:-/tmp/dynagraph_out}; mkdir -p $OUT
python runner.py --list models_all.txt --out $OUT/results_capture.jsonl --jobs 6 --fake
python runner.py --list models_all.txt --out $OUT/results_default.jsonl --jobs 6 --fake --no-capture
```

| Configuration | What it answers | Cost |
|---|---|---|
| PyTorch default (`--no-capture`, i.e. PyTorch's shipped setting) | How much layer 1 blocks: how many models get Dynamo graph breaks because of data dependence | Inductor-level partition data is all 0 |
| Capture (runner.py's default behavior) | The truth at layer 2: once data dependence reaches Inductor, how many segments it is split into, and why | Some models fail to compile outright with `GuardOnDataDependentSymNode` |

Running only the former leads to the wrong conclusion "there is no partition problem"; running only the latter misses the fact that "under the default config the problem is hidden one layer up".

## Is fake mode trustworthy? Checked: yes

Running the survey with fake tensors has a fundamental risk: **what if it systematically changes the partition results?**
Then Figure 1 would be wrong. This cannot be assumed; it has to be checked.

`fake_vs_real.py` starts two independent subprocesses for the same model, one fake and one with real tensors, and compares 7 key fields
(nodes that cannot enter the graph, reason distribution, partition count, segments that gave up cudagraph, Dynamo graph breaks).

| Model | Result |
|---|---|
| ResNet-18 | 7/7 agree |
| ViT-B/16 | 7/7 agree |
| BERT (MaskedLM) | 7/7 agree |
| `builtin:nonzero` | 7/7 agree |
| MobileNet-V2 | Fake side segfaults (exit code 139); no data to compare |

**29 fields agree and 6 disagree, and all 6 come from the MobileNet-V2 crash**, not from a disagreement in conclusions.

### ...but this conclusion was later overturned: fake is only for scouting

That round only checked "do they agree when both sides run through", and the four models picked all have **0 CPU nodes**.
After filling that gap (`tvdet:ssd300_vgg16`, 91 `cpu_ops`, and it did not crash under fake):

| Field | fake | real tensors |
|---|---|---|
| Nodes that cannot enter the graph | 92 | **101** |
| `cpu_ops` / `device_copy` / `unbacked_binding` | 91 / 1 / **0** | 98 / 2 / **1** |
| Segments per `graph_partition` call | `[2]` | `[2, 2, 2, 1, 1, 0, 1, 1, 1]` |
| Segments that gave up cudagraph | 1 | **4** |

**Under fake only 1 subgraph was compiled; with real tensors there are 9.** Dynamo graph breaks produce multiple subgraphs;
fake crashes after the first one, so the following 8 are never compiled, and the criterion "`graph_partition` ran, so it succeeded"
treated this incomplete data as complete.

Together with another known issue (the fake-mode segfaults hit precisely the models that have CPU nodes), the conclusion is:

> **Fake can only be used for scouting; the final data must use real tensors.**
> Records from `--fake` runs now carry the `data_maybe_truncated` flag.

Validating a measurement method means asking three things, and missing any one of them can invert the conclusion:
do the numbers agree when both sides succeed; are the failures correlated with the phenomenon being measured; can the "success" criterion mistake incomplete data for complete data.

## In fake mode the model must be constructed on **meta**

This is the trap most likely to fool you.

Constructing the model directly inside `FakeTensorMode` makes **weight initialization produce unbacked symbols**.
`trunc_normal_` in `torch/nn/init.py` goes into `_no_grad_trunc_normal_`, which in fake mode
produces a `u0`, and compilation later blows up on

```
GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u0, 1)
Caused by: (utils/_device.py:122 in __torch_function__)
```

ViT, Swin, MaxViT, ConvNeXt, Inception and BEiT all hit it. **It looks as if these models have a data-dependence problem,
but it is introduced entirely by the harness itself**; recording them as failures would be a wrong conclusion.

The check is direct: construct without compiling and look for unbacked symbols in `ShapeEnv`.
ResNet-18 has 0 symbols after construction; ViT's construction itself throws a data-dependent error.

The right way is to **construct on the meta device** (init on meta is a no-op with no real computation),
then enter `FakeTensorMode` and use `to_empty(device="cuda")` to turn the parameters into fake cuda tensors:

```python
with torch.device("meta"):
    model, args, kwargs = build(spec)
fm = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
with fm:
    model = model.to_empty(device="cuda")
    compiled = torch.compile(model, dynamic=True, mode="reduce-overhead")
    compiled(*to_cuda(args), **to_cuda(kwargs))
```

After this change, ViT's result is **1 segment in forward, 1 segment in backward, whole graph cudagraphable** -- the exact opposite of the earlier record,
"data dependence causes compile failure".

> The ViT number was later checked against real tensors (`fake_vs_real.py` 7/7 agree), so it is trustworthy.
> ConvNeXt / Inception / BEiT from the same batch **only have fake numbers**,
> and fake systematically undercounts (see the section on fake being only for scouting), so for those, wait for the real-tensor round.

As an aside: **do not** drop the `ShapeEnv`. Measured: without `ShapeEnv` it is worse; not even ResNet-18 gets compile data.

## In non-fake mode you cannot build first and then call `.to()`

`models.build()` **does not call `.to(device)` itself**; the caller must wrap it in `with models.device_ctx():`.

Reason: under `FakeTensorMode` the model parameters are `FakeTensor`s as soon as they are created, and a later `.to()` goes through
`torch.utils.swap_tensors`, which fails directly with

```
RuntimeError: _apply(): Couldn't swap Conv2d.weight
```

In the first full run, all of the first 27 torchvision models failed on this. The `torch.device` context makes the parameters
get created on the target device from the start. Fake mode takes the meta route above instead of this one.

## Pitfalls of the three hooks

**`should_partition` is called ~3 times per node** (once each from `reorder_for_minimizing_partition` and other places);
deduplicate by `(graph_id, node_name)`, otherwise counts are 3x too high. Measured: 12 before dedup, 4 after.

**Do not use `counters["inductor"]["cudagraph_partitions"]` as the partition count**:
it is only incremented when there is more than 1 segment, so an empty counter does not mean the graph was not split -- it may be exactly 1 segment; it also accumulates across forward/backward.
Hook `Scheduler.graph_partition` instead; it returns `(partitions, signatures)`.

**`log_cudagraph_skip_and_bump_counter` must be rebound in four modules separately**:
`cudagraph_utils` / `output_code` / `compile_fx` / `cudagraph_trees`; patching only the definition does nothing.

**Reason strings only exist on main**: in the container's bundled 2.11, `should_partition` only returns `bool`;
on main it is `-> str | None`. With 2.11 you can only count "how many nodes cannot enter the graph", not "why".

## Output fields

`snapshot()` reports the data in three layers:

- **Layer 1, Dynamo**: `dynamo_graph_breaks` / `dynamo_break_reasons`
- **Layer 2, Inductor partitions**: `n_nodes_not_cudagraphable` / `partition_reason_counts` (normalized slugs) /
  `n_partitions_observed` (one entry each for forward and backward) / `n_partitions_max`
- **Layer 3, whole graph given up**: `cudagraph_skips` / `skip_reason_slugs`
- **Self-check**: `scheduler_api` (`str|None` means main) / `should_partition_raw_calls` / `skip_hook_rebound`

## Two operational lessons

**Give each model a timeout of at least 900 seconds.** Compilation is much slower than you would expect; measured:

| Model | Compile time |
|---|---|
| ResNet-18 | 66 s |
| BERT (MaskedLM) | 67 s |
| TIMM ResNet-50 | 101 s |
| ConvNeXt-Tiny | 176 s |
| Inception-v3 | 273 s |
| DPN-107 | 353 s |
| Swin-T / MaxViT-T | **> 500 s (has timed out at 500 s)** |

`--fast` saves ~40%, but loses the layer-3 data; only use it when you are sure you don't need the "whole graph gives up cudagraph" numbers.

**`pkill -f` does not kill cleanly; get the PIDs with `pgrep` and `kill -9` each one.**
In `docker exec ... pkill -9 -f "runner.py"`, the command line of pkill's own `bash -lc` also contains that
string, so it kills itself first (exit code 137) and the target processes keep running. Reliable form:

```bash
docker exec <container> bash -lc 'for p in $(pgrep -f "survey/runner.py"); do kill -9 $p; done'
```

After killing, always confirm with `nvidia-smi --query-compute-apps=pid,used_memory` that the GPU memory was really released;
other people are still running on that card.

## GPU discipline

The 8 cards on this machine always have other people's jobs running. Only use **cards 4 and 5**; before using one, check
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` to see whose processes are there,
and after a run confirm you left no GPU memory behind. In `--fake` mode each process only takes a ~520 MiB CUDA context, with zero tensors and zero kernels.
`torch.cuda.get_device_properties()` itself does not create a context, so it is free.
