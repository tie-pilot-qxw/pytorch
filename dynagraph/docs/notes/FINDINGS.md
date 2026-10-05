# 2026-09-17 Summary of findings

The full derivation and data are in `FEASIBILITY.md`, the environment in `SETUP.md`, and the harness in `dynagraph/survey/README.md`.
This page holds only the conclusions.

## Environment

In the container, **torch `2.15.0a0+git71b3251`** and **torchvision `0.30.0a0`** were installed from source;
always `source dynagraph/setup/use_main.sh` when you enter. The hard reason we had to build it ourselves:
**the partition reason strings only exist on main** (`should_partition -> str | None`).
NVIDIA's packaged 2.11 only returns a `bool`, so with it you can only count "how many nodes cannot go into the graph", not "why".

Four pitfalls that will lead you astray (none of their error messages point to the real cause) are recorded in `SETUP.md`:
NVIDIA's preset `PYTORCH_BUILD_VERSION`, a disconnected `docker exec` session interrupting ninja,
`LD_LIBRARY_PATH` making the new torch load old libraries, and the self-built torch breaking the preinstalled torchvision and taking transformers down with it.

## Four methodology corrections -- each one would have inverted the conclusion

| # | Problem | Symptom | Fix |
|---|---|---|---|
| 1 | Constructing the model inside `FakeTensorMode` | Weight initialization produces unbacked symbols; ViT / Swin / ConvNeXt / BEiT all report "data-dependent failure". **It looks like a problem with the models, but the harness introduced it** | Construct on `meta`, then convert to fake |
| 2 | fake-mode crashes **have a selection bias** | The segfaults hit exactly the models that have CPU nodes (triggered by `DeviceCopy`); the whole MobileNet family crashes, so the "models with problems" vanish from the statistics | Use real tensors instead |
| 3 | fake mode also **silently truncates** | For models with several subgraphs only the first subgraph is compiled; `ssd300_vgg16` shows 1 under fake and 9 with real tensors | Same as above |
| 4 | Forward only | Whether the backward graph gets compiled depends entirely on AOTAutograd's lazy mechanism; the current data is the **inference scenario** | Add a `--train` switch and report the two runs separately |

### Generalization: to validate a measurement method, ask four questions

None of today's four corrections was "the code is wrong"; in every case **the measurement setup itself quietly changed the thing being measured**,
and each one was found by asking one more question, not by running more models. These four questions must be walked through again every time the harness changes:

| # | Question | Today's pitfall | How to check |
|---|---|---|---|
| 1 | Do the numbers agree under a different way of measuring? | fake vs real tensors | Measure the same model both ways and compare the key fields |
| 2 | **Are the failures correlated with the phenomenon being measured?** | The fake segfaults hit exactly the models with CPU nodes | Look for a shared trait in the list of failures |
| 3 | **Can the "success" criterion mistake an incomplete result for a complete one?** | fake compiling only the first subgraph also counted as success | Add an independent completeness signal (here `ran=True`) |
| 4 | **Does the measurement itself disturb the thing being measured?** | Hooking `should_partition` (about 3 calls per node) | Run once with the hook and once without, and compare time and results |

Question 4 has already been checked (`_hook_overhead.py`):

| Model | With hook | Without hook | Partition result | Decision calls |
|---|---|---|---|---|
| ResNet-18 | 36.6 s | 37.3 s | `[1, 1]` on both | 657 |
| MobileNet-V2 | 105.4 s | 108.2 s | `[2, 2]` on both | 2226 |

With the hook it is actually slightly faster (within noise), and the partition results are identical. Even with MobileNet-V2's 2226 decision calls,
the overhead is within noise. The conclusions are not affected by the measurement.

Question 1 has an easy-to-miss aspect: **comparing agreement only on the samples where "both sides succeeded" is not enough**;
that is exactly the gap question 2 fills. Today we first asked only question 1 and concluded "fake is trustworthy",
and only later found that the failures were biased and that the success criterion also let incomplete results through.

## A bug that can be reported upstream: `nn.ReLU6`

Under `dynamic=True`, **floating-point** attributes of an `nn.Module` are symbolized, materialized as CPU tensors, and then copied back to the GPU.
All three conditions are required: dynamic shapes + access through a module attribute + floating point. `nn.ReLU6` hits all three
(it is `Hardtanh(0.0, 6.0)`, and its forward reads `self.min_val` / `self.max_val`).

| Written as | Non-GPU nodes |
|---|---|
| `nn.ReLU6()` / `nn.Hardtanh(0.0, 6.0)` | 4 |
| `nn.Hardtanh(0, 6)` with integers / `F.relu6()` / `clamp` / dynamic turned off | 0 |

Because of this, MobileNet-V2 accumulates 71 CPU nodes. A self-contained repro is in `dynagraph/survey/repro_relu6.py`.
Per the repository's AI policy, reporting it requires confirmation first.

We searched upstream: **there is no exactly matching issue**. The two closest are different problems:
`pytorch/pytorch#123471` (Inductor's type handling that converts integer literals to float when an op takes a SymInt)
and `pytorch/pytorch#155193` (the API design of `ReLU6` inheriting from `Hardtanh` rather than `Module`).
A search is not exhaustive; before reporting, it is best to search the issues once more with the keywords
`hardtanh min_val symbolic float attribute cpu scalar`.

## Attribution breakdown: what can be subtracted and what cannot

Replace `nn.ReLU6` with the equivalent literal form, leave everything else unchanged, and re-measure:

| Model | As is | After the replacement | Conclusion |
|---|---|---|---|
| `mobilenet_v2` | 72 | **0** | All from the upstream bug |
| `ssd300_vgg16` | 99 | **99, not a single one fewer** | All structural (VGG has no ReLU6) |
| `ssdlite320_mobilenet_v3_large` | 155 | 106 | 49 subtracted, **104 structural remain** |

Broken down by origin, the 97 nodes of `ssd300_vgg16` are all **anchor generation**:
`repeat` to tile the anchors, `add` to add offsets, `clamp` to clip to the boundary, `cat` to concatenate the levels.
They land on the CPU because they depend on the spatial size of the feature map, and under dynamic shapes that is a symbol.

> **This is good news for the design**, and this one is **verified**, not a guess.
> `dynagraph/survey/_purity2.py` uses Inductor's buffer table to find out what each CPU node reads and on which device:
>
> | Metric | Value |
> |---|---|
> | Scheduler nodes on the CPU | 91 (255 buffers in the graph in total) |
> | Reads by these nodes from buffers on the **CPU** | 96 |
> | Reads by these nodes from buffers on the **GPU** | **0** |
>
> **Not a single one reads the GPU.** In other words, these hundred-odd nodes form a **fully independent computation chain that depends only on shapes**;
> a planner kernel can move the whole chain to the device without waiting for any GPU result.

### Measuring the data-dependence probes with the same ruler gives a clean dichotomy

| Model / probe | CPU nodes | Of which read the GPU | Nature |
|---|---|---|---|
| `ssd300_vgg16` (anchor generation) | **91** | **0** | Shape-driven, pure function |
| `extra:nms` (filtering boxes by score) | 2 | **2 (all)** | Value-driven, depends on the GPU |
| `extra:sparse` (active voxels) | 2 | **2 (all)** | Value-driven, depends on the GPU |

**This is the essential difference between the two kinds of targets, and it is quantified cleanly:**

1. **Shape-driven CPU computation**: large in number (91 in a single detection model), and it **reads nothing from the GPU**.
   Moving it to the device is a pure engineering problem; a planner kernel just evaluates the symbolic expressions.
2. **Value-driven CPU computation**: small in number (2 in each case), and **all of it reads the GPU**.
   This is the real source of "the host must be in the loop" -- the host has to wait for the GPU to finish before it knows what to do.

91 versus 2. **Most host involvement is not caused by data dependence; it happens because shapes are computed on the CPU.**
This is key for the roadmap: doing the first kind first covers the vast majority of nodes, and does not require solving the hard data-dependence problem first.

The exact source of the detection models' `unbacked_binding` was also found: torchvision's NMS **explicitly creates an unbacked symint
in its meta implementation** (`torchvision/_meta_registrations.py:173` and `:187`,
`torchvision::nms` and `torchvision::qnms`):

```python
ctx = torch._custom_ops.get_ctx()
num_to_keep = ctx.create_unbacked_symint()
return dets.new_empty(num_to_keep, dtype=torch.long)
```

"How many boxes NMS keeps" is a textbook data dependence; the host cannot know it in advance no matter what.
This is not a bug that can be fixed; it is the problem itself.

So the detection models carry both kinds of targets at once, at different levels of difficulty:

| Source | Scale | Nature | Difficulty |
|---|---|---|---|
| CPU computation in anchor generation | About 100 nodes per model | **Pure function**, depends only on shapes | A planner kernel computes it directly |
| NMS's unbacked symint | 1-2 per model | **True data dependence** | Needs device-side branching / upper bound + mask |

So the argument must be layered like this, not reported lumped together:

| Category | Scale | What to do |
|---|---|---|
| `cpu_ops` of CV classification models | Zero once subtracted | **Report the bug; it cannot serve as a justification for the project** |
| `cpu_ops` of detection models | About 100 each, cannot be subtracted | **Target** |
| Data-dependent scenarios | 7 of the eight probes have problems | **Target** |

### Origin analysis: the `cpu_ops` of the CV models are **one and the same pattern**; the detection models are not

Breaking the partitioned models down by the source op of their CPU nodes (`dynagraph/survey/_cpuops_probe.py`):

| Model | CPU nodes | Origin distribution |
|---|---|---|
| `timm:nfnet_l0` | 94 | **All 94 are** `aten.cat` + `aten.unsqueeze` |
| `timm:cspdarknet53` | 68 | **All 68 are** the same pattern |
| `timm:swin_base_patch4_window7_224` | 25 | **All 25 are** the same pattern |
| `timm:botnet26t_256` | 4 | **All 4 are** the same pattern |
| `tv:mobilenet_v2` | 71 | **All 71 are** the same pattern (confirmed to come from `nn.ReLU6`) |
| `hf:DebertaForMaskedLM` | 26 | **All 26 are** the same pattern (**NLP models too**) |
| `tvdet:ssd300_vgg16` | 97 | **Diverse**: `add+cat+clone` 57, `cat+clamp` 12, `add+cat+reshape` 9, `cat` 7, `cat+unsqueeze` **only 6**, `clamp+repeat` 5 |

Six models spanning CV classification, CV Transformers and NLP **all share one signature**; only the detection model is diverse.

#### A trap to watch for when interpreting the ablation results

The criterion in `_dyn_ablation.py` is "does turning off `dynamic` bring the count to zero", but **going to zero does not mean fixable**:

| Case | Goes to zero with dynamic off? | Is it a bug? |
|---|---|---|
| `0.0` / `6.0` of `nn.ReLU6` | Yes | **Yes**. Those two are **constants** and should never have been symbolized in the first place (written as integers or literals, the problem goes away) |
| The feature map size that the anchors depend on | Yes | **No**. That is a **true variable**; under dynamic shapes it is necessarily a symbol and necessarily has to be computed |

The two behave the same in that "turning off dynamic makes it disappear", but their nature is exactly opposite.
The criterion needs one more check: **is the quantity being symbolized itself a constant or a true variable?**
A constant being symbolized is a bug; a true variable being symbolized is the problem itself, only computed on the CPU --
and the latter is exactly the job a planner kernel should take over.

The results are in, and they are surprisingly clean:

| Model | `dynamic=True` | `dynamic=False` | Assessment |
|---|---|---|---|
| `timm:botnet26t_256` | 5 | **0** | All fixable |
| `tv:mobilenet_v2` | 72 | **0** | All fixable |
| `timm:swin_base_patch4_window7_224` | 26 | **0** | All fixable |
| `timm:cspdarknet53` | 69 | **0** | All fixable |
| **`tvdet:ssd300_vgg16`** | 99 | **92** | **91 structural, only 6 are symbol materialization** |

**All four non-detection models go to zero (172 nodes in total, none left); the detection model drops only 7.**
And the remaining 91 match exactly the "91 CPU scheduler nodes, zero reading the GPU" measured by `_purity2.py`.
Three independent pieces of evidence (replacing ReLU6 removes not a single one, turning off dynamic removes only 7, zero read the GPU)
point to the same number; with that, the attribution is closed.

`aten.cat` + `aten.unsqueeze` is the signature of "materializing a symbolic scalar into a CPU tensor",
exactly the same as the already confirmed `nn.ReLU6`. So **the `cpu_ops` of CV classification and Transformer models
very likely all come from different instances of the same bug** (module floating-point attributes symbolized under `dynamic=True`),
which `_dyn_ablation.py` is currently confirming by "turn off dynamic and see whether it goes to zero".

The origin in `ssd300_vgg16`, by contrast, is **diverse anchor computation**; that signature accounts for only 6/97. This is structural.

> If the ablation confirms it, the story tightens to:
> **The non-GPU ops in non-detection models are essentially one upstream bug; the real targets are detection/segmentation + data-dependent scenarios.**
> This is good for the argument -- one fewer weakness a reviewer could knock down with a single question.
>
> But we have to face the change in scale: right now 26 models in the statistics have `cpu_ops`;
> **if that bug is fixed, perhaps only 2-3 remain (the detection models)**.
> The headline of Figure 1 would then no longer be "25% of models have non-GPU ops",
> but "detection/segmentation: 0% can go into a full graph" + "data-dependent scenarios: 7/8 have problems".
> The numbers get smaller, but each one holds up under scrutiny.

## Representativeness of the model list: choose different models and the conclusion flips

Of the 131 models, torchvision classification + TIMM make up 62%, and those are the most regular kind.
The categories that really have data dependence were written in pure PyTorch as eight probes (`dynagraph/survey/models_extra.py`):

| Status | Probes |
|---|---|
| Clean, the full graph can go in | Pure-aggregation GNN (the output row count of `index_add_` is determined by the node count; it is not a data dependence) |
| Split into 4-5 partitions | Molecular dynamics, NMS, sparse voxels, variable-length packing, **GNN sampling** |
| **Compile fails** once capture is enabled | **MoE routing**, **LLM decode with variable-length KV** |

**7/8 have problems**, while the CV classification models that compile successfully are 100% clean.
The two hardest categories happen to be the core of LLM serving; on those, today's PyTorch does not just "split into many pieces", it **fails to compile**.

## An easily overlooked point: being split != one graph per partition

After adding a partition-level metric to `report.py`, we found that the proportions in the two columns **match exactly**:

| Suite | Split into 2-9 partitions | Some partition gives up cudagraph |
|---|---|---|
| torchvision classification | 1 (6%) | 1 (6%) |
| torchvision detection/segmentation | 2 (100%) | 2 (100%) |
| TIMM | 17 (30%) | 17 (30%) |
| HuggingFace | 6 (32%) | 6 (32%) |

In other words, **whenever the graph is split, there is always a partition that ends up outside the cudagraph**; it is not "split into two, each with its own graph".
What a split model loses is not the opportunity for fusion across graphs, but that **part of the computation gets no cudagraph benefit at all**,
and goes through normal kernel launches per step.

The partition distribution is also tidy: the maximum number of partitions takes only the values 1 and 2 (`{1: 69, 2: 26}`),
nothing more fragmented. Among the split models, `ssd300_vgg16` is the worst, with 4 partitions giving up.

This directly affects the wording of Figure 1: it cannot just plot "how many partitions", it has to plot "how much computation is kept out of the graph".

## Main-group data (125 models, real tensors, finished 2026-09-17 11:54)

| Suite | Attempted | Compiled | Full graph cudagraph-able | Split | Some partition gives up cudagraph |
|---|---|---|---|---|---|
| torchvision classification | 20 | 18 (90%) | 17 (94%) | 1 (6%) | 1 (6%) |
| **torchvision detection/segmentation** | 7 | **2 (29%)** | **0 (0%)** | **2 (100%)** | **2 (100%)** |
| TIMM | 61 | 56 (92%) | 39 (70%) | 17 (30%) | 17 (30%) |
| HuggingFace | 37 | 37 (100%) | 23 (62%) | **14 (38%)** | 14 (38%) |

Partition reasons (number of models where the reason appears / number of nodes involved):

| Reason | Models | Nodes |
|---|---|---|
| `cpu_ops` | **34** | 976 |
| `device_copy` | 34 | 36 |
| `unbacked_binding` | **2** | 2 |

**The 12 with no data, all listed** (we cannot report only the successes):

- **7 timeouts** (1200 s limit): `tv:swin_t`, `tv:maxvit_t`, `timm:dm_nfnet_f0`,
  `timm:eca_halonext26ts`, `timm:pnasnet5large`, `timm:tf_efficientnet_b0`, `timm:tf_mixnet_l`
- **5 compile failures, all detection models**: `fasterrcnn_resnet50_fpn`,
  `fasterrcnn_mobilenet_v3_large_fpn`, `retinanet_resnet50_fpn`,
  `maskrcnn_resnet50_fpn`, `keypointrcnn_resnet50_fpn`.
  The error is the same for all: `RuntimeError when making fake tensor call`;
  under symbolic shapes the size of `new_full` cannot be inferred (internally it rescales by 800/1333, and the expression is deeply nested).

### Three numbers worth watching

1. **Of the 7 detection/segmentation models, 5 fail to compile and 2 are split; 0 are full-graph cudagraph-able.**
   This is the cleanest one, and it does not depend on any attribution assumption.
2. **HuggingFace is 38% split**, even higher than TIMM's 30%, and all 37 compile successfully.
   This shows the phenomenon is not unique to CV.
3. **`cpu_ops` in 34 models vs `unbacked_binding` in 2 models.**
   On real models, CPU ops are far more common than data dependence -- but how many of those 34 are upstream bugs
   of the `nn.ReLU6` kind will only be known once the attribution is fully broken down. The ablation has already confirmed that BotNet and MobileNet-V2 belong to the fixable kind.

The control group (PyTorch's out-of-the-box configuration) is running; after that come the 6 very large models, and then the tables, figures and attribution are produced automatically.

The TODO list is in `dynagraph/survey/TODO.md`, ordered by importance.

---

## Full results and their impact on the topic choice (close of 2026-09-17)

All 125 models in the main group finished: 113 complete, 0 truncated.

|  | tv classification | tv detection/segmentation | TIMM | HuggingFace |
|---|---|---|---|---|
| Attempted | 20 | 7 | 61 | 37 |
| Compiled successfully | 18 (90%) | **2 (29%)** | 56 (92%) | 37 (100%) |
| Full graph cudagraph-able | 17 (94%) | **0 (0%)** | 39 (70%) | 23 (62%) |
| Split into 2-9 partitions | 1 (6%) | 2 (100%) | 17 (30%) | 14 (38%) |
| Gave up cudagraph for the full graph | 0 | **2 (100%)** | 0 | 0 |

Partition reasons: `cpu_ops` 34 models / 976 nodes, `device_copy` 34 / 36, `unbacked_binding` 2 / 2.

### Splitting the 976 nodes according to the ablation results

| | Models | Nodes | Share |
|---|---|---|---|
| Detection models (judged structural by three independent lines of evidence) | 2 | 251 | 26% |
| The rest (ablation shows they come from symbol materialization) | 32 | 725 | **74%** |

**This split undercuts the original headline claim.** "34 models have non-GPU ops, 976 nodes
split off" sounds impressive, but three quarters of it is caused by **a single class of upstream bug**
(under `dynamic=True`, a module's float attributes get symbolized and then materialized back onto
the GPU). A reviewer could sink it with one question: "why didn't you just fix that bug?"

**Extrapolation risk that must be flagged:** the ablation actually measured only 4 of 5 models
(`botnet26t` 5->0, `mobilenet_v2` 72->0, `swin_base` 26->0, `cspdarknet53` 69->0); `nfnet_l0`
was interrupted. **28 of the 32 were not measured**, and the HuggingFace family (MT5 44, T5 33,
Deberta 26x2, XGLM 26) **was not tested at all**; whether they share the same cause is a guess.
Until the remaining measurements are done, the 74% figure can only be treated as an upper bound.

### So how should the topic be positioned

The model count cannot be the selling point; the three points that actually hold up are:

1. **Detection models do not compile at all.** 5 of the 7 fail outright
   (`fasterrcnn` x2, `retinanet`, `maskrcnn`, `keypointrcnn`, all with
   `RuntimeError when making fake tensor call`), and the 2 that do compile **give up cudagraph for
   the full graph 100% of the time**. These 251 nodes cannot be subtracted away: replacing ReLU6
   removes none of them, turning off dynamic removes only 7, and zero of them read the GPU.
2. **7/8 of the data-dependent probes have problems**, and among them MoE and LLM decode
   **do not even compile**. This is the serving scenario users care about most, and the reason
   vLLM / SGLang have to do piecewise cudagraph.
3. **Those 251 structural nodes are pure shape computation** (`_purity2.py`: zero of them read
   the GPU), which means a planner could potentially compute them -- that is the real
   methodological entry point.

Figure 1 / Figure 2 need to be redrawn accordingly; Figure 2 must be split into two columns,
"fixable upstream" and "structural", and must not report them mixed together. (Originally TODO D2,
now settled.)

---

## Two checks of the load-bearing assumptions (2026-09-17, source-level, no GPU runs)

Trigger: Xinwei asked, "With dynamic, doesn't changing the shape still require re-capture
anyway? And are there really many shapes?" Both questions were checked, and each line of inquiry
was paired with a dedicated verification agent whose job was to poke holes.

### Question 1: `dynamic=True` does not save capture -- holds, not overturned

The two layers are decoupled. At the Inductor layer one kernel covers everything (symints are
runtime int arguments), but the `fn_cache` in `cudagraph_trees` **uses exactly those int arguments
as its key**:

- `cudagraph_trees.py:447-448,457,462,491` -- `int_key = get_ints(inputs)`;
  `fn = fn_cache.get(int_key)`; on a miss it calls `cudagraphify()` -> new `FunctionID` -> new recording.
- `check_invariants` (:2049-2110) **never looks at shape anywhere**; it only compares `data_ptr`
  and liveness -- because shapes have already been separated by `fn_cache` before reaching the manager.
- Official comment at `config.py:2083`: *"If False, we will re-record a graph for each unique
  set of shape inputs"* (`cudagraph_skip_dynamic_graphs` defaults to False).
- Warning text at `cudagraph_utils.py:523-549`: *"recording a new graph for each distinct
  input size"*; the threshold of 8 only warns, it is not a cap.
- The upstream test `test_cudagraph_trees.py:5607-5617` uses `mode="reduce-overhead"` and directly
  asserts 3 shapes = 3 graphs.
- Cost of each new shape: **1 eager warmup + 1 capture + 3 `cudaStreamSynchronize` calls**.
- The number of recorded graphs is **unbounded**: `recompile_limit=8` belongs to Dynamo and does not
  apply under dynamic; `cudagraph_unexpected_rerecord_limit=128` only counts pointer/liveness
  mismatches, and different shapes have different `function_id`s, so they never count toward it.

**One claim needs correcting:** we cannot say "PyTorch has no mechanism for modifying a graph after
capture". torch 2.15 already has conditional nodes (`torch/_higher_order_ops/cudagraph_conditional_nodes.py`,
`set_conditional_handle_kernel` in `aten/src/ATen/cuda/CUDAGraph.cu:9`).
What is missing is **patching kernel-node parameters on the device side**: across the whole tree,
`cudaGraphExecUpdate` / `cudaGraphKernelNodeSetParams` appear only in the hipify name tables and
are never called at runtime.

### Question 2: are there really many shapes? -- what the serving **shape-count** measurements show

Measured numbers (logs on this machine, not estimates): vLLM on H100 with default parameters has
**51 batch buckets** (x2 graph bodies = 102 captures, 9 s / 0.92 GiB); SGLang has 36 decode +
58 prefill = 94. So "there are only 5 shapes" was indeed wrong.

**But the measurements contradict the cost half of the argument, and that half is what decides
whether this counts as a research problem:**

- **Memory does not grow with the bucket count.** vLLM's own model (`gpu_model_runner.py:6578-6583`) is
  `max(shared) + sum(per_graph)`, with `per_graph` about 1 MiB. Of the measured 0.68 GiB, **>=85%
  is a fixed cost paid even when capturing just 1 graph**; going from 51 to 5 buckets saves at most 13%.
- **The SGLang logs on the same machine refute it directly**: 1 bucket 0.00-0.08 GB, 3 large
  buckets 1.97 GB, 36 buckets 5.18 GB -- it follows the **max bs**, not the bucket count.
  Capturing 1 bucket took 69.57 s, longer than the 31.22 s for capturing 36 buckets
  (dominated by warmup/autotune).
- **Startup time**: 81 ms/graph, 4.3% of the 3 min 31 s total startup.
- **The only real cost is padding**, and both systems pay it deliberately
  (`cuda_graph_config.py:48-60`, verbatim: *"the padding waste is the operator's call"*).
- **Root cause**: the serving shape space is **one-dimensional and bounded**. seq_len / KV len are
  absorbed by resident buffers preallocated to the max plus a fixed grid
  (`flash_attn_max_num_splits_for_cuda_graph=32`). The bucket count is an operations knob, not a
  property of the workload.

**Measured: in serving, graph memory follows the max bs rather than the bucket count, and capture
adds 81 ms/graph (4.3% of startup); on these numbers, "serving has to capture dozens of graphs" is
not by itself a cost argument.** A case in serving
would have to target the dimensions that **multiply**: `ShapeKey(size, stream_idx, variant_label, dsa_variant)`
is a product, enabling LoRA specialize multiplies it by N, and pdmux streams multiply it by N --
these are what one-dimensional bucketing cannot absorb.

> **Correction, 2026-09-17: the section above only addresses the "shape count" axis; it does not
> settle the serving line as a whole.**
> There are two independent axes here, which I initially conflated into one:
>
> - **Axis 1, how many distinct shapes there are** -> determines how many graphs must be captured.
>   Shown above: in serving it is one-dimensional and bounded, and its cost is in neither memory nor
>   startup time. On this axis, the measurements above do not support the cost argument.
> - **Axis 2, whether an op can go into the graph** -> determines whether it can be captured as
>   **one full graph**. **The very existence of piecewise cudagraph is evidence for axis 2** -- if
>   there were merely many shapes, capturing a few more full graphs would do; there would be no need
>   to cut the graph. The only reason to cut it is that some ops cannot go in.
>
> And axis 2 is exactly what this project's survey measures (Inductor partitioning because some
> node is not cudagraph-able). **The piecewise that vLLM does by hand is the same thing the Inductor
> partitioner does automatically.** Currently checking what piecewise actually splits on, why
> prefill is harder, and the blocking points in training (the survey so far is inference-only,
> `--train` has never been run; this is a real hole).

### Question 3: beyond serving -- unknown, and the project itself admits it is unmeasured

- `docs/notes/FEASIBILITY.md` (section "9. Eval strategy: copy the PyTorch 2 paper") states it in black and white: **"Recaptures per step under real
  input variation | not measured | target 0"**. This is the load-bearing number for the whole topic, and
  today we do not have it.
- The existing microbench `template_vs_recapture.py`: 200 distinct shapes -> **48.6x**
  (0.552 s vs 0.011 s); **with only 8 shapes -> only 2.1x**.
  `docs/notes/FEASIBILITY.md` (section "Mandatory objection 2: "A fake run is also a capture, so why not just replay the graph that was just captured?"") itself notes that if there are only a few shapes to begin with, the
  gap is the 2.1x above and capture-replay / bucketing already covers that case.
- **Detection cannot be counted as "already absorbed by bucketing"**: `detections_per_img` is 200 in
  `ssd300` and 300 in `ssdlite320` (not 100); worse, the unbacked symint is minted **inside**
  `torchvision::nms` (`_meta_registrations.py:172-174`), while the `keep[:detections_per_img]`
  truncation is **downstream** of it -- the truncation cannot constrain the shape at which the graph
  is split. Two-stage detectors also have `torch.where(scores > thresh)` at `roi_heads.py:719`
  before NMS, whose domain is ~1000x91 ~= 9.1e4 and is not bounded by any cap. **The real answer is
  unknown, not a few hundred.**

### The argument that still stands today, and its hole

A proposed reframing: not "there are many shapes", but "with data-dependent sizes, even **choosing
which bucket** requires a D2H sync", which already holds at K=2. **But the verification agent found
the hole**: the real baseline is not K=2 but **K=1 = pad-to-max**, and pad-to-max needs neither the
host to know the size nor a sync (`docs/notes/FEASIBILITY.md` (section "What was overturned, by severity") says this itself). So that argument only wins
against a multi-bucket scheme nobody is forced to adopt.

**A narrower version that does hold up**: one must argue that **the max upper bound itself is so
absurd it is unusable** -- for MD `cdist` neighbor pairs the worst case is N^2, and the max for
`nonzero` is the whole tensor. That is a different proposition and needs its own proof.

### Next step (no GPU needed)

Fill in that cell at `docs/notes/FEASIBILITY.md` (section "9. Eval strategy: copy the PyTorch 2 paper"): run the dataloader on CPU only and count the distinct sizes under
real workloads, plus the capture hit-rate curve. This number bears directly on the project's
premise, and it does not need a GPU.

---

## The real reason for piecewise, the training blind spot, and a measurement flaw that must be admitted (2026-09-17)

Trigger: Xinwei asked, "Why does prefill need piecewise? Training should also have plenty of
cases where cudagraph can't be used, right? Don't just stare at serving." All three lines were
checked, and in all three the verification agent overturned part of the claim.

### 1. Piecewise is op-level, not shape-level -- the main argument holds

vLLM's own design doc settles it in one sentence (`docs/design/cuda_graphs.md:24`):

> *"Initial piecewise compilation was built to allow piecewise cudagraph capture,
> **excluding cudagraph-unsupported operations (mainly attention)**. This allowed some
> speedup from cudagraphs while maintaining compatibility with all attention backends."*

**Nowhere** in the document is the number of shapes given as the motivation. The split points are a
**literal list of op names** (`_attention_ops` in `vllm/config/compilation.py:459-472`, 12 op names).
`docs/design/torch_compile.md:141-143` puts it more bluntly: capture "the segment between two
attentions", because "the computation between attentions is usually token-wise and
cudagraph-friendly, while attention itself is non-trivial".

**This is exactly the phenomenon this project's survey measures** (Inductor partitioning because
some node is not cudagraph-able). `vllm/compilation/partition_rules.py:39-72` even directly sets
`torch._inductor.config.custom_should_partition_ops = splitting_ops`.

### 2. But three key sub-claims were overturned and must be recorded

**(1) "vLLM's piecewise is Inductor partitioning" -- wrong on the default path.**
`use_inductor_graph_partition` defaults to **False** (`compilation.py:392`).
In `backends.py:646-652` the two paths are mutually exclusive: by default vLLM **pre-splits at the
FX level itself**, and Inductor's `should_partition` is never even consulted about attention.
SGLang does not touch Inductor at all (`piecewise_cuda_graph_compiler = "eager"`).
-> **"The same code path" holds only under one opt-in switch in one framework. The phenomenon has
the same origin; the implementations do not.**

**(2) "Attention cannot be captured" -- wrong.** `cudagraph_mode=FULL` captures the very same
`unified_attention`; `set_splitting_ops_for_attn_fusion` at `compilation.py:812-827` even
deliberately sets `splitting_ops = []` and forces `FULL`, actively putting attention into the graph.
**The real mechanism is**: metadata tensors must live at fixed addresses, and launch geometry
computed on the host must be pinned (FA3 has to change `max_num_splits` from a heuristic to a
constant, preallocate a `scheduler_metadata` buffer, copy the host-computed schedule into it and
zero the tail).
**This is something each backend can fix with engineering effort; it is not a fundamental obstacle.
Piecewise is the fallback that needs "zero work per backend".**

**(3) "Only FA3 is ALWAYS" -- wrong**; `triton_attn.py:69` and `rocm_attn.py:67` are too.
Moreover, `cuda_graphs.md:176,178` says FA2 and Triton are actually both `ALWAYS`, and vLLM still
chooses `FULL_AND_PIECEWISE` **for performance** ("prefill/mixed and pure decode use different kernels").

**(4) The prefill cell actually partly rescues the shape argument.** SGLang's piecewise **opens a
separate token axis** (`server_args.py:1102-1120`): 42 buckets at max=2048, 58 at 8192, 74 at 16384,
completely disjoint from `cuda_graph_bs` -- **in SGLang, piecewise roughly doubles the bucket count.**
(Not so on the vLLM side, which shares the same `cudagraph_capture_sizes` with FULL.)

### 3. Training: the blocker is not in the backward pass but **outside** it

The original guess that "the autograd engine itself is the obstacle" **was wrong**: "Whole-network
capture" in `docs/source/notes/cuda.md:1521-1548` captures forward + `loss.backward()` +
`optimizer.step()` into **one graph**, with the comment reading verbatim
`# replay() includes forward, backward, and step.`.
"The optimizer step must be capturable=True/fused=True" is also wrong -- `torch/optim/sgd.py` has no
`capturable` key at all, and the whole-network example in the docs uses SGD.

**The version that holds (this is the correct framing):**

> Training rarely fails to capture because of autograd or optimizer kernels;
> it fails to capture because **per-step host-side decisions** sit between the capturable regions --
> skip the step on inf, accumulate or step, whether to clip,
> and data-dependent shapes from unpadding / routing / sampling.

Examples verified one by one: `sum(v.item() ...)` in `GradScaler._maybe_opt_step`
(`grad_scaler.py:363-373`; only fused Adam/SGD/Adagrad have a bypass);
`clip_grad` itself is designed to be sync-free, but `error_if_nonfinite` adds the sync back (`:109`);
DeepSpeed's `if clip_coef < 1` acting on a device tensor, and `overflow.item()`;
HF `_get_unpad_data`'s `nonzero` + `.max().item()`;
torchvision's `nonzero`/`randperm` under `if self.training`;
MoE's all-reduced `new_capacity` **being used as a shape**.

**-> So prefill's piecewise and training's partial capture have the same answer:
every real system ends up with "partial capture", because host-side decisions sit in the middle.**

### 4. A measurement flaw that must be admitted: the existing 125 results are neither inference nor training

This one hurts the most, and it was the project's own data that proved it.

- The runs had **no `no_grad`**, so the `aot_dispatch_base` / `is_inference=True` path was never
  taken -> **not a clean inference measurement**.
- There was also **no `.backward()`**, and under the min-cut partitioner, Inductor compilation of
  the backward is **conditionally lazy**: `num_symints_saved_for_bw` is > 0 only if the min-cut
  solution actually saves sym nodes the backward needs. **`dynamic=True` does not guarantee this**
  (`partitioners.py:4344-4348`) -> **not a training measurement either**.
- **The project had already measured this long ago**: `dynagraph/survey/_batch_effect.py` uses exactly the
  runner's parameters (`dynamic=True, mode="reduce-overhead"`), and `dynagraph/survey/README.md` (the `_batch_effect.py` row and the section "Inference and training are two scenarios; run them separately")
  records **"batch 2/32 has a backward, batch 256 does not"** -- backward compilation happens
  **sporadically**.
- More importantly: under min-cut, **the forward graph is determined in reverse by the backward's
  needs** (`choose_saved_values_set`), so the forward under `model.train()` + a real loss and the
  current forward are **not the same graph**.
- The ASPLOS'24 paper reports inference and training **in two separate columns**. **The existing
  jsonl is neither column.**

**-> The survey has to be rerun twice** (once for inference under `no_grad`, once for training with
`--train`); this is not a matter of filling in gaps.

**Another structural limitation**: the largest "cannot use cudagraph" surface in training
(eager autograd's `AccumulateGrad`, the optimizer step, gradient clipping, DDP/FSDP communication
hooks) is **structurally invisible to a harness that only hooks `Scheduler.should_partition`**,
because `compiled_autograd` defaults to False (`_dynamo/config.py:770`).
**No amount of `--train` instrumentation can measure it**; that requires end-to-end training-step
measurement.

### 5. Two harness bugs found along the way

- `n_partitions` dedups using a shared `graph_id` + per-graph `op0/buf0`, **which collides**.
  The fix is `post_grad_graph_id` (`graph.py:563`) + `get_training_phase()` (`:783-788`), three lines.
- **All 7 tvdet models will abort under `--train`**: `_build_tvdet` passes `([tensor],), {}`
  with no targets, and in training mode it hits `torch._assert(False, ...)`. This must be fixed
  before running training.

---

## Workload survey: shape count is the **wrong** statistic (2026-09-17, all measured on CPU)

The question Xinwei kept asking was "is there actually any workload with very dynamic shapes?". I looked into it, and this time I **actually downloaded the datasets and measured**;
scripts and logs are in `dynagraph/survey/workload_shapes/`. The finding has two layers, and the second layer overturns the first.

### First layer: there really are many shapes (measured, reproducible)

| Workload | Measured | Data |
|---|---|---|
| KITTI sparse conv | **433 frames, 433 distinct** four-stage active-voxel tuples | real HDL-64E point clouds |
| ogbn-arxiv neighbor sampling | **200 steps, 200 distinct** shape tuples (6 axes) | real graph |
| QM9 molecules | 156 batches, 155 distinct (atom count, edge count) | real molecules |
| MD trajectory N=370 | **94% of consecutive frame pairs have different shapes** | real MD22 trajectory |
| MovieLens-25M | 500 steps, 495 distinct total lengths | real interaction logs |

For scale: PyTorch's own `cudagraph_dynamic_shape_warn_limit` defaults to **8**
(`config.py:2133`); the workloads above exceed it by 25-55x.

### Second layer: but this number alone does not make the case, because the baseline is not "one graph per shape"

**The real baseline is K=1 = pad-to-max**: pad everything to the global maximum and run it all with one static graph.
It needs no host knowledge of the sizes and no synchronization of any kind; `dynamic=False` already does it
(`docs/notes/FEASIBILITY.md` (section "What was overturned, by severity") said so long ago). So the right statistic is the cost curve:

    cost(K) = E[pad_K(s)^p] / E[s^p]

p is the kernel's real cost exponent (jagged scatter/pooling=1, attention and pair repr=2, triangle=3).
**The criterion requires both conditions**: `cost(1)` is unacceptably expensive, **and** `cost(8)` is still unacceptable.

**Once K=1 is computed, pad-to-max for the four "winners" above costs only 1.03-1.42x more.**
(The reason is that their costs are all **linear**, and the distributions have no heavy tail.)

> **Criterion corrected (2026-09-18, pointed out by Xinwei):** I originally used `cost(1) < 1.5x` as the threshold for ruling a workload out;
> **that was wrong**. The percentage wasted by padding **is exactly the recoverable speedup** -- a template neither pads nor re-records,
> so its gain relative to pad-to-max is exactly that waste. 1.29x means **29% recoverable speedup**,
> which is not a small number in a systems paper.
>
> For p=1 workloads, `cost(1)` is exactly `max/mean`, which can be computed directly from `results.log`:
>
> | KITTI sparse conv | max/mean | recoverable |
> |---|---|---|
> | stride-1 | 41703/32291 | **29%** |
> | stride-2 | 24715/16373 | **51%** |
> | stride-4 | 12469/7086 | **76%** |
> | stride-8 | 5405/2781 | **94%** |
>
> **The deeper the stage, the worse the waste**, because there are fewer active voxels and the distribution is more spread out. The total waste of jointly padding all four stages
> has to be weighted by each stage's work, but no stage is below 29%.
>
> So the right criterion is not "is it expensive enough", but the question in the next section: **is cudagraph actually useful in this workload at all**.
> If eager is already fast enough, then neither the 29% nor the 2590x can be captured.
>
> (Side note: MovieLens's "14.0x~33.7x" is **padding each sequence to the batch max in a dense layout**,
> which is a different thing from "padding the jagged total length to the global max"; the latter is only 1.09x. These two numbers must not be mixed;
> they describe two mutually exclusive implementations.)

### What survives: superlinear cost x heavy-tailed distribution

Measured (`dynagraph/survey/workload_shapes/bucket_cost.py`, UniProt human proteome, 147,520 sequences,
lengths 2 to 35,963, 3,058 distinct):

| AlphaFold-like | K=1 | K=2 | K=4 | **K=8** | K=16 | K=32 |
|---|---|---|---|---|---|---|
| pair representation (p=2) | **2,590x** | 107x | 9.7x | **2.75x** | 1.64x | 1.27x |
| triangle ops (p=3) | **16,186x** | 15,973x | 218x | **7.37x** | 2.47x | 1.48x |

**Both criteria are met; among the workloads measured so far, this is the only one that meets both.**

> **Fixed a load-bearing calculation bug along the way**: the
> `if b is None: continue` in `workload_shapes/prot.py:24` **dropped sequences beyond the largest bucket from both the numerator and the denominator**,
> and that 0.06% of long sequences happens to account for 79% of the total L^3 work. Because of this it had reported that AF3's thirteen buckets cost only
> 1.39x/1.43x more, and on that basis proteins were **demoted** -- exactly the wrong direction.

### Not yet measured but most likely to survive: RL rollout training

For ordinary variable-length SFT, packing removes the variation: greedy packing makes the total token count **strictly constant**,
and only the number of sequences per pack varies (measured 13/23/42 distinct). **Packing eliminates the problem outright.**

But there is one class where packing cannot be used: **RL rollout training (GRPO, PPO in verl / OpenRLHF)**.
Generation lengths are naturally scattered and uneven, the frameworks cut micro-batches dynamically by token budget,
so the sequence count and the total length **both vary**, and attention is O(L^2).
Measured on tulu-3, small-batch pad-to-longest already costs 2.41x at B=8 and 3.28x at B=32.
**Nobody has measured this scenario so far**; it is the next measurement to make, and it is pure CPU (parse rollout logs + numpy).

### A precondition that must be checked first

The equation "distinct shape count == recapture count" **has a precondition**: the varying dimension must
**enter the compiled region** as a symint. spconv's voxelization and rulebook are opaque
custom ops, and Inductor partitions around them -- in that case KITTI might have **no cudagraph at all**,
rather than 433. **This precondition must be checked on CPU first** (compile a stub model and look at the partition boundaries and each partition's symint inputs);
otherwise the whole shape-cardinality argument has nothing behind it.

### Net effect on the choice of research topic

- "Many models have many shapes" cannot be the selling point; **shape count is not the criterion**.
- The narrow claim that holds up is: **the upper bound of pad-to-max is so absurd it is unusable**,
  and this only holds when "cost is superlinear + the distribution is heavy-tailed". Protein structure prediction meets this as measured,
  RL rollout very likely does, and the other measured workloads do not.
- The other, independent axis (ops that cannot enter the graph / the host must synchronize to learn the sizes) is unaffected;
  the evidence is in earlier sections of this document and **does not need the workload survey to support it**.

---

## Is cudagraph actually useful: the anti-correlation hypothesis is **refuted** (2026-09-18)

Xinwei asked the real crux: "If things are also fine without cudagraph, that would be very bad."
I proposed a falsifiable anti-correlation hypothesis: the scenarios where pad-to-max is expensive (large kernels, GPU-bound)
and the scenarios where cudagraph helps (small kernels, launch-bound) are mutually exclusive.
**All three survey threads said the hypothesis held; three verification agents each independently refuted it**, and all of them pinpointed the same arithmetic error.

### The decisive error: the kernel count was underestimated by 50x

The survey side estimated about 2,976 kernels per AlphaFold step (48 Evoformer blocks x 62 ops),
from which it computed a launch share of 0.0% and a cudagraph speedup of 1.003x, and ruled it out.

**Actual profiling data** (ScaleFold, NVIDIA, MLPerf HPC v3.0 OpenFold, Table 1):

    18,147 (math-bound) + 97,749 (memory-bound) + 34,991 (memory ops) = 150,887 per step

**Off by 50.7x.** And the same table directly lists **"CPU Overhead: 9.10%"** -- profiled, not estimated.

### Three pieces of measured evidence

**1. AlphaFold training (an increment NVIDIA isolated itself)**

> *"After applying DAP-8, CudaGraph and disable gradient checkpointing, we got 1.79X speedup.
> Without CudaGraph, DAP-8 ... only achieved 1.52X"* -> **cudagraph alone contributes 1.79/1.52 = 1.18x**

More importantly, ScaleFold **ran into exactly the problem this project sets out to solve, and built its own workaround by hand**:

> *"if the CUDA kernels within this scope are modified due to dynamic computation graph,
> such as in the case of recycling in the AlphaFold training, CUDA Graph needs to be recaptured.
> To address this, **we designed a CUDA Graph cache** that can capture multiple graphs for
> different recycling scenarios."*

**To avoid re-recording, NVIDIA hand-wrote a graph cache for its flagship protein workload.** This both proves the problem really exists
and shows that a K-bucket scheme was "good enough" for them -- both sides have to be written up faithfully.

**2. NVIDIA NIM OpenFold3 1.6.0 (the strongest one; measured on H100, with cudagraph isolated on its own)**

> *"The BioNeMo Inference Runtime now captures the diffusion module as a CUDA graph.
> What this removes is per-request kernel launch overhead ... across the benchmark suite
> on H100 it takes roughly **7-8 seconds off every prediction**, whatever the input size."*

| Residues | Before | After | Speedup |
|---|---|---|---|
| 186 | 10.2 s | 2.9 s | **3.48x** |
| 1869 | 82.5 s | 72.5 s | 1.14x |

And the original text states: *"runtime below roughly **400 residues** was dominated by fixed per-request overhead"*.

**The median length of the human proteome is 349 -- right inside the launch-bound range.**
This is why the anti-correlation does not hold: I assumed proteins belong to "large kernels, GPU-bound", but in fact their **median falls at the small end**.

Refitting the human proteome from these two measured points (t(L)=0.002045*L^1.391, fixed overhead D=7.27 s):

- cudagraph gain over the whole proteome: **1.56x-1.76x**
- **50.9%** of proteins get >=2.0x from cudagraph alone; 93.1% get >=1.2x
- Net headroom of **DynaGraph relative to "the better of the two baselines"** (eager with exact shapes / AF3's thirteen buckets + static graphs):
  **1.31x** over the whole proteome, **1.56x** with eight work-weighted buckets, **1.70x** at the median length

**3. The full crossover curve for MLIP/MD** (TorchMD-Net 2.0 Table 2, RTX4090, every entry recomputed)

| Atoms | 22 | 49 | 166 | 2489 | 5807 |
|---|---|---|---|---|---|
| cudagraph speedup | **6.64x** | 4.65x | 1.97x | 1.02x | 1.02x |

Small molecules get 5-10x; past about 2500 atoms the gain drops to nothing. And this project measured MD22 at N=118 with
486 distinct shapes and 84% of consecutive frames having different shapes -- **N=118 sits right in the effective part of the curve**.

### Four pieces of unfavorable evidence that must be written up alongside

1. **The GNN cell has already been done.** ZeroGNN (arXiv:2605.29346, 2026-05, William & Mary)
   measured GraphSAGE/Reddit at bs=128 as **45% GPU-active / 55% idle**, and 8.75x at bs=64.
   This is **prior work**, not "cudagraph is useless". The two must be kept apart.
2. **NequIP's 2025 overhaul did not use cudagraph at all** -- it got 4-18x from AOTInductor + Triton,
   the string `cuda graph` appears **0** times in the whole paper, and it explicitly says it chose TorchInductor because it
   *"can accommodate dynamic tensor shapes"*. **Launch-bound domains are solving the same problem by a different route.**
3. **cudagraph can backfire.** PyGraph measured **29 of 116 graphs (25%) getting slower**,
   worst case EOS regressing 29% and Deep Recommender regressing 15%.
4. **AlphaFold2 training has no shape variation**: ScaleFold's text says
   *"all local batches are cropped into the same shape"*. The protein shape space exists only in **inference**,
   not in training. Earlier use of the UniProt length distribution as a training scenario was wrong.

### Net takeaway

**Xinwei's worry was a real question, but the answer points the other way.** cudagraph **does help** in these scenarios,
and precisely on the workload whose shapes vary the most (protein structure prediction inference) --
because the median protein length of 349 falls in the launch-bound range.

But the net headroom has to be stated honestly: not the 2590x of pad-to-max, but **1.3-1.7x relative to the better of the two baselines**.
The criterion should be fixed as:

    net headroom = min(pad waste, cudagraph speedup), taken relative to the better of {eager with exact shapes, K-bucket static graphs}

Because these two quantities are **controlled by the same variable** (GPU work per kernel vs about 2-8 us of launch overhead),
they naturally suppress each other -- this is the one thing the survey side got right, and it is a valuable insight worth keeping.
