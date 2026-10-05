# TODO (end of day 2026-09-17, by importance)

Conclusions are in `../docs/notes/FINDINGS.md`.

## 0. GPUs taken, all GPU jobs stopped

At end of day 2026-09-17 **yichen's `bench/extpower` power test was running on GPU 4 and GPU 5**.
Per the "do not share a card with others" rule, all GPU jobs were stopped and confirmed exited.
**GPU 6 / 7 are idle but not authorized**; ask before using them.

## 0.5 Already fixed, ready to run once a GPU is free (done this round, all checked on CPU)

**The existing 125 results are invalid**: they were run with neither `no_grad` (the clean inference path
`aot_dispatch_base` / `is_inference=True` was never taken) nor `.backward()`
(and the backward Inductor compile under min-cut is **conditionally lazy**; `dynamic=True` does not guarantee it fires,
see `partitioners.py:4344-4348`; the project itself measured in `_batch_effect.py`
"batch 2/32 has a backward, 256 does not"). **The ASPLOS paper reports inference and training as two independent columns; these results fit neither.**

Fixed:

1. **`instrument.py` dedup collision** -- it used `V.graph.graph_id` as the key, but forward and backward share that id,
   and node names restart at `op0`/`buf0` in every graph, so the backward's `op0` was dropped as the forward's.
   Now uses `post_grad_graph_id` (a global counter, `graph.py:563`).
   Also hooked up `get_training_phase()` (`graph.py:783-788`) and added two output fields,
   **`partition_phase_counts`** and **`partition_reason_by_phase`** --
   the only reliable basis for reporting inference and training separately.
2. **`runner.py` got a real inference path** -- without train, explicit `torch.no_grad()` + `model.eval()`;
   with train, `model.train()` + backward. Results record `phase_mode`.
3. **`models.py` fixed tvdet training** -- all 7 detection models used to fail under `--train`
   (`ssd.py:330 torch._assert(False, "targets should not be none when in training mode")`).
   Now targets are passed, and there must be **2 images** (ssdlite320's BatchNorm throws
   "Expected more than 1 value per channel" at batch=1) and **every label is 1**
   (keypointrcnn defaults to `num_classes=2`, so 2 is out of range).
   **All 7 pass forward+backward on CPU.**
4. **New `train_step_probe.py`** -- an end-to-end training step probe.
   Compile-time hooks cannot see the largest surface in training (eager autograd's `AccumulateGrad`,
   optimizer step, gradient clipping, GradScaler's inf check, DDP/FSDP comm hooks),
   because `compiled_autograd` defaults to False (`_dynamo/config.py:770`).
   So it uses two other measurement points: **runtime capture count** (hook `CUDAGraphTreeManager.record_function`,
   `cudagraph_trees.py:2830`) and **per-section host sync count**
   (`torch.cuda.set_sync_debug_mode("warn")`, collected over six sections: forward/loss/backward/clip/step/zero_grad).
   Both hooks have been verified to install on CPU.
   Note the sync count is a **lower bound**: the official docs say the distributed and sparse namespaces are not covered.

To run once a GPU is free: one `--no-grad` inference round, one `--train` training round,
and `train_step_probe.py` once each with/without AMP, with/without clip, and SGD/Adam/fused-Adam.

## A. Can continue once a GPU is free

**A1. Fill the extrapolation gap in the ablation (most urgent).** The current "74% fixable" extrapolates from 4 models to 32;
**not a single model from the HuggingFace family was tested** (MT5 44 / T5 33 / Deberta 26x2 / XGLM 26 nodes).
Fill in 3~4 HF models and `nfnet_l0` first; otherwise 74% can only be written as an upper bound.

**A2. Rerun the control group.** `results_real_default.jsonl` has only 6 rows;
whether the two measurement methods agree (question 1) has not been checked on the full set.

**A3. Redraw Figure 2 with split bars.** Split into "fixable upstream" and "structural"; `batch_attribution.py` needs extending:
it only swaps ReLU6 now, but `nfnet_l0` / `cspdarknet53` / `swin_base` / `Deberta`
have the same `aten.cat`+`aten.unsqueeze` signature and should be swapped as a whole class ("module float attribute").

## B. Missing data

**B1. Training scenario.** Only forward is run now, and whether the backward graph compiles depends on AOTAutograd's laziness,
so this round is the **inference scenario**. `runner_next.py` already has `--train`; rerun one round once a GPU is free and report it separately.
The "messy open-source training code" the user cares about is a training scenario.

**B2. Merge the eight probes into the official list.** `models_extra.py` was tested (7/8 have problems;
MoE and decode do not even compile), but it has not gone through the official `runner.py` flow or into the jsonl.
Use spec prefix `extra:`.

**B3. The list is not representative enough.** Of 131 models, torchvision classification + TIMM make up 62%.
The probes are only "core patterns", not real code; with spare capacity, install PyG/DGL to run real GNNs,
and pull a real decode loop out of vLLM / SGLang.

## C. The harness itself

**C1. Switch to `instrument_next.py`** (records `origins`, fixes `scheduler_api` detection).
**No jobs are running now, so this is the time to switch.**
Note: `scheduler_api={'unknown': 79, 'str|None': 34}` in the self-check is a **false alarm, not lost data** --
the 79 "unknown" are exactly the models that never triggered a partition reason, and all 34 with `cpu_ops` were correctly identified as
`str|None`; the two numbers match up.

**C2. Widen `_purity2.py` coverage.** It only measured SSD300 and two probes,
getting a 91-vs-2 split (shape-driven: 0 read the GPU; value-driven: all read the GPU).
This split is the basis of the roadmap and should be confirmed on more models.

## D. External

**D1. Whether to report the `nn.ReLU6` bug upstream.** Self-contained repro: `repro_relu6.py`;
a search upstream found no exactly matching issue. **Per the repo's AI policy, the user must confirm before it is filed.**
The ablation shows a larger impact than first written (4 models, 172 nodes, all go to zero), which makes it more worth reporting.

**D2. Settled** -- see "Full results and their impact on the topic choice" in FINDINGS: the model count by itself cannot be the main argument.
