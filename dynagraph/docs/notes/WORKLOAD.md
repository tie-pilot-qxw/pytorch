# Elastic serving workload: what to look for, what to use (survey, 2026-09-21)

This file answers one question: which workload to use to show that "one CUDA graph across shapes / across parallel widths" is useful.
The survey ran along two parallel tracks -- one over public systems and papers, one over what already exists on this machine. The findings are below, with the evidence after each item.

## The single most important fact

**In production engines, elasticity and CUDA graphs are currently mutually exclusive.** We did not infer this; they write it in their own code:

- vLLM's own elastic EP launch script `examples/ray_serving/elastic_ep/serve_deepseek_v2.sh` passes
  `--enforce-eager` outright.
- On the path that does enable graphs, `vllm/distributed/elastic_ep/elastic_execute.py` calls
  `_release_cuda_graphs()` and then `warm_and_capture()` on every scaling event; the comment says "on scale-up the workspace is reallocated,
  and already-captured graphs would hold stale pointers, so throw all the graphs away first".
- All three SGLang elastic scale test classes set `CUDA_GRAPH_ARGS = DISABLED_CUDA_GRAPH_ARGS`.
- Xinwei's own GF-DiT / vllm-omni has zero cudagraph references under `runtime_v2/` -- not because it is unwanted, but because it cannot be done.
- The only one that keeps its graphs through a switch (Moebius, arXiv:2606.26607, SGLang v0.5.5, 8xH200, EP<->TP switch in
  215-434ms) does it by keeping a full resident set of graphs for each mode, i.e. pad-to-max along the "number of graphs" dimension. The code has not been released.

This is exactly DynaGraph's thesis statement: everyone else either turns graphs off or pads to the maximum. We let one graph follow the change.

## What to use (two arms)

**Arm A: Elastic EP in vLLM 0.29.0.** Cheap, standard stack, public traces, can produce a table directly.
`tests/distributed/test_elastic_ep.py` is `@multi_gpu_test(num_gpus=4)`, model
`deepseek-ai/DeepSeek-V2-Lite-Chat` (bf16, about 31GB), DP 2->4, and it **already runs parametrized over
`enforce_eager` and `cuda_graphs`**, and prints
`scale_seconds` / `downtime_seconds` itself. The difference between those two numbers is the re-capture tax we claim to erase.
Our contribution is adding a third arm. Watch out for the `has_nixl()` skip; if it cannot be passed, run
`serve_deepseek_v2.sh` + `scale.py` by hand.

**Arm B: GF-DiT / vllm-omni (Xinwei's own).** Highest scientific value: the DiT switches from sp=1 to sp=2 at denoising step 10,
so the per-GPU sequence length and the graph topology change at the same time. The paper's own table shows the optimal SP flips with latent size
(512^2: SP1 155ms beats SP4 172ms; 1536^2: SP4 464ms beats SP1 1557ms), so static configurations are Pareto-dominated.
Code: github.com/SJTU-Liquid/GF-DiT, already checked out in the container `xinwei_vllm_omni`.

**LoongServe as the ancestor citation, not as an artifact.** Last push 2024-11-11, torch==2.4.0,
triton-nightly pulled from an Azure feed, assumes 8x A800/A100 + full NVLink, slurm and /mnt/petrelfs absolute paths hardcoded,
plus an unresolved rnccl build issue. Its datasets (ShareGPT/LEval/LV-Eval) can be downloaded, but the arrival process is synthetic Poisson,
not a real trace.

## Trace

vLLM main already has a `timed_trace` dataset (`--self-timed` replays by timestamp) and a built-in `burstgpt` loader,
so the two below can be driven with zero glue:

- Mooncake FAST'25 conversation trace: `kvcache-ai/Mooncake` repo, `FAST25-release/traces/`.
- BurstGPT v2.0: `HPMLL/BurstGPT` release assets, 51-232MB.

Also available in the lab (ask Xinwei for the location): four Azure LLM inference trace CSVs (public, from
`Azure/AzurePublicDataset`; up to 27M rows, peak-to-mean ratio about 3.5x, which is exactly the "should trigger a width
change" signal), and a real coding-agent trace (syfi_coding_trace, 665,453 turns / 8,058 sessions, 2025-09 to 2026-07,
companion paper arXiv:2606.30560).

## Blockers up front (nothing was touched during the survey; all read-only)

1. The torch in the two containers is incompatible: DynaGraph needs the source-built `/workspace/pytorch-main` (which has no
   vllm/sglang/diffusers/vllm_omni), and vllm-omni has its own venv (torch 2.10 / vllm 0.17).
   Getting the two into the same process is Arm B's first hurdle.
2. The Wan2.2-TI2V-5B named by the serve script was never downloaded (only an empty .lock directory remains); what is complete locally is
   Wan2.1-T2V-1.3B (27GB, in /ssd2/xinwei/.cache). Whether it loads in vllm-omni has not been verified.
3. serve-elastic.sh needs 3 ranks.
4. Whether the DiT forward goes through Inductor at all has not been verified -- if it does not, DynaGraph has nothing to do there; this is Arm B's
   decisive unknown. The suggested cheapest first step is to feed a single sp=1 DiT step to
   `torch.compile(dynamic=True)` on an idle GPU and see how many graphs it compiles and whether there are graph breaks.

## Suggested order

Arm A first (half a day; it can produce a "re-capture tax" table), and in parallel run Arm B's step 4 check on an idle GPU.
Decide the full shape of Arm B once step 4 has a result.
