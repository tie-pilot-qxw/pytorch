# How to measure on this machine

These are lessons from the past few months. Most of them cost at least a day before we understood
them. Read this before trusting any number, ours included.

## The machine

- **Check who else is on a card before you use it.** It is a shared 8x H100 box. Run
  `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv`, then
  `ps -o user,etime,args -p <pid>` for each process. Eval, training and simulation jobs are fine to
  share a card with for correctness runs. A job that is being timed (a benchmark, a serving server
  under a load generator) is not: you corrupt the other person's numbers and your own. Utilization
  alone does not tell you which kind of job it is. For timing, use an empty card, and if you had to
  share, say so next to the numbers.
- **All cards are power-capped at 550 W** (the default is 700 W). They hit the cap a large part of the
  time, and SM clocks then drop by up to ~30%. The same kernel can differ by 40% between cards or
  between hours. Check before timing:
  `nvidia-smi --query-gpu=index,power.limit,power.draw,clocks.sm,clocks_event_reasons.sw_power_cap --format=csv`.
  Keep an A/B comparison on one card, run interleaved.
- **The host is usually loaded by other people's jobs.** That is not necessarily contamination:
  real training and serving hosts are busy too, and reducing CPU launch overhead is what CUDA graphs
  are for. The problem is load that drifts between configurations. Do not run config A for ten
  minutes and then config B. Keep all configurations alive and round-robin their timed passes, take
  min-of-N or the median, and report the spread. Graph count and peak memory do not depend on host
  load, so they are useful even when timings are noisy.
- **A container can lose its GPUs, and a card can become unusable.** Symptoms:
  `Failed to initialize NVML: Unknown Error`, or `CUDA_ERROR_UNKNOWN from cuDevicePrimaryCtxRetain`
  / `cudaErrorDevicesUnavailable` on a card that `nvidia-smi` on the host shows as idle. Before a
  long run, check the card with a one-line `torch.randn(4, device="cuda").sum()`. If it fails,
  repeat the check in a fresh `docker run --rm --gpus '"device=N"' <image>`. If only your container
  fails, `docker restart` it. If the fresh container fails too, the card itself is broken (seen on
  card 6 in October 2026); use another card and tell whoever administers the machine.
- **Run git on the host, never inside the container.** The container runs as root; a `git stash`
  inside it once left `.git` objects and source files owned by root, and the next commit failed.

## What to compare against

- **Put every baseline on the table from the start: eager, compile, trees, pad, bucket and oracle**
  (definitions in `MEASUREMENTS.md`). Comparing only against `compile(dynamic=True)` makes any
  graph scheme look good. Bucketing is what serving frameworks actually do and is the baseline
  that matters.
- **Define the oracle precisely.** "One graph per shape, replay time only" excludes the Dynamo guard
  evaluation and wrapper cost (about 0.35 ms per call on BERT, see MEASUREMENTS) that every `torch.compile` scheme pays. It is
  a floor, not a reachable target; treating it as the target overstates the available headroom by
  2x or more on small models.
- **When something wins, find out where the time went.** On ESM-2, DG beat compile by a few ms per
  step. That turned out to be the per-call Python dispatch of the DeepGEMM custom op that compile
  pays and DG skips. With Triton GEMMs the gap closed, so it was a property of the custom op, not of
  graph patching.
- **Time the right segment.** `e2e/harness.py` runs three segments: `warm` (compilation and first recordings; its total time is printed as `warm N s` but it is not part of the per-step numbers),
  `new` (every step is a shape not seen before) and `replay` (the same shapes again). For dynamic
  shapes, `new` is the realistic one. `replay` is what per-shape recording would give you if the
  shape set were small and closed, which it usually is not (the human proteome has 3058 distinct
  sequence lengths).
- **1-D and 2-D shape spaces are different problems.** With only L varying, 6 power-of-two buckets
  cover [16, 512]. With B and L both varying, it takes 29 buckets and 30 compiles, and buckets keep
  appearing during the timed run. Results from one do not transfer to the other.
- **A separate compile is not a numerics reference.** Inductor autotunes reduction configs on first
  run, and under load the winner varies between compiles. That changes the reduction order and
  causes ~3e-5 relative differences at the output. Compare within one compile (same graph, feature on
  vs off), or set `config.triton.autotune_pointwise = False` on both sides.

## Reading the numbers

- **Profile before you explain.** A first-pass cost of 2.6 s was written up twice as "12 harvests,
  each with a cuDNN warmup". A 30-line cProfile run showed that 72% of it was two nvcc subprocess
  waits; the harvests took 60 ms in total. Every first-pass or build-time number should come with a
  cProfile top-10 (`tottime`, unfiltered) before you attribute it. In the harness: `DGPROF=new` or
  `DGPROF=replay`.
- **Removing something that looks expensive is not the same as finding the bottleneck.** Twice a
  "fix" (dropping a per-call sync, caching an `ast.parse`) did not reproducibly change anything. If a
  measurement is not reproducible, fix the measurement first.
- **Compare medians with medians and means with means.** Step-time distributions are skewed: in the
  2-D setting most steps are small and a few have B=16/32 with long L. A no-sync run only yields a
  mean (one sync per segment), so it must be compared with the sync mean, not the sync median.
- **Sync vs no-sync measure different things.** Syncing every step gives single-request latency
  (host + GPU). No-sync lets the host work of step n+1 overlap the GPU work of step n, which is what
  an asynchronous training loop sees. Graph schemes help more in the first case.
- **Relative loss differences are meaningless near zero.** `loss_reldiff` (printed by
  `e2e/harness.py`, so also by `e2e/zoo.py`) divides by a mean output that can be close to 0. Check
  element-wise errors instead.
- **Framework stage timers may not sync.** SGLang's per-stage timings do not synchronize. With
  graphs enabled, "denoise time" is just host submission, and the GPU time shows up in the next stage.
  Use end-to-end request time.

## Library behaviour that skews dynamic-shape measurements

- **cuDNN SDPA builds an execution plan for every new shape** (~5.4 ms per layer on H100). It is
  PyTorch main's default attention backend on H100. It dominates any new-shape step: 65 ms for BERT's
  12 layers against 1.66 ms of GPU work. The zoo harness disables it unless `CUDNN_SDP=1`.
- **cuBLAS picks different kernels and node topologies (e.g. split-K) as M changes**, so a new M can
  mean a new child graph for DG to harvest.
- **cuDNN convolutions choose an algorithm the first time they see a shape** (~0.5-0.7 s for SANA's
  VAE decoder at a new resolution).
- **Duck sizing** (`torch.fx.experimental._config.use_duck_shape`, on by default) gives dimensions
  that are equal at trace time the same symbol. At SANA 1024^2, channels, H and W are all 32, so a
  guard on one pins all three. Turn it off for workloads where that happens.
- **`reduce-overhead` outputs live in the graph's memory pool.** If a caller keeps the output of one
  call and then calls the same graph again (CFG does exactly this), the first output is overwritten.
  Call `torch.compiler.cudagraph_mark_step_begin()` per step and clone outputs that must survive.

## DynaGraph-specific debugging

- **A CUfunction handle is not an identity.** When autotuning unloads the losing configs, the driver
  reuses their addresses for kernels JIT-compiled later (e.g. DeepGEMM). DG matched graph nodes to
  kernels by function handle and wrote parameters into somebody else's kernel. We saw all-zero
  outputs once and illegal instruction (715) another time, and it did not happen on every run.
  `TORCHINDUCTOR_DYNAGRAPH_DEBUG=1` prints `debug node k = <func name> grid ...` lines that show a
  mismatch directly. If the bug is intermittent, suspect address reuse first.
- **The first new shapes are checked against eager.** By default the first 3 new shapes of a region
  are verified (`dynagraph_verify_shapes`), and a mismatch retires the region. Every fallback is
  logged with a tag. Run with the `torch._inductor.dynagraph` logger at INFO (as
  `probes/regress_quick.sh` does) and read the tags before you read the timings.
- **Building third-party CUDA extensions in the container can silently use the wrong torch.**
  `torch.utils.cmake_prefix_path` points at a directory that does not exist in this in-place build,
  and CMake then falls back to the container's torch 2.11. Source `setup/use_main.sh` and pass
  `CMAKE_PREFIX_PATH=/workspace/.venv/lib/python3.12/site-packages/torch/share/cmake` explicitly.
  Check with `grep -o "\-isystem /[^ ]*torch/include" build.log | sort -u`.
