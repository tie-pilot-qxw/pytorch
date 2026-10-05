#!/usr/bin/env python3
"""How much extra time each decode step costs: DynaGraph patching nodes vs vLLM replaying directly.

The earlier "how many graphs got captured" numbers are all startup cost. What actually decides whether this
approach pays off is **how long replacing nodes takes within one forward**, because that is paid on every step.

How it is measured: the same batch, run with two max_tokens values (T1 < T2), taking the slope

    per-step latency = (t(T2) - t(T1)) / (T2 - T1)

This subtracts all the overhead that depends only on the request count and not on the step count: prefill,
scheduling, detokenize. Each config runs ROUNDS rounds and takes the median; different configs **run in separate
processes, interleaved** (see _latency_sweep.sh, not included in this repo), because the host load on this machine
is the operating condition, not noise.

In-process it also directly counts the time DynaGraph spends on host-side patching (`_host_step`) and the number
of calls, so "the extra wall time per step" and "the time the patch reports for itself" can be reconciled.

    MODE=vllm|dynagraph  BS=8  T1=8  T2=72  ROUNDS=5  python serving/probe_vllm_latency.py
"""
import os, sys, time, logging, statistics

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
_STUB = os.path.join(os.environ.get("DG_DEPS", "/workspace/_deps"), "fa4site")
if os.path.isdir(_STUB) and _STUB not in sys.path:
    sys.path.insert(0, _STUB)

MODE = os.environ.get("MODE", "dynagraph")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
BS = int(os.environ.get("BS", "8"))
T1 = int(os.environ.get("T1", "8"))
T2 = int(os.environ.get("T2", "72"))
ROUNDS = int(os.environ.get("ROUNDS", "5"))

import torch
import torch._inductor.config as ic

logging.basicConfig(level=logging.WARNING)

if MODE != "vllm":
    ic.triton.cudagraphs = True
if MODE == "dynagraph":
    ic.triton.dynagraph = True
    ic.triton.dynagraph_extern_child = os.environ.get("EXTERN_CHILD", "1") == "1"
    ic.triton.dynagraph_update = os.environ.get("TORCHINDUCTOR_DYNAGRAPH_UPDATE", "auto")

# Time DynaGraph spends patching nodes on the host. `_host_step` covers both the patch and the launch,
# so it is the most direct answer to the "replacing nodes" question.
patch = {"n": 0, "t": 0.0}
call = {"n": 0, "t": 0.0}
# Time the region really spends on the GPU. Host time is only a few percent of each step and both sides are
# GPU-bound, so "is the extra time on the host or on the device" must be measured with events, not guessed by
# subtracting wall times.
gpu = {"n": 0, "t": 0.0, "ev": []}
GPUTIME = os.environ.get("GPUTIME", "0") == "1"


def _gpu_wrap(fn):
    def inner(self, *a, **kw):
        if not GPUTIME:
            return fn(self, *a, **kw)
        s0 = torch.cuda.Event(enable_timing=True)
        s1 = torch.cuda.Event(enable_timing=True)
        s0.record()
        r = fn(self, *a, **kw)
        s1.record()
        gpu["ev"].append((s0, s1))
        gpu["n"] += 1
        return r

    return inner


def _gpu_drain():
    if not gpu["ev"]:
        return
    torch.cuda.synchronize()
    gpu["t"] += sum(a.elapsed_time(b) for a, b in gpu["ev"]) / 1e3
    gpu["ev"].clear()
if MODE == "dynagraph":
    from torch._inductor import dynagraph as _dgm

    _oh = _dgm.DynaGraphRunner._host_step

    def _hs(self, *a, **kw):
        t0 = time.perf_counter()
        r = _oh(self, *a, **kw)
        patch["t"] += time.perf_counter() - t0
        patch["n"] += 1
        return r

    _dgm.DynaGraphRunner._host_step = _hs

    _oc = _dgm.DynaGraphRunner.__call__

    def _cc(self, *a, **kw):
        t0 = time.perf_counter()
        r = _oc(self, *a, **kw)
        call["t"] += time.perf_counter() - t0
        call["n"] += 1
        return r

    _dgm.DynaGraphRunner.__call__ = _gpu_wrap(_cc)

# The same ruler for the baseline: when DynaGraph does not take over, every forward of this region goes through
# cudagraph_trees' replay. Only with that measured too does "how much more DynaGraph costs" mean anything.
replay = {"n": 0, "t": 0.0}
if MODE != "vllm":
    from torch._inductor import cudagraph_trees as _ct

    _or = _ct.CUDAGraphNode.run

    def _rr(self, *a, **kw):
        t0 = time.perf_counter()
        r = _or(self, *a, **kw)
        replay["t"] += time.perf_counter() - t0
        replay["n"] += 1
        return r

    _ct.CUDAGraphNode.run = _rr if MODE == "dynagraph" else _gpu_wrap(_rr)

# PIN_SCRATCH=1: keep every scheduler_metadata passed into attention (FA3's
# tile_count_semaphore aliases it) alive forever, so the allocator cannot reclaim it.
# Hypothesis: the child graph captured this buffer's **pointer** but does not hold its **storage**; after harvest
# the buffer is freed and reused by someone else, so every replay reads and writes the scheduling table in memory
# that now belongs to someone else. If the hypothesis holds, the slowdown should disappear once it is pinned.
_pinned = []
if os.environ.get("PIN_SCRATCH") == "1":
    import vllm.v1.attention.backends.flash_attn as _fa

    _ofa = _fa.flash_attn_varlen_func

    def _pin(*a, **kw):
        sm = kw.get("scheduler_metadata")
        if sm is not None and len(_pinned) < 20000:
            _pinned.append(sm)
        return _ofa(*a, **kw)

    _fa.flash_attn_varlen_func = _pin

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationMode

if MODE == "vllm":
    comp = None
else:
    # CONTRACT=1: tell the engine "someone is capturing graphs" so its replay-safe metadata contract takes effect
    # (scheduler_metadata moves to a resident fixed address, refreshed in place, tail zeroed),
    # but empty the capture sizes so it never actually captures anything itself.
    comp = {
        "mode": int(CompilationMode.STOCK_TORCH_COMPILE),
        "inductor_compile_config": {"triton.cudagraphs": True},
    }
    if os.environ.get("CONTRACT") == "1":
        comp["cudagraph_mode"] = "FULL"
        # Must not be empty (vLLM validates it), so give it a minimal ladder: it only captures bs=1,
        # which our bs=8 run never uses, but use_full_cuda_graph is true, so
        # scheduler_metadata switches to the resident fixed-address buffer.
        comp["cudagraph_capture_sizes"] = [1]
    else:
        comp["cudagraph_mode"] = "NONE"

kw = dict(
    model=MODEL,
    max_model_len=2048,
    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.35")),
    enforce_eager=False,
    max_num_seqs=BS,
    disable_log_stats=True,
)
if comp is not None:
    kw["compilation_config"] = comp

llm = LLM(**kw)
# PLEN controls the prompt length. Hypothesis: DynaGraph's penalty is proportional to the padding size,
# and padding = prefill token count - decode token count (= bs).
# So a very short prompt (padding ~0) should see almost no penalty, and a long prompt a large one.
PLEN = int(os.environ.get("PLEN", "0"))
if PLEN <= 0:
    prompts = [f"Count from {i} to one hundred, one number per line:" for i in range(BS)]
else:
    prompts = [("a " * PLEN) + f"{i}" for i in range(BS)]
_tk = llm.get_tokenizer()
print(
    f"  prompt token counts {[len(_tk.encode(x)) for x in prompts[:2]]}  bs={BS}",
    flush=True,
)


def run(t):
    sp = SamplingParams(temperature=0.0, max_tokens=t, ignore_eos=True)
    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    _gpu_drain()
    return dt


# Warmup: get every capture / harvest / autotune out of the way so none of it lands in the timing.
for _ in range(2):
    run(T1)
    run(T2)
patch0, call0, replay0 = dict(patch), dict(call), dict(replay)
gpu0 = dict(gpu)

# PROFILE=1: put only steady-state decode inside the profiler window. Prefill, capture and autotune
# all finished during warmup and stay outside, otherwise the kernel stats would be all startup work.
if os.environ.get("PROFILE") == "1":
    torch.cuda.profiler.start()
    run(T2)
    torch.cuda.profiler.stop()
    _gpu_drain()
    print(f"  mode {MODE}  profile window done (one T{T2} round)", flush=True)
    raise SystemExit(0)

a, b = [], []
for _ in range(ROUNDS):
    a.append(run(T1))
    b.append(run(T2))

per_step = [(y - x) / (T2 - T1) for x, y in zip(a, b)]
med = statistics.median(per_step)
lo, hi = min(per_step), max(per_step)

extra = ""
if replay["n"] > replay0["n"]:
    rn = replay["n"] - replay0["n"]
    rt = replay["t"] - replay0["t"]
    extra += f"  |  cudagraph replay {rn} calls avg {rt / rn * 1e6:.0f} us"
if MODE == "dynagraph" and patch["n"] > patch0["n"]:
    dn = patch["n"] - patch0["n"]
    dt = patch["t"] - patch0["t"]
    cn = call["n"] - call0["n"]
    ct = call["t"] - call0["t"]
    extra += (
        f"  |  _host_step {dn} calls avg {dt / dn * 1e6:.0f} us"
        f"  runner.__call__ {cn} calls avg {ct / cn * 1e6:.0f} us"
    )

if GPUTIME and gpu["n"] > gpu0["n"]:
    gn = gpu["n"] - gpu0["n"]
    gt = gpu["t"] - gpu0["t"]
    extra += f"  |  region GPU time {gn} calls avg {gt / gn * 1e6:.0f} us"

print(
    f"  mode {MODE:<10} bs={BS}  per decode step {med * 1e6:.0f} us"
    f"  (min {lo * 1e6:.0f} / max {hi * 1e6:.0f}, {ROUNDS} rounds)"
    f"  T{T1} {statistics.median(a):.3f}s  T{T2} {statistics.median(b):.3f}s{extra}",
    flush=True,
)
