#!/usr/bin/env python3
"""Where does harvest time go: per-site warm-up + capture, or running the wrapper itself?

This decides whether tier 2 ("do not run the wrapper, call this op directly once") is worth doing. If most of it is
per-site capture (unavoidable), then skipping the Triton kernels in the wrapper gains little.

Run the probe_vllm_latency workload (serving/probe_vllm_latency.py) and tally along the way:
  - total time and call count of `_harvest`
  - the part of it between capture_begin..capture_end (once per site, unavoidable)
  - the difference = cost of running the wrapper (Triton launches, allocation, eager ops)
"""
import os, sys, time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
_STUB = os.environ.get("DG_DEPS", "/workspace/_deps") + "/fa4site"
if os.path.isdir(_STUB) and _STUB not in sys.path:
    sys.path.insert(0, _STUB)

import torch

t = {"harvest": 0.0, "n": 0, "cap": 0.0, "ncap": 0, "warm": 0.0, "nwarm": 0}

_cb = torch.cuda.CUDAGraph.capture_begin
_ce = torch.cuda.CUDAGraph.capture_end
_open = {"t": 0.0}


def cb(self, *a, **kw):
    r = _cb(self, *a, **kw)
    _open["t"] = time.perf_counter()
    return r


def ce(self, *a, **kw):
    r = _ce(self, *a, **kw)
    if _open["t"]:
        t["cap"] += time.perf_counter() - _open["t"]
        t["ncap"] += 1
        _open["t"] = 0.0
    return r


torch.cuda.CUDAGraph.capture_begin = cb
torch.cuda.CUDAGraph.capture_end = ce

from torch._inductor import dynagraph as _dgm
import torch._inductor.config as ic

ic.triton.cudagraphs = True
ic.triton.dynagraph = True
ic.triton.dynagraph_extern_child = True

_oh = _dgm.DynaGraphRunner._harvest


def h(self, *a, **kw):
    t0 = time.perf_counter()
    try:
        return _oh(self, *a, **kw)
    finally:
        t["harvest"] += time.perf_counter() - t0
        t["n"] += 1


_dgm.DynaGraphRunner._harvest = h

# the extern calls themselves (the warm-up calls + the one inside capture)
_ori = _dgm.DynaGraphRunner._run_intercepted


def ri(self, args, views, on_extern):
    def wrapped(i, fn, a, kw):
        t0 = time.perf_counter()
        try:
            return on_extern(i, fn, a, kw)
        finally:
            t["warm"] += time.perf_counter() - t0
            t["nwarm"] += 1

    return _ori(self, args, views, wrapped)


_dgm.DynaGraphRunner._run_intercepted = ri

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationMode

llm = LLM(
    model=os.environ.get("MODEL", "Qwen/Qwen3-0.6B"),
    max_model_len=1024,
    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.3")),
    enforce_eager=False,
    max_num_seqs=8,
    disable_log_stats=True,
    compilation_config={
        "mode": int(CompilationMode.STOCK_TORCH_COMPILE),
        "cudagraph_mode": "NONE",
        "inductor_compile_config": {"triton.cudagraphs": True},
    },
)
sp = SamplingParams(temperature=0.0, max_tokens=8)
for b in [int(v) for v in os.environ.get("BS", "1,2,3,4").split(",")]:
    llm.generate([f"Count from {i} to ten:" for i in range(b)], sp, use_tqdm=False)

hv, cap, warm = t["harvest"], t["cap"], t["warm"]
print(f"\n  _harvest called {t['n']} times, {hv * 1000:.0f} ms total")
print(f"    of which on_extern (warm-up+capture) {t['nwarm']} calls, {warm * 1000:.0f} ms "
      f"({warm / hv * 100 if hv else 0:.0f}%)")
print(f"      of which capture_begin..end {t['ncap']} times, {cap * 1000:.0f} ms "
      f"({cap / hv * 100 if hv else 0:.0f}%)")
print(f"    remaining cost of running the wrapper (Triton launch/alloc/eager) "
      f"{(hv - warm) * 1000:.0f} ms ({(hv - warm) / hv * 100 if hv else 0:.0f}%)")
