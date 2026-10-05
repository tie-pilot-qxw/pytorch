#!/usr/bin/env python3
"""How long does recapturing a whole cudagraph take (compare with the ~50 ms of DynaGraph harvest).

MODE=vllm: vLLM's own FULL cudagraph, captured one capture size at a time, each size timed
          (covering CUDAGraph.capture_begin..capture_end plus the warm-up forward before it).
MODE=trees: Inductor cudagraph_trees (dynagraph off), records one graph per new shape, times record.
"""
import os, sys, time, statistics
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
_STUB = os.environ.get("DG_DEPS", "/workspace/_deps") + "/fa4site"
if os.path.isdir(_STUB) and _STUB not in sys.path:
    sys.path.insert(0, _STUB)
import torch
MODE = os.environ.get("MODE", "vllm")
T = {"cap": [], "inst": [], "rec": []}

_CG = torch.cuda.CUDAGraph
_cb, _ce = _CG.capture_begin, _CG.capture_end
_cur = {}
def cb(self, *a, **kw):
    _cur[id(self)] = time.perf_counter()
    return _cb(self, *a, **kw)
_last = [None]


def ce(self, *a, **kw):
    r = _ce(self, *a, **kw)
    torch.cuda.synchronize()
    now = time.perf_counter()
    T["cap"].append((now - _cur.pop(id(self), now)) * 1e3)
    if _last[0] is not None:
        # One whole capture-loop iteration: the eager warm-up forward, the
        # capture forward, instantiate.
        T.setdefault("per_size", []).append((now - _last[0]) * 1e3)
    _last[0] = now
    return r
_CG.capture_begin, _CG.capture_end = cb, ce
_ins = _CG.instantiate
def ins(self, *a, **kw):
    t0 = time.perf_counter(); r = _ins(self, *a, **kw)
    T["inst"].append((time.perf_counter() - t0) * 1e3); return r
_CG.instantiate = ins

from vllm import LLM, SamplingParams
REPLAY = []
if MODE == "vllm":
    import vllm.compilation.cuda_graph as _cg
    _ow = _cg.CUDAGraphWrapper.__call__

    def _w(self, *a, **k):
        t0 = time.perf_counter()
        r = _ow(self, *a, **k)
        if STEADY[0]:
            REPLAY.append((time.perf_counter() - t0) * 1e3)
        return r

    _cg.CUDAGraphWrapper.__call__ = _w
    _orp = _CG.replay

    def _rp(self, *a, **k):
        t0 = time.perf_counter()
        r = _orp(self, *a, **k)
        if STEADY[0]:
            REPLAY.append((time.perf_counter() - t0) * 1e3)
        return r

    _CG.replay = _rp
STEADY = [False]
from vllm.config.compilation import CompilationMode
if MODE == "vllm":
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 3, 5, 13, 24, 40, 80, 100, 120]
    comp = {"cudagraph_mode": "FULL", "cudagraph_capture_sizes": sizes}
else:
    import torch._inductor.config as ic
    ic.triton.cudagraphs = True
    from torch._inductor import cudagraph_trees as ct
    _rf = ct.CUDAGraphTreeManager.record_function
    def rf(self, *a, **kw):
        t0 = time.perf_counter(); r = _rf(self, *a, **kw); torch.cuda.synchronize()
        T["rec"].append((time.perf_counter() - t0) * 1e3); return r
    ct.CUDAGraphTreeManager.record_function = rf
    comp = {"mode": int(CompilationMode.STOCK_TORCH_COMPILE),
            "inductor_compile_config": {"triton.cudagraphs": True}, "cudagraph_mode": "NONE"}
t0 = time.perf_counter()
llm = LLM(model="Qwen/Qwen3-0.6B", max_model_len=1024, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.15")),
          max_num_seqs=16, disable_log_stats=True, compilation_config=comp)
print(f"init {time.perf_counter()-t0:.1f}s")
if MODE == "vllm":
    STEADY[0] = True
    sp = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
    for bs in (1, 3, 5, 8, 13, 16):
        llm.generate([f"fact {i} about the ocean:" for i in range(bs)], sp, use_tqdm=False)
    _r = sorted(REPLAY)
    if _r:
        print(f"vllm replay call: n={len(_r)} median {_r[len(_r)//2]:.3f}ms min {_r[0]:.3f} max {_r[-1]:.3f}")
if MODE == "trees":
    sp = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
    for bs in (1, 3, 5, 8, 13, 16):
        llm.generate([f"fact {i} about the ocean:" for i in range(bs)], sp, use_tqdm=False)
for k, v in T.items():
    if v:
        print(f"{MODE} {k}: n={len(v)} median={statistics.median(v):.1f}ms "
              f"min={min(v):.1f} max={max(v):.1f} all={[round(x,1) for x in v[:20]]}")
