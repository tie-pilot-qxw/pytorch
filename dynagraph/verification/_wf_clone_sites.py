#!/usr/bin/env python3
"""Capture a library call once per signature and clone the other sites by repointing:
is it correct on real vLLM, and how much does it save?

Run the same set of requests (several batch sizes, greedy decoding) and compare the
tokens one by one against the reference (CLONE=0, every site captured); also count,
per harvest, how many sites were captured, how many cloned, and how long it took.

    CLONE=0|1 OUT=${DG_OUT:-/tmp/dynagraph_out}/_clone_x.json python _wf_clone_sites.py
"""
import json, os, sys, time, logging, re

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ["TORCHINDUCTOR_DYNAGRAPH_CLONE_SITES"] = os.environ.get("CLONE", "1")
os.environ["TORCHINDUCTOR_DYNAGRAPH_QUIET_HARVEST"] = os.environ.get("QUIET", "0")
_STUB = os.environ.get("DG_DEPS", "/workspace/_deps") + "/fa4site"
if os.path.isdir(_STUB) and _STUB not in sys.path:
    sys.path.insert(0, _STUB)
import torch
import torch._inductor.config as ic

ic.triton.cudagraphs = True
ic.triton.dynagraph = os.environ.get("DG", "1") == "1"
if os.environ.get("GEMM") == "triton":
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"
ic.triton.dynagraph_extern_child = True
from torch._inductor import dynagraph as _dgm
if os.environ.get("DECL") == "1":
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "serving"))
    import vllm_launches
    vllm_launches.graph_safe_metadata(int(os.environ.get("MAX_TOKENS", "1024")))
    if os.environ.get("ATTN") == "described":
        vllm_launches.use_described_attention()

if os.environ.get("TB") == "1":
    import traceback
    _of = _dgm._fallback

    def _fb(tag, detail=""):
        if tag in ("exception", "capture-failed"):
            traceback.print_exc()
        return _of(tag, detail)

    _dgm._fallback = _fb
hv = {"n": 0, "t": 0.0}
_oh = _dgm.DynaGraphRunner._harvest


import cProfile, pstats
PROF = cProfile.Profile() if os.environ.get("PROFILE") == "1" else None


def _h(self, *a, **kw):
    t0 = time.perf_counter()
    hv["in"] = True
    if PROF is not None and hv["n"] >= 2:
        PROF.enable()
    r = _oh(self, *a, **kw)
    hv["in"] = False
    if PROF is not None:
        PROF.disable()
    th = time.perf_counter() - t0
    hv["host"] = hv.get("host", 0.0) + th
    hv.setdefault("each_host", []).append(round(th * 1e3, 1))
    torch.cuda.synchronize()
    hv["t"] += time.perf_counter() - t0
    hv["n"] += 1
    hv.setdefault("each", []).append(round((time.perf_counter() - t0) * 1e3, 1))
    return r


_dgm.DynaGraphRunner._harvest = _h

served = {"ok": 0, "skip": 0, "none": 0}
FLAG = {"new": False}
_ocall = _dgm.DynaGraphRunner.__call__


CALLS = {"new": [], "hit": []}


def _call(self, inputs):
    t0 = time.perf_counter()
    FLAG["new"] = False
    r = _ocall(self, inputs)
    CALLS["new" if FLAG["new"] else "hit"].append((time.perf_counter() - t0) * 1e3)
    served["skip" if r is _dgm.SKIP_SHAPE else "none" if r is None else "ok"] += 1
    return r


_dgm.DynaGraphRunner.__call__ = _call

cl = {"n": 0, "t": 0.0, "p": 0.0}
cl_seen = []
_oc, _op = _dgm._clone_child, _dgm._clone_plan


def _c(*a, **kw):
    t0 = time.perf_counter()
    r = _oc(*a, **kw)
    cl_seen.append(1)
    cl["t"] += time.perf_counter() - t0
    cl["n"] += 1
    return r


def _p(*a, **kw):
    t0 = time.perf_counter()
    r = _op(*a, **kw)
    cl["p"] += time.perf_counter() - t0
    return r


_dgm._clone_child, _dgm._clone_plan = _c, _p

ct = {}
R = _dgm.DynaGraphRunner


def _tm(nm, f):
    def w(*a, **kw):
        t0 = time.perf_counter()
        try:
            return f(*a, **kw)
        finally:
            e = ct.setdefault(nm, [0, 0.0])
            e[0] += 1
            e[1] += time.perf_counter() - t0
    return w


for _nm in ("_recapture", "_arena_views", "_shape_inputs", "_capture_ranges", "_apply_inline", "_prepare_inline", "_site_operands"):
    setattr(R, _nm, _tm(_nm, getattr(R, _nm)))
_ori = R._run_intercepted


def _ri(self, args, views, on_extern, *rest):
    t0 = time.perf_counter()
    tag = "H" if hv.get("in") else "C"

    def oe(i, fn, a, kw):
        t1 = time.perf_counter()
        n0 = len(cl_seen)
        r = on_extern(i, fn, a, kw)
        site = self.extern_sites[i]
        kind = "clone" if len(cl_seen) > n0 else site.split(".")[-2 if site.startswith("ops:") else -1]
        e = ct.setdefault(f"{tag} on_extern[{kind}]", [0, 0.0])
        e[0] += 1
        e[1] += time.perf_counter() - t1
        return r

    r = _ori(self, args, views, oe, *rest)
    e = ct.setdefault(f"{tag} _run_intercepted(all)", [0, 0.0])
    e[0] += 1
    e[1] += time.perf_counter() - t0
    return r


R._run_intercepted = _ri

_opi = R._prepare_inline


def _pi(self, i, *a, **kw):
    t0 = time.perf_counter()
    try:
        return _opi(self, i, *a, **kw)
    finally:
        nm = "prep[" + self.extern_sites[i].split(".")[1] + "]"
        e = ct.setdefault(nm, [0, 0.0]); e[0] += 1; e[1] += time.perf_counter() - t0


R._prepare_inline = _pi

_oai = R._apply_inline


def _ai(self, ex, env, hkey, *a, **kw):
    fresh = hkey not in self._inline_ops
    n0 = len(self._inline_prepared)
    t0 = time.perf_counter()
    try:
        return _oai(self, ex, env, hkey, *a, **kw)
    finally:
        dt = time.perf_counter() - t0
        if fresh:
            FLAG["new"] = True
        kind = "new-shape" if fresh else ("re-bind" if len(self._inline_prepared) != n0 else "hit")
        e = ct.setdefault(f"apply[{kind}]", [0, 0.0]); e[0] += 1; e[1] += dt


R._apply_inline = _ai
from torch.utils import _capture_launch as _clm
for _nm in ("_record_graph", "read_node", "_chain_order"):
    setattr(_clm, _nm, _tm("cl." + _nm, getattr(_clm, _nm)))
_CG = torch.cuda.CUDAGraph
for _nm in ("capture_begin", "capture_end_pre", "capture_end_post", "capture_end"):
    def _mk(nm, f=getattr(_CG, _nm)):
        def w(self, *a, **kw):
            t0 = time.perf_counter()
            try:
                return f(self, *a, **kw)
            finally:
                e = ct.setdefault(nm, [0, 0.0])
                e[0] += 1
                e[1] += time.perf_counter() - t0
        return w
    setattr(_CG, _nm, _mk(_nm))

lines = []


class Grab(logging.Handler):
    def emit(self, rec):
        m = rec.getMessage()
        if "captured" in m and "cloned" in m or "fallback" in m or "mismatch" in m:
            lines.append(m)


lg = logging.getLogger("torch._inductor.dynagraph")
lg.setLevel(logging.INFO)
lg.addHandler(Grab())

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationMode

llm = LLM(
    model=os.environ.get("MODEL", "Qwen/Qwen3-0.6B"),
    max_model_len=1024,
    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.2")),
    enforce_eager=False,
    max_num_seqs=16,
    disable_log_stats=True,
    compilation_config={
        "mode": int(CompilationMode.STOCK_TORCH_COMPILE),
        "inductor_compile_config": dict(
            {"triton.cudagraphs": True},
            **({"max_autotune_gemm": True, "max_autotune_gemm_backends": "TRITON"}
               if os.environ.get("GEMM") == "triton" else {}),
        ),
        "cudagraph_mode": "NONE",
    },
)
sp = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
out = {}
for bs in (1, 3, 5, 8, 13, 16):
    ps = [f"Tell me fact number {i} about the ocean:" for i in range(bs)]
    res = llm.generate(ps, sp, use_tqdm=False)
    out[bs] = [list(r.outputs[0].token_ids) for r in res]

cap = clo = 0
for m in lines:
    g = re.search(r"captured (\d+) sites, cloned (\d+)", m)
    if g:
        cap += int(g.group(1))
        clo += int(g.group(2))
fb = [m for m in lines if "fallback" in m]
print(f"served {served}")
for _k, _v in CALLS.items():
    if _v:
        _v2 = sorted(_v)
        print(f"  __call__[{_k}]: n={len(_v)} median {_v2[len(_v)//2]:.2f}ms min {_v2[0]:.2f} max {_v2[-1]:.2f}")
if os.environ.get("ATTN") == "described":
    _ts = sorted(vllm_launches.DESCRIBE_TIME[2:])
    if _ts:
        print(f"  fa3 fwd_describe median {_ts[len(_ts)//2]*1e6:.0f}us p90 {_ts[int(len(_ts)*.9)]*1e6:.0f}us max {_ts[-1]*1e6:.0f}us")
    print(f"  fa3 fwd_describe: {vllm_launches.DESCRIBE_TIME[1]} x {vllm_launches.DESCRIBE_TIME[0] / max(vllm_launches.DESCRIBE_TIME[1], 1) * 1e6:.0f}us")
print(f"QUIET={os.environ.get('QUIET','0')} host-only {hv.get('host',0)*1e3:.0f}ms per-harvest host {hv.get('each_host')}")
print(f"CLONE={os.environ['CLONE']} harvests={hv['n']} harvest_time={hv['t']*1e3:.0f}ms "
      f"captured={cap} cloned={clo} fallbacks={len(fb)}")
if PROF is not None:
    pstats.Stats(PROF).sort_stats("tottime").print_stats(18)
    pstats.Stats(PROF).sort_stats("cumulative").print_stats(30)
for k, (n, t) in ct.items():
    print(f"  {k}: {n} x {t / max(n, 1) * 1e6:.0f}us = {t * 1e3:.0f}ms")
print("per harvest ms", hv.get("each"))
print(f"clone {cl['n']} x {cl['t']/max(cl['n'],1)*1e6:.0f}us, plan total {cl['p']*1e3:.1f}ms")
for m in fb[:8]:
    print("  ", m[:200])
DG_OUT = os.environ.get("DG_OUT", "/tmp/dynagraph_out")
os.makedirs(DG_OUT, exist_ok=True)
json.dump({"tokens": out, "harvests": hv["n"], "ms": hv["t"] * 1e3, "cap": cap, "clo": clo},
          open(os.environ.get("OUT", os.path.join(DG_OUT, "_clone.json")), "w"))
