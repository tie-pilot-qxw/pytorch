#!/usr/bin/env python3
"""Do the arguments `_site_operands` builds from the call-site text match, one by one, what the wrapper really passes in?

For every extern call of every harvest: build one copy and compare it with the real (a, kw) by address, shape, stride,
dtype; non-tensor arguments are compared by value. Only a match shows that "the operands can be obtained without running the wrapper".
"""
import collections, os, sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
_STUB = os.environ.get("DG_DEPS", "/workspace/_deps") + "/fa4site"
if os.path.isdir(_STUB) and _STUB not in sys.path:
    sys.path.insert(0, _STUB)
import torch
import torch._inductor.config as ic

ic.triton.cudagraphs = True
ic.triton.dynagraph = True
ic.triton.dynagraph_extern_child = True
from torch._inductor import dynagraph as _dgm

stats = collections.defaultdict(collections.Counter)
first_bad = {}
_cur = {"env": None}


def same(x, y):
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            return f"one is a tensor and the other is not: {type(x).__name__} vs {type(y).__name__}"
        for what, u, v in (("address", x.data_ptr(), y.data_ptr()),
                           ("shape", tuple(x.shape), tuple(y.shape)),
                           ("stride", tuple(x.stride()), tuple(y.stride())),
                           ("dtype", x.dtype, y.dtype)):
            if u != v:
                return f"{what} {u} vs {v}"
        return None
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            return "length differs"
        for u, v in zip(x, y):
            r = same(u, v)
            if r:
                return r
        return None
    return None if x == y else f"value {x!r} vs {y!r}"


_ori = _dgm.DynaGraphRunner._run_intercepted


def ri(self, args, views, on_extern):
    def wrapped(i, fn, a, kw):
        name = self.extern_sites[i].replace("ops:vllm.", "").replace(".default", "")
        env = _cur["env"]
        if env is not None and isinstance(views, list):
            st = stats[name]
            st["calls"] += 1
            got = self._site_operands(i, env, views, args)
            if got is None:
                st["unbuildable"] += 1
            else:
                pa, pk = got
                why = same(list(pa), list(a))
                if why is None and sorted(pk) != sorted(kw):
                    why = f"keywords {sorted(pk)} vs {sorted(kw)}"
                if why is None:
                    for k in kw:
                        why = same(pk[k], kw[k])
                        if why:
                            why = f"{k}: {why}"
                            break
                if why is None:
                    st["exact"] += 1
                else:
                    st["mismatch"] += 1
                    first_bad.setdefault(name, (self.site_args[i][:120], why))
        return on_extern(i, fn, a, kw)
    return _ori(self, args, views, wrapped)


_dgm.DynaGraphRunner._run_intercepted = ri
_oh = _dgm.DynaGraphRunner._harvest


def h(self, env, *a, **kw):
    _cur["env"] = dict(env)
    try:
        return _oh(self, env, *a, **kw)
    finally:
        _cur["env"] = None


_dgm.DynaGraphRunner._harvest = h

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationMode

llm = LLM(model=os.environ.get("MODEL", "Qwen/Qwen3-0.6B"), max_model_len=1024,
          gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.12")),
          enforce_eager=False, max_num_seqs=8, disable_log_stats=True,
          compilation_config={"mode": int(CompilationMode.STOCK_TORCH_COMPILE),
                              "cudagraph_mode": "NONE",
                              "inductor_compile_config": {"triton.cudagraphs": True}})
sp = SamplingParams(temperature=0.0, max_tokens=8)
for b in [int(v) for v in os.environ.get("BS", "1,2,3,4,5").split(",")]:
    llm.generate([f"Count from {i} to ten:" for i in range(b)], sp, use_tqdm=False)

print("\n  operands built from text vs what the wrapper really passed in")
for name, st in stats.items():
    print(f"    {name:<32} calls {st['calls']:>5}  exact {st['exact']:>5}  "
          f"mismatch {st['mismatch']:>4}  unbuildable {st['unbuildable']:>4}")
for name, (txt, why) in first_bad.items():
    print(f"    [{name}] first mismatch: {why}\n        {txt}")
