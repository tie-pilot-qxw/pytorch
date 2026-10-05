#!/usr/bin/env python3
"""Print the argument text of each kind of extern call site in the vLLM region, to see what expressions they are."""
import collections, os, re, sys

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

runners = []
_oi = _dgm.DynaGraphRunner.__init__


def init(self, *a, **kw):
    _oi(self, *a, **kw)
    runners.append(self)


_dgm.DynaGraphRunner.__init__ = init
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationMode

llm = LLM(model="Qwen/Qwen3-0.6B", max_model_len=1024,
          gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.12")),
          enforce_eager=False, max_num_seqs=4, disable_log_stats=True,
          compilation_config={"mode": int(CompilationMode.STOCK_TORCH_COMPILE),
                              "cudagraph_mode": "NONE",
                              "inductor_compile_config": {"triton.cudagraphs": True}})
llm.generate(["hi"], SamplingParams(max_tokens=2), use_tqdm=False)

r = max(runners, key=lambda x: len(x.extern_sites))
print(f"\n  runner: {len(r.extern_sites)} sites")
shown = collections.Counter()
for line in r.body.splitlines():
    code = line.split("#", 1)[0]
    for _at, name, buf, args in _dgm._scan_line(code):
        if shown[name] < 3:
            shown[name] += 1
            print(f"  [{name}] {code.strip()[:230]}")
kinds = collections.Counter()
for line in r.body.splitlines():
    code = line.split("#", 1)[0]
    for _at, name, buf, args in _dgm._scan_line(code):
        depth, j = 1, 0
        while j < len(args) and depth:
            depth += {"(": 1, ")": -1}.get(args[j], 0)
            j += 1
        for a in _dgm._split_args(args[: j - 1]):
            a = a.strip()
            v = a.split("=", 1)[1].strip() if re.match(r"^\w+\s*=[^=]", a) else a
            if re.fullmatch(r"\w+", v) and (v in r.layouts or v in r.views or v in r.argv or v in r.alias):
                k = "name (buffer/view/input)"
            elif v.startswith("reinterpret_tensor("):
                k = "reinterpret_tensor(...)"
            elif re.fullmatch(r"-?[\d.e+-]+|'[^']*'|\"[^\"]*\"|True|False|None", v):
                k = "literal"
            else:
                k = f"other: {v[:60]}"
            kinds[k] += 1
print("\n  Expression kinds across all call-site arguments:")
for k, c in kinds.most_common():
    print(f"    {c:>5}  {k}")
