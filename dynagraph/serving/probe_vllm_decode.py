#!/usr/bin/env python3
"""Decode in a real engine: hand vLLM's regions to DynaGraph and see whether one graph can serve a range of batch sizes.

The target to beat is vLLM's own approach. By default it captures a ladder of cudagraphs --
`[1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)`, max = min(max_num_seqs*2, 512),
i.e. **~51 graphs**, and every batch size in between is padded up (B=9 runs the B=16 graph).
DynaGraph aims for 1 graph and no padding.

This probe does not write its own decode: a hand-written one would not match the engine and would prove nothing. It
starts vLLM directly with our own torch, turns compilation on, and counts three things: how many graphs were captured,
which kernels became opaque sites, and what the fallback tags are.

    MODE=vllm      python probe_vllm_decode.py   # vLLM default (piecewise + its own graph capture)
    MODE=stock     python probe_vllm_decode.py   # plain torch.compile, inductor captures, DynaGraph off
    MODE=dynagraph python probe_vllm_decode.py   # same as above, DynaGraph on
    MODE=piecewise python probe_vllm_decode.py   # keep vLLM's piecewise compilation, but hand capture to
                                                 # inductor (+DynaGraph). Fallback option if stock does not work.

`MODEL` picks the model, `BS` the decode batch sizes to run (comma-separated).
"""
import os, sys, logging, collections

# Run the engine in the same process: otherwise neither the inductor config nor the log handlers reach the worker.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# flashinfer is not installed (its wheel was built against a different torch), and the sampler's probing path
# is an unguarded import. vLLM's own top-p/top-k implementation works just as well.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# The flash_attn 2.7.4 in the container's dist-packages was built for torch 2.11; importing it fails with an
# undefined symbol, and transformers probes for it. fa4site has an empty flash_attn package; putting it
# first on sys.path shadows the broken one (see dynagraph/docs/notes/SETUP.md).
# Note this has nothing to do with the attention backend: vLLM uses its own vllm_flash_attn build (FA2/FA3).
_STUB = os.environ.get("DG_DEPS", "/workspace/_deps") + "/fa4site"
if os.path.isdir(_STUB) and _STUB not in sys.path:
    sys.path.insert(0, _STUB)

MODE = os.environ.get("MODE", "dynagraph")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
BS = [int(b) for b in os.environ.get("BS", "1,2,3,4,7,5,2").split(",")]

import torch
import torch._inductor.config as ic

logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph")
lg.setLevel(logging.INFO)

tags, opaque, notes, served = [], [], [], []
sites = set()


class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "fallback [" in m:
            tags.append(m.split("[", 1)[1].split("]", 1)[0])
        elif m.startswith("DynaGraph opaque kernel "):
            opaque.append(m[len("DynaGraph opaque kernel ") :].split(":", 1)[0])
        elif m.startswith("DynaGraph served"):
            served.append(m)
        elif m.startswith("DynaGraph site "):
            sites.add(m.split()[2].rstrip(":"))
        elif m.startswith("DynaGraph captured graph") or " -> host" in m:
            notes.append(m[:120])


# The "served" message is logged by cudagraph_trees, not the dynagraph logger, so attach to both.
_grab = _Grab()
lg.addHandler(_grab)
_lg_ct = logging.getLogger("torch._inductor.cudagraph_trees")
_lg_ct.setLevel(logging.INFO)
_lg_ct.addHandler(_grab)

# Every cudagraph capture, whoever starts it: both vLLM's own captures and inductor cudagraph_trees'
# captures go through here, so this is the only count that measures both paths with the same ruler.
n_cap = {"v": 0}
_orig_cg = torch.cuda.CUDAGraph.__init__


TRACE_CAP = int(os.environ.get("TRACE_CAP", "0"))


def _cg(self, *a, **kw):
    n_cap["v"] += 1
    if TRACE_CAP and n_cap["v"] in (1, TRACE_CAP, TRACE_CAP + 1):
        import traceback

        print(f"  --- CUDAGraph #{n_cap['v']} from:", flush=True)
        for ln in traceback.format_stack()[-7:-1]:
            print("      " + ln.strip().replace("\n", " | ")[:150], flush=True)
    return _orig_cg(self, *a, **kw)


torch.cuda.CUDAGraph.__init__ = _cg

# A count specific to inductor's cudagraph_trees: the CUDAGraph count above counts every capture,
# this one counts only inductor's, which is what tells "captured by vLLM" apart from "captured by us".
from torch._inductor import cudagraph_trees as _ct

n_ind = {"v": 0}
_orig_node = _ct.CUDAGraphNode.__init__


def _node(self, *a, **kw):
    n_ind["v"] += 1
    return _orig_node(self, *a, **kw)


_ct.CUDAGraphNode.__init__ = _node

# The actual shape of every harvest: the only thing that lets us look at prefill and decode separately.
# For decode num_tokens equals the batch size; for prefill it equals the batch's total token count.
from torch._inductor import dynagraph as _dgm

harvests = []  # (env, how many CUDAGraphs this harvest took)
_orig_harvest = _dgm.DynaGraphRunner._harvest


def _harv(self, env, key, hkey, inputs, *a, **kw):
    before = n_cap["v"]
    r = _orig_harvest(self, env, key, hkey, inputs, *a, **kw)
    harvests.append((dict(env), n_cap["v"] - before))
    return r


_dgm.DynaGraphRunner._harvest = _harv

if MODE != "vllm":
    # Set globally, not passed through vLLM's inductor_compile_config: in testing, that path did not take effect
    # (not even cudagraph_trees' "skipping" log showed up, so it was never invoked at all).
    ic.triton.cudagraphs = True

if MODE in ("dynagraph", "piecewise"):
    ic.triton.dynagraph = True
    # EXTERN_CHILD=0: extern calls are not captured as child graphs; they stay in the main graph and get their params patched.
    ic.triton.dynagraph_extern_child = os.environ.get("EXTERN_CHILD", "1") == "1"
    ic.triton.dynagraph_update = os.environ.get("TORCHINDUCTOR_DYNAGRAPH_UPDATE", "auto")

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationMode

if MODE == "vllm":
    comp = None  # vLLM's default: piecewise compilation + its own ladder of captures
elif MODE == "piecewise":
    # Compilation unchanged (vLLM's own backend, split at attention); only the capture work moves
    # from vLLM to inductor's cudagraph_trees -- DynaGraph hooks into that layer.
    comp = {
        "cudagraph_mode": "NONE",
        "inductor_compile_config": {"triton.cudagraphs": True},
    }
else:
    # Plain torch.compile, inductor turns on cudagraphs itself (DynaGraph hooks into that layer),
    # and vLLM is kept from capturing a second time.
    comp = {
        "mode": int(CompilationMode.STOCK_TORCH_COMPILE),
        "cudagraph_mode": "NONE",
        "inductor_compile_config": {"triton.cudagraphs": True},
    }

kw = dict(
    model=MODEL,
    max_model_len=1024,
    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.35")),
    enforce_eager=False,
    max_num_seqs=max(BS) if BS else 8,
    disable_log_stats=True,
)
if comp is not None:
    kw["compilation_config"] = comp

err = None
texts = []
try:
    llm = LLM(**kw)
    n_after_init = n_cap["v"]
    sp = SamplingParams(temperature=0.0, max_tokens=int(os.environ.get("MAXTOK", "16")))
    for b in BS:
        prompts = [f"Count from {i} to ten:" for i in range(b)]
        before = n_cap["v"]
        outs = llm.generate(prompts, sp)
        texts.append((b, len(outs), outs[0].outputs[0].text[:24].replace("\n", " ")))
        print(
            f"    bs={b:3d}  returned {len(outs)}  new captures this round {n_cap['v'] - before}  "
            f"total {n_cap['v']}",
            flush=True,
        )
except Exception as e:
    import traceback

    err = f"{type(e).__name__}: {e}"
    traceback.print_exc()
    n_after_init = n_cap["v"]

print(
    f"  mode {MODE}  model {MODEL}  batch {BS}\n"
    f"  cudagraph captures at init {n_after_init} total {n_cap['v']}  inductor recordings {n_ind['v']}  "
    f"opaque {len(set(opaque))}  extern sites {len(sites)}  "
    f"regions served by one graph {len(served)}  tags {sorted(set(tags)) or '-'}"
)
if harvests:
    import collections as _c

    per_shape = _c.Counter()
    cost = _c.Counter()
    for env, n in harvests:
        k = tuple(sorted(env.items()))
        per_shape[k] += 1
        cost[k] += n
    print(f"  harvests {len(harvests)}, distinct shapes {len(per_shape)}:")
    for k, c in per_shape.most_common(12):
        print(f"    {dict(k)}  harvests {c}  child graphs {cost[k]} in total")
for m in notes[:4]:
    print(f"    {m}")
if err:
    print(f"  error {err[:300]}")
    sys.exit(1)
if not texts or any(n != b for b, n, _ in texts):
    print("  FAILED: some requests did not return")
    sys.exit(1)
print("  sample output:", texts[0][2])
sys.exit(0)
