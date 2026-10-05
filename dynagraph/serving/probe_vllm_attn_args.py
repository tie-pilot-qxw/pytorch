#!/usr/bin/env python3
"""What attention actually receives: print the arguments that get baked into the graph at capture time.

During replay Python does not run, so nothing is visible. But **it does run during capture and warmup**,
and that is exactly when host scalars and pointers get baked into the kernel arguments. Hooking
`flash_attn_varlen_func` shows what each of the two capture modes pins down.

Compared:
  MODE=eager      enforce_eager, no graph capture at all -- every call is fresh
  MODE=stock      plain torch.compile + inductor graph capture, DynaGraph off
  MODE=dynagraph  same, DynaGraph on

    MODE=stock python serving/probe_vllm_attn_args.py
"""
import os, sys, logging

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
_STUB = os.path.join(os.environ.get("DG_DEPS", "/workspace/_deps"), "fa4site")
if os.path.isdir(_STUB) and _STUB not in sys.path:
    sys.path.insert(0, _STUB)

MODE = os.environ.get("MODE", "dynagraph")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-14B")
BS = int(os.environ.get("BS", "8"))

import torch
import torch._inductor.config as ic

logging.basicConfig(level=logging.WARNING)

if MODE in ("stock", "dynagraph"):
    ic.triton.cudagraphs = True
if MODE == "dynagraph":
    ic.triton.dynagraph = True
    ic.triton.dynagraph_extern_child = True

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationMode
import vllm.v1.attention.backends.flash_attn as fa

seen = []
_orig = fa.flash_attn_varlen_func


def _al(t):
    """Pointer alignment: FA's fast path is sensitive to the alignment of q/k/v."""
    if t is None:
        return None
    p = t.data_ptr()
    for a in (256, 128, 64, 32, 16):
        if p % a == 0:
            return a
    return p % 16


def spy(*a, **kw):
    def g(name, idx):
        return kw[name] if name in kw else (a[idx] if len(a) > idx else None)

    q = g("q", 0)
    sk = kw.get("seqused_k")
    seen.append(
        dict(
            q_shape=tuple(q.shape) if q is not None else None,
            q_align=_al(q),
            k_align=_al(g("k", 1)),
            out_align=_al(kw.get("out")),
            max_seqlen_q=kw.get("max_seqlen_q"),
            max_seqlen_k=kw.get("max_seqlen_k"),
            num_splits=kw.get("num_splits"),
            causal=kw.get("causal"),
            sched=None if kw.get("scheduler_metadata") is None
            else tuple(kw["scheduler_metadata"].shape),
            # No D2H copy is allowed during capture (it raises "Cannot copy between CPU and CUDA
            # tensors during CUDA graph capture"), so values are read only when not capturing.
            seqused_k_head=(
                sk[:4].tolist()
                if sk is not None and not torch.cuda.is_current_stream_capturing()
                else None
            ),
            block_table=tuple(kw["block_table"].shape)
            if kw.get("block_table") is not None
            else None,
        )
    )
    return _orig(*a, **kw)


fa.flash_attn_varlen_func = spy

kw = dict(
    model=MODEL,
    max_model_len=2048,
    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.6")),
    max_num_seqs=BS,
    disable_log_stats=True,
)
if MODE == "eager":
    kw["enforce_eager"] = True
else:
    kw["enforce_eager"] = False
    kw["compilation_config"] = {
        "mode": int(CompilationMode.STOCK_TORCH_COMPILE),
        "cudagraph_mode": "NONE",
        "inductor_compile_config": {"triton.cudagraphs": True},
    }

llm = LLM(**kw)
prompts = [f"Count from {i} to one hundred, one number per line:" for i in range(BS)]
sp = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
n_init = len(seen)
llm.generate(prompts, sp, use_tqdm=False)

print(f"\n  mode {MODE}  attention calls that reached python: {len(seen)} "
      f"(startup {n_init}, during generate {len(seen) - n_init})")
# Only the generate phase, listed by distinct argument combination
uniq = {}
for d in seen[n_init:]:
    k = tuple(sorted((x, str(y)) for x, y in d.items() if x != "seqused_k_head"))
    uniq.setdefault(k, [0, d])
    uniq[k][0] += 1
for k, (cnt, d) in list(uniq.items())[:6]:
    print(f"    x{cnt:<4} q{d['q_shape']} align q{d['q_align']}/k{d['k_align']}/o{d['out_align']}"
          f"  max_q={d['max_seqlen_q']} max_k={d['max_seqlen_k']}"
          f"  splits={d['num_splits']} causal={d['causal']} sched={d['sched']}"
          f"  bt={d['block_table']}  seq[:4]={d['seqused_k_head']}")
if not uniq:
    print("    no python calls during generate -- everything replays inside the graph")
