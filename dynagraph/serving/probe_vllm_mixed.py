#!/usr/bin/env python3
"""vLLM mixed prefill + decode batches: how much do piecewise graphs lose against full graphs? (without DynaGraph)

A batch of requests with ShareGPT's length distribution is submitted at once, with chunked prefill on: every step is a
few prefill chunks + a batch of decodes, and the token count differs every step. Compares vLLM's own cudagraph modes:

  NONE                 compiled, no graph capture
  PIECEWISE            graph split at attention, attention runs eagerly between the pieces
  FULL_AND_PIECEWISE   vLLM default: full graph for pure decode, piecewise for mixed batches
  FULL                 full graph for mixed batches too (supported by the FA3 backend, AttentionCGSupport.ALWAYS)

max_cudagraph_capture_size = max_num_batched_tokens, so every step has a graph available (padded to the next capture size).

  CG=PIECEWISE MODEL=Qwen/Qwen3-0.6B MBT=2048 NREQ=256 STEPS=1 python probe_vllm_mixed.py

With STEPS=1, records the token count and time of every step (syncs every step, so only relative values count).
"""
import json
import os
import random
import statistics
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
_STUB = os.environ.get("DG_DEPS", "/workspace/_deps") + "/fa4site"
if os.path.isdir(_STUB) and _STUB not in sys.path:
    sys.path.insert(0, _STUB)

CG = os.environ.get("CG", "FULL_AND_PIECEWISE")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
MBT = int(os.environ.get("MBT", "2048"))
CAP = int(os.environ.get("CAP", str(MBT)))
NREQ = int(os.environ.get("NREQ", "256"))
MAXSEQ = int(os.environ.get("MAXSEQ", "128"))
MAXIN = int(os.environ.get("MAXIN", "1024"))
MAXOUT = int(os.environ.get("MAXOUT", "256"))
STEPS = os.environ.get("STEPS", "0") == "1"
SHAREGPT = os.environ.get(
    "SHAREGPT",
    os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/"
        "192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json",
    ),
)

import torch  # noqa: E402

steps: list[tuple[int, float]] = []
if STEPS:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _orig = GPUModelRunner.execute_model

    def _timed(self, scheduler_output, *a, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = _orig(self, scheduler_output, *a, **kw)
        torch.cuda.synchronize()
        n = scheduler_output.total_num_scheduled_tokens
        if n:
            steps.append((n, time.perf_counter() - t))
        return r

    GPUModelRunner.execute_model = _timed

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402


def workload(tok):
    """(prompt token ids, output length) per request, lengths from ShareGPT's first turns."""
    rng = random.Random(0)
    data = json.load(open(SHAREGPT))
    rng.shuffle(data)
    out = []
    for d in data:
        c = d.get("conversations") or []
        if len(c) < 2 or c[0].get("from") != "human":
            continue
        p = tok(c[0]["value"]).input_ids[:MAXIN]
        o = len(tok(c[1]["value"]).input_ids)
        if len(p) < 4 or o < 4:
            continue
        out.append((p, min(o, MAXOUT)))
        if len(out) == NREQ:
            break
    return out


def main():
    t0 = time.perf_counter()
    llm = LLM(
        MODEL,
        max_model_len=MAXIN + MAXOUT + 16,
        max_num_batched_tokens=MBT,
        max_num_seqs=MAXSEQ,
        enable_chunked_prefill=True,
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.5")),
        compilation_config={"cudagraph_mode": CG, "max_cudagraph_capture_size": CAP},
        seed=0,
    )
    init = time.perf_counter() - t0
    reqs = workload(llm.get_tokenizer())
    prompts = [TokensPrompt(prompt_token_ids=p) for p, _ in reqs]
    params = [SamplingParams(max_tokens=o, ignore_eos=True, temperature=0.0) for _, o in reqs]
    # warm-up: every code path once, outside the timing
    llm.generate(prompts[:16], params[:16], use_tqdm=False)
    steps.clear()
    torch.cuda.synchronize()
    t = time.perf_counter()
    res = llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    n_in = sum(len(p) for p, _ in reqs)
    n_out = sum(len(r.outputs[0].token_ids) for r in res)
    print(f"[mixed {CG} {MODEL} MBT={MBT}] {NREQ} requests, input {n_in} / output {n_out} tokens; "
          f"time {dt:.2f} s, output {n_out / dt:.0f} tok/s, total {(n_in + n_out) / dt:.0f} tok/s; "
          f"startup (incl. compile, capture) {init:.1f} s", flush=True)
    if steps:
        mixed = [s for s in steps if s[0] > MAXSEQ]
        pure = [s for s in steps if s[0] <= MAXSEQ]
        med = lambda xs: statistics.median(x[1] for x in xs) * 1e3 if xs else float("nan")
        print(f"   steps {len(steps)}: tokens > {MAXSEQ} (incl. prefill) {len(mixed)} steps, median {med(mixed):.2f} ms; "
              f"rest {len(pure)} steps, median {med(pure):.2f} ms; total {sum(s[1] for s in steps):.2f} s", flush=True)
        for lo, hi in ((1, 64), (65, 256), (257, 512), (513, 1024), (1025, 2048), (2049, 1 << 20)):
            xs = [s for s in steps if lo <= s[0] <= hi]
            if xs:
                print(f"   token {lo:5d}..{hi:<7d} {len(xs):5d} steps, median {med(xs):7.2f} ms, "
                      f"total {sum(x[1] for x in xs):6.2f} s", flush=True)


if __name__ == "__main__":
    main()
