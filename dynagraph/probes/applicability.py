#!/usr/bin/env python3
"""DynaGraph applicability on real models: how many are served, and at which step they fall back.

    python applicability.py --list ../survey/models_small.txt --out applicability.jsonl
    python applicability.py --tally applicability.jsonl        # only summarize existing results

The criterion comes from the single fallback logging point in `dynagraph.py`. Every rejection
logs a `DynaGraph fallback [tag]` line and a success logs `DynaGraph served`, so there is
no need to guess what happened here -- just collect the tags and count them.

**No timing**: this only checks "can one graph serve every shape", so it can run on a shared card.

Each model runs in its own subprocess. DynaGraph goes through CUDA graph capture, and when
something goes wrong it is a segfault rather than an exception (we hit that in the previous
round), which a try in the parent process cannot catch.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SURVEY = os.path.join(os.path.dirname(HERE), "survey")

CHILD = r'''
import json, logging, sys, os
sys.path.insert(0, %(survey)r)

import torch
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct
import models

spec = sys.argv[1]
out = {"spec": spec}

# Every fallback reason goes through dynagraph._fallback; the success line is in cudagraph_trees.
lines = []
class Grab(logging.Handler):
    def emit(self, rec):
        lines.append(rec.getMessage())
h = Grab()
for nm in ("torch._inductor.dynagraph", "torch._inductor.cudagraph_trees"):
    lg = logging.getLogger(nm); lg.setLevel(logging.INFO); lg.addHandler(h)


def vary(args, kwargs, n):
    """Replace every tensor whose batch dim equals the original batch with an n-row one.

    Tile/truncate the original tensor instead of drawing a fresh randn: HF input_ids are
    integer (randn fails with normal_kernel not implemented for Long), and even if they could
    be generated, random integers would go past the embedding's vocabulary range. Tiling
    keeps the dtype and the value range.

    Both args and kwargs have to change: if only args[0] is replaced, tensors that share the
    batch dim such as attention_mask stay at the original batch and the shapes do not match.
    """
    # The batch is the leading dim of the first tensor, positional or keyword
    # (HF models take input_ids and attention_mask as keywords).
    first = next((t for t in list(args) + list(kwargs.values())
                  if isinstance(t, torch.Tensor) and t.dim() > 0), None)
    if first is None:
        return None, None
    b = first.shape[0]

    def fix(t):
        if isinstance(t, torch.Tensor) and t.dim() > 0 and t.shape[0] == b:
            reps = -(-n // t.shape[0])
            return t.repeat(reps, *([1] * (t.dim() - 1)))[:n].contiguous()
        return t

    return tuple(fix(a) for a in args), {k: fix(v) for k, v in kwargs.items()}


def run(dynagraph):
    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    # Inductor's default config: GEMM goes through extern_kernels (cuBLAS), conv through cuDNN.
    # These now stay in the graph as child graphs (extern_child), so what this measures is how
    # much is still blocked after the child route; GEMM is no longer forced onto Triton.
    ic.triton.dynagraph_extern_child = True
    ic.triton.dynagraph_update = os.environ.get("TORCHINDUCTOR_DYNAGRAPH_UPDATE", "auto")

    n = {"v": 0}
    orig = ct.CUDAGraphTreeManager.record_function
    def spy(self, *a, **kw):
        n["v"] += 1
        return orig(self, *a, **kw)
    ct.CUDAGraphTreeManager.record_function = spy
    try:
        with models.device_ctx():
            m, a, k = models.build(spec)
        m = m.eval()
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        outs = []
        for bs in BATCHES:
            av, kv = vary(a, k, bs)
            if av is None:
                return None, None, "first argument is not a tensor with a variable batch"
            with torch.no_grad():
                # Twice: the first time cudagraph_trees sees a FunctionID it only does an eager
                # warmup, and it records for real on the second call. With a single call the
                # control group would never use cudagraph at all.
                f(*av, **kv)
                r = f(*av, **kv)
            outs.append(_flat(r))
        return n["v"], outs, None
    finally:
        ct.CUDAGraphTreeManager.record_function = orig


def _flat(r):
    if isinstance(r, torch.Tensor):
        return r.detach().float().reshape(-1).cpu()
    if isinstance(r, (list, tuple)):
        parts = [_flat(x) for x in r]
        parts = [p for p in parts if p is not None and p.numel()]
        return torch.cat(parts) if parts else None
    if isinstance(r, dict):
        return _flat(list(r.values()))
    return None


# Largest first: with the fixed layout, arena slots and the input store are sized by the first
# shape to arrive, so ascending order would hit a rebuild first; the dynamic layout (default)
# grows on demand and order no longer matters -- descending is kept only so the two layouts are comparable.
BATCHES = (8, 2, 5, 3)
try:
    torch.manual_seed(0)
    n_ctl, out_ctl, skip = run(False)
    if skip:
        out["skipped"] = skip
    else:
        torch.manual_seed(0)
        n_dg, out_dg, _ = run(True)
        out["n_record_control"] = n_ctl
        out["n_record_dynagraph"] = n_dg
        out["served"] = any(l.startswith("DynaGraph served") for l in lines)
        out["tags"] = sorted({l.split("[", 1)[1].split("]", 1)[0]
                              for l in lines if l.startswith("DynaGraph fallback [")})
        # A tag only says "which category"; a catch-all category like `exception` cannot be
        # investigated without the original message.
        out["detail"] = [l[:240] for l in lines
                         if l.startswith("DynaGraph fallback [")][:4]
        # If served, the numbers must match; otherwise it is a silent error -- far worse than a fallback.
        diffs = []
        for a, b in zip(out_ctl or [], out_dg or []):
            if a is None or b is None or a.shape != b.shape:
                diffs.append(None)
            else:
                scale = max(a.abs().max().item(), 1e-9)
                diffs.append((a - b).abs().max().item() / scale)
        out["rel_diff"] = diffs
except Exception as e:
    out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    out["tags"] = sorted({l.split("[", 1)[1].split("]", 1)[0]
                          for l in lines if l.startswith("DynaGraph fallback [")})

print("__DG__ " + json.dumps(out))
''' % {"survey": SURVEY}


def one(spec: str, timeout: int) -> dict:
    env = {**os.environ, "TORCHINDUCTOR_COMPILE_THREADS": "8"}
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, spec], cwd=HERE, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"spec": spec, "error": "timeout"}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("__DG__ "):
            return json.loads(line[len("__DG__ "):])
    tail = (p.stderr or p.stdout).strip().splitlines()
    return {"spec": spec,
            "error": f"exit code {p.returncode}: " + (tail[-1][:200] if tail else "no output")}


def tally(rows: list[dict]) -> None:
    served = [r for r in rows if r.get("served")]
    fell = [r for r in rows if "served" in r and not r["served"]]
    skipped = [r for r in rows if r.get("skipped")]
    errored = [r for r in rows if r.get("error")]

    print(f"\n{len(rows)} models: "
          f"served {len(served)}, fell back {len(fell)}, "
          f"skipped {len(skipped)}, errored {len(errored)}")

    judged = len(served) + len(fell)
    if judged:
        print(f"  of the {judged} that could be judged, served rate {100 * len(served) / judged:.0f}%")

    if fell:
        print("\nfallback reasons (one model may hit several):")
        c = Counter(t for r in fell for t in r.get("tags", []))
        for tag, n in c.most_common():
            print(f"  {n:>4}  {tag}")

    # Served but computed wrong is the one result this system cannot accept.
    bad = [r for r in served
           if any(d is None or d > 1e-3 for d in (r.get("rel_diff") or []))]
    if bad:
        print("\nFAIL: served but the numbers do not match (must investigate):")
        for r in bad:
            print(f"  {r['spec']}  rel_diff={r.get('rel_diff')}")
    elif served:
        print(f"\nall {len(served)} served models match the control group numerically.")

    # Served but still recording graphs means only part of the region is covered; not an error, but worth knowing.
    partial = [r for r in served if (r.get("n_record_dynagraph") or 0) > 0]
    if partial:
        print(f"\n{len(partial)} served but still re-recording (the graph has regions DynaGraph cannot take):")
        for r in partial[:10]:
            print(f"  {r['spec']}  recordings {r['n_record_control']} -> "
                  f"{r['n_record_dynagraph']}  tags={r.get('tags')}")

    if errored:
        print("\nerrors (first 10):")
        for r in errored[:10]:
            print(f"  {r['spec']}: {r['error'][:120]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=os.path.join(SURVEY, "models_small.txt"))
    ap.add_argument("--out", default="applicability.jsonl")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tally", help="only summarize an existing jsonl, do not run models")
    a = ap.parse_args()

    if a.tally:
        tally([json.loads(l) for l in open(a.tally) if l.strip()])
        return 0

    specs = [l.strip() for l in open(a.list) if l.strip() and not l.startswith("#")]
    if a.limit:
        specs = specs[: a.limit]
    print(f"{len(specs)} models, one subprocess each, timeout {a.timeout}s")

    rows = []
    with open(a.out, "w") as fh:
        for i, spec in enumerate(specs, 1):
            r = one(spec, a.timeout)
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            mark = ("served" if r.get("served") else
                    "skipped" if r.get("skipped") else
                    "error" if r.get("error") else "fallback")
            extra = ",".join(r.get("tags", []) or [])
            print(f"  [{i}/{len(specs)}] {spec:<42} {mark}  {extra}", flush=True)

    tally(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
