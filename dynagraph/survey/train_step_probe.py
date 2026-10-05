#!/usr/bin/env python3
"""
End-to-end training-step probe: measure how much in a **real training step** actually blocks cudagraph.

Why the measurement point has to change
---------------------------------------
`instrument.py` only hooks `Scheduler.should_partition` / `graph_partition`,
which are **compile time**. But in training the biggest "cannot use cudagraph" surface is not at compile time at all:

  - eager autograd's `AccumulateGrad` (in C++, never goes through Inductor)
  - optimizer step, gradient clipping, GradScaler's inf check
  - DDP / FSDP communication hooks

And `compiled_autograd` defaults to False (`torch/_dynamo/config.py:770`),
so **no amount of `--train` instrumentation can see them**. This script uses two other measurement points:

  1. **runtime capture count** -- hook `CUDAGraphTreeManager.record_function`
     (`cudagraph_trees.py:2830`); every real recording of a new graph passes through here.
     The compile-time partition count is "can it be captured"; this is "how many were actually captured" -- two different things.
  2. **host sync count** -- `torch.cuda.set_sync_debug_mode("warn")`
     emits a Python warning for every synchronizing op; they are collected per **section** of the training step.
     One sync means the graph is cut at that point.

Usage
-----
    python train_step_probe.py --model tv:resnet18 --steps 12
    python train_step_probe.py --model tv:resnet18 --shapes 8,16,8,32,16 --amp --clip
    python train_step_probe.py --model tvdet:ssd300_vgg16 --steps 6 --no-compile

Note: before running, check that nobody is using the card (`nvidia-smi --query-compute-apps=...`).
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import warnings
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ---------------------------------------------------------------- measurement point 1: captures
_captures: list = []


def install_capture_counter():
    """Record one entry per newly recorded graph. Returns a () -> int reader."""
    try:
        import torch._inductor.cudagraph_trees as ct
    except Exception:
        return lambda: 0
    mgr = getattr(ct, "CUDAGraphTreeManager", None)
    if mgr is None or not hasattr(mgr, "record_function"):
        return lambda: 0
    orig = mgr.record_function

    def patched(self, new_inputs, function_id, *a, **kw):
        _captures.append(function_id)
        return orig(self, new_inputs, function_id, *a, **kw)

    mgr.record_function = patched
    return lambda: len(_captures)


# ---------------------------------------------------------------- measurement point 2: syncs
class SyncCounter:
    """Count host syncs per section.

    `set_sync_debug_mode("warn")` emits one warning per synchronizing op.
    We collect them with catch_warnings(record=True) and bucket them by section.
    Note: the torch.distributed and sparse namespaces are **not covered** (the official docs say so),
    so this number is a **lower bound**.
    """

    def __init__(self):
        self.per_section: dict[str, Counter] = defaultdict(Counter)
        self.enabled = False

    def enable(self):
        import torch
        try:
            torch.cuda.set_sync_debug_mode("warn")
            self.enabled = True
        except Exception as e:
            print(f"  (sync detection unavailable: {e})")

    def disable(self):
        import torch
        with contextlib.suppress(Exception):
            torch.cuda.set_sync_debug_mode("default")

    @contextlib.contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
        for w in caught:
            msg = str(w.message)
            if "synchron" in msg.lower() or "sync" in msg.lower():
                # use the first sentence as a fingerprint so per-call tensor sizes do not blow up the keys
                self.per_section[name][msg.split(".")[0][:90]] += 1

    def total(self, name: str) -> int:
        return sum(self.per_section[name].values())


# ---------------------------------------------------------------- training step
def build_model(spec, device, train):
    import models as _models
    _models.TRAIN = train
    with _models.device_ctx():
        m, args, kwargs = _models.build(spec)
    m.train() if train else m.eval()
    return m, args, kwargs


def reduce_loss(out):
    import torch
    if isinstance(out, torch.Tensor):
        return out.float().sum() if out.is_floating_point() else None
    if isinstance(out, (list, tuple)):
        parts = [p for p in (reduce_loss(o) for o in out) if p is not None]
        return sum(parts) if parts else None
    if isinstance(out, dict):
        return reduce_loss(list(out.values()))
    for attr in ("loss", "logits", "last_hidden_state"):
        v = getattr(out, attr, None)
        if v is not None:
            r = reduce_loss(v)
            if r is not None:
                return r
    return None


def resize_batch(args, bs):
    """Replace the batch dim of the first 4-D tensor argument with bs, to produce different shapes."""
    import torch
    out = []
    for a in args:
        if isinstance(a, torch.Tensor) and a.dim() >= 2:
            shp = list(a.shape)
            shp[0] = bs
            out.append(torch.randn(*shp, device=a.device, dtype=a.dtype))
        elif isinstance(a, list) and a and isinstance(a[0], torch.Tensor):
            out.append([torch.randn_like(a[0]) for _ in range(max(1, bs))])
        else:
            out.append(a)
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tv:resnet18")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--shapes", default="",
                    help="comma-separated batch list, used cyclically; empty = always the same shape")
    ap.add_argument("--amp", action="store_true", help="use GradScaler (adds the inf-check sync)")
    ap.add_argument("--clip", action="store_true", help="do gradient clipping")
    ap.add_argument("--clip-nonfinite", action="store_true",
                    help="enable error_if_nonfinite when clipping (adds the sync back)")
    ap.add_argument("--opt", default="sgd", choices=["sgd", "adam", "adam_fused"])
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
    import torch

    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1

    read_captures = install_capture_counter()
    sc = SyncCounter()

    model, args, kwargs = build_model(a.model, "cuda", train=True)
    params = [p for p in model.parameters() if p.requires_grad]
    if a.opt == "sgd":
        opt = torch.optim.SGD(params, lr=1e-3)
    elif a.opt == "adam":
        opt = torch.optim.Adam(params, lr=1e-3)
    else:
        opt = torch.optim.Adam(params, lr=1e-3, fused=True)
    scaler = torch.amp.GradScaler("cuda") if a.amp else None

    fn = model if a.no_compile else torch.compile(
        model, dynamic=True, mode="reduce-overhead")

    shapes = [int(x) for x in a.shapes.split(",") if x.strip()] or None
    print(f"torch {torch.__version__}  model {a.model}")
    print(f"opt={a.opt} amp={a.amp} clip={a.clip} compile={not a.no_compile} "
          f"shapes={shapes or 'fixed'}\n")

    sc.enable()
    per_step = []
    for i in range(a.steps):
        cur = args if shapes is None else resize_batch(args, shapes[i % len(shapes)])
        before = read_captures()

        with sc.section("forward"):
            out = fn(*cur, **kwargs)
        with sc.section("loss"):
            loss = reduce_loss(out)
            if loss is None:
                print("no scalar output to backpropagate from"); return 1
            if scaler is not None:
                loss = scaler.scale(loss)
        with sc.section("backward"):
            loss.backward()
        if a.clip:
            with sc.section("clip"):
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    params, 1.0, error_if_nonfinite=a.clip_nonfinite)
        with sc.section("step"):
            if scaler is not None:
                scaler.step(opt)      # this contains the sum(v.item()) inf check
                scaler.update()
            else:
                opt.step()
        with sc.section("zero_grad"):
            opt.zero_grad(set_to_none=True)

        after = read_captures()
        bs = shapes[i % len(shapes)] if shapes else "-"
        per_step.append((i, bs, after - before))
        print(f"  step {i:>2}  batch={str(bs):>4}  new captures this step {after - before}"
              f"  total {after}")

    sc.disable()

    print(f"\ntotal captures: {read_captures()}   "
          f"distinct shapes: {len(set(s for _, s, _ in per_step))}")
    print("\nhost syncs per section (more means the graph is cut into more pieces; "
          "note distributed/sparse are not covered, so this is a lower bound)")
    for sec in ("forward", "loss", "backward", "clip", "step", "zero_grad"):
        n = sc.total(sec)
        flag = "  <-- blocks the graph here" if n else ""
        print(f"  {sec:<10} {n:>4}{flag}")
        for msg, c in sc.per_section[sec].most_common(3):
            print(f"      {c:>3}x  {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
