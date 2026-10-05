#!/usr/bin/env python3
"""
P0 coverage survey harness.

Each model is compiled once in its **own subprocess**, and Inductor's cudagraph partition decisions are recorded as one JSON line.
Why subprocess isolation: Dynamo/Inductor crashing on messy models is the norm, and one crash must not take down the whole batch;
also, compile caches and global config cannot contaminate each other.

Usage:
    python runner.py --list models.txt --out results.jsonl [--timeout 600] [--jobs 4]
    python runner.py --one timm:resnet50            # debug a single model
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- child side
class _CodegenDone(Exception):
    """Sentinel exception for stopping right after codegen; not an error."""


def _stop_after_codegen():
    """
    The stats only need scheduling/codegen to run; whether the cubin gets built does not matter.
    Patch GraphLowering.compile_to_module to raise a sentinel after codegen; measured 4.1s -> 2.4s per model.
    """
    import torch._inductor.graph as G

    def patched(self, *a, **kw):
        self.codegen()
        raise _CodegenDone

    G.GraphLowering.compile_to_module = patched


def _reduce_for_backward(out):
    """Reduce an output of any structure to a scalar, used to trigger compilation of the backward graph."""
    import torch
    if isinstance(out, torch.Tensor):
        return out.float().sum() if out.is_floating_point() else None
    if isinstance(out, (list, tuple)):
        parts = [p for p in (_reduce_for_backward(o) for o in out) if p is not None]
        return sum(parts) if parts else None
    if isinstance(out, dict):
        return _reduce_for_backward(list(out.values()))
    # HuggingFace ModelOutput and the like
    for attr in ("loss", "logits", "last_hidden_state"):
        v = getattr(out, attr, None)
        if v is not None:
            r = _reduce_for_backward(v)
            if r is not None:
                return r
    return None


def run_one(spec: str, fake: bool = False, fast: bool = False,
            capture: bool = True, train: bool = False) -> dict:
    """Compile one model in the current process and return the record. Only called in the child process."""
    rec = {"model": spec, "ok": False, "ran": False, "error": None,
           "error_type": None, "fake": fake, "stop_after_codegen": fast,
           "capture_unbacked": capture, "train": train}
    t0 = time.time()
    try:
        import torch
        import torch._inductor.config as ic
        import torch._dynamo.config as dc

        # Four required settings, each one affects whether the stats mean anything; see docs/notes/FEASIBILITY.md
        ic.force_disable_caches = True            # otherwise the cache hits and the scheduler never runs
        # Both configs must be run; they answer different questions:
        #   capture=False (PyTorch default): data dependence causes a graph break at the Dynamo layer; Inductor never sees it.
        #                                    Measures "how much the first layer blocks".
        #   capture=True:                    data dependence reaches Inductor, so partition reasons can be measured,
        #                                    but some models fail to compile outright with GuardOnDataDependentSymNode.
        dc.capture_dynamic_output_shape_ops = capture
        dc.capture_scalar_outputs = capture

        from instrument import install, reset            # noqa: E402
        from models import build                         # noqa: E402
        install()
        reset()
        if fast:
            _stop_after_codegen()

        import models as _models

        def _to(obj, dev):
            if isinstance(obj, torch.Tensor):
                return obj.to(dev)
            if isinstance(obj, (list, tuple)):
                return type(obj)(_to(o, dev) for o in obj)
            if isinstance(obj, dict):
                return {kk: _to(v, dev) for kk, v in obj.items()}
            return obj

        def _run(model, args, kwargs):
            # only mode="reduce-overhead" enables cudagraph, which is what gives should_partition its meaning
            compiled = torch.compile(model, dynamic=True, mode="reduce-overhead")
            out = compiled(*args, **kwargs)
            if train:
                # Without an explicit backward, whether the backward graph is compiled depends on AOTAutograd's laziness,
                # and the stats would only cover inference. The paper reports inference and training separately.
                loss = _reduce_for_backward(out)
                if loss is None:
                    raise RuntimeError("no scalar output to backpropagate from")
                loss.backward()

        if fake:
            # The model must be built on **meta**. Building inside FakeTensorMode makes weight init
            # (trunc_normal_ etc. in torch/nn/init.py) produce unbacked symbols, after which
            # ViT / Swin / ConvNeXt / BEiT style models all blow up on
            # "GuardOnDataDependentSymNode: Eq(u0, 1)" -- which has nothing to do with the model itself
            # and is purely self-inflicted. Init on meta is a no-op, so it is clean.
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            with torch.device("meta"):
                model, args, kwargs = build(spec)
            # main has already fixed fake mode's cross-mode problem:
            #   torch/_guards.py detect_fake_mode authoritatively returns the TracingContext's mode
            #   torch/_functorch/_aot_autograd/frontend_utils.py process_inputs re-fakifies
            fm = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
            with fm:
                model = model.to_empty(device="cuda")
                _run(model, _to(args, "cuda"), _to(kwargs, "cuda"))
        else:
            with _models.device_ctx():
                model, args, kwargs = build(spec)
            if train:
                model.train()
            _run(model, args, kwargs)
        rec["ok"] = rec["ran"] = True
    except _CodegenDone:
        rec["ok"] = True                                  # sentinel, not a failure
    except Exception as e:                                # noqa: BLE001
        rec["error_type"] = type(e).__name__
        rec["error"] = str(e)[:2000]
        rec["traceback"] = traceback.format_exc()[-4000:]
    finally:
        try:
            from instrument import snapshot
            snap = snapshot()
            rec.update(snap)
            # Two criteria:
            #   real tensors: ran=True is required for the data to count as complete.
            #   fake: compilation often crashes midway (cudagraph_trees record phase); as long as graph_partition
            #         ran, record it, but **mark it as possibly truncated**. Measured: ssd300_vgg16 under fake
            #         only counts 1 subgraph ([2]), with real tensors it is 9 ([2,2,2,1,1,0,1,1,1]),
            #         cpu_ops 91 vs 98, and unbacked_binding is missed too.
            #         So fake numbers are only a lower bound, not a conclusion.
            if not rec["ok"] and snap.get("n_partitions_observed"):
                rec["ok"] = True
                rec["compiled_but_not_run"] = True
                rec["data_maybe_truncated"] = bool(fake)
        except Exception as e:                            # noqa: BLE001
            rec["snapshot_error"] = f"{type(e).__name__}: {e}"
        rec["wall_s"] = round(time.time() - t0, 2)
    return rec


# ---------------------------------------------------------------- parent side
def spawn(spec: str, timeout: int, child_flags: list) -> dict:
    cmd = [sys.executable, os.path.join(HERE, "runner.py"), "--child", spec] + child_flags
    env = dict(os.environ)
    # Must not be 1. Setting 1 turns Inductor's kernel compilation from parallel to serial, and kernel-heavy models
    # (MaxViT / Swin / dm_nfnet_f0 / mobilevit_s) hit the 900 s timeout.
    # The machine has 256 cores, others use ~190, the rest is enough for jobs x 8 compile threads.
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS",
                   os.environ.get("DYNAGRAPH_COMPILE_THREADS", "8"))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=HERE)
    except subprocess.TimeoutExpired:
        return {"model": spec, "ok": False, "error_type": "Timeout",
                "error": f"exceeded {timeout}s", "wall_s": timeout}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("__RESULT__ "):
            return json.loads(line[len("__RESULT__ "):])
    return {"model": spec, "ok": False, "error_type": "NoResult",
            "error": (p.stderr or p.stdout)[-2000:], "returncode": p.returncode}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", help="file with one model spec per line")
    ap.add_argument("--one", help="run just one model spec")
    ap.add_argument("--child", help="internal: run this spec in this process")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--fake", action="store_true",
                    help="fake tensors throughout, no GPU memory. **Only for scouting**: compilation often crashes after the first subgraph, "
                         "so multi-subgraph models are systematically undercounted; see README")
    ap.add_argument("--train", action="store_true",
                    help="training: explicit backward, so the backward graph gets compiled too")
    ap.add_argument("--no-capture", action="store_true",
                    help="use the PyTorch default config (no data-dependent capture) to measure first-layer Dynamo graph breaks")
    ap.add_argument("--fast", action="store_true",
                    help="stop after codegen, skip cubin compilation. Note: loses layer-3 data "
                         "(whole graph giving up cudagraph), which happens after compile_to_module")
    a = ap.parse_args()

    if a.child:
        print("__RESULT__ " + json.dumps(
            run_one(a.child, a.fake, a.fast, not a.no_capture, a.train),
            ensure_ascii=False))
        return 0

    specs = [a.one] if a.one else [
        ln.strip() for ln in open(a.list) if ln.strip() and not ln.startswith("#")
    ]
    print(f"{len(specs)} models, jobs {a.jobs}, per-model timeout {a.timeout}s, "
          f"fake={a.fake} fast={a.fast} capture={not a.no_capture} "
          f"train={a.train} -> {a.out}")

    done = 0
    with open(a.out, "w") as fh, ThreadPoolExecutor(max_workers=a.jobs) as ex:
        child_flags = ((["--fake"] if a.fake else [])
                       + (["--fast"] if a.fast else [])
                       + (["--no-capture"] if a.no_capture else [])
                       + (["--train"] if a.train else []))
        futs = {ex.submit(spawn, s, a.timeout, child_flags): s for s in specs}
        for fut in as_completed(futs):
            rec = fut.result()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done += 1
            flag = "ok " if rec.get("ok") else "FAIL"
            extra = "" if rec.get("ok") else f"  {rec.get('error_type')}"
            print(f"[{done}/{len(specs)}] {flag} {rec['model']}"
                  f"  partitions={rec.get('n_partitions_max','?')}"
                  f"  {rec.get('wall_s','?')}s{extra}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
