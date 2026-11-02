#!/usr/bin/env python3
"""Interleaved cuBLAS-bucket latency experiment; writes raw samples and profiles.

Run with the built fork, on an idle GPU:
  CUDA_VISIBLE_DEVICES=3 python dynagraph/microbench/cublas_buckets.py \
    --workload mlp --output /tmp/cublas_buckets_mlp.json
  # --workload bert requires transformers; no pretrained weights are downloaded.
"""

import argparse
import contextlib
import cProfile
import io
import json
import logging
import math
import os
from pathlib import Path
import pstats
import random
import statistics
import subprocess
import time
from unittest.mock import patch

import torch
from torch import nn
from torch._inductor import config as ic, dynagraph as dg
from torch._inductor.compile_fx import compile_fx


class MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.up = nn.Linear(768, 3072)
        self.down = nn.Linear(3072, 768)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm(x))))


class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        # Only the text encoder is used. Avoid loading optional multimedia
        # extensions built against a different torch in research containers.
        import transformers.utils as utils
        import transformers.utils.import_utils as imports

        for name in (
            "is_torchvision_available",
            "is_torchaudio_available",
            "is_librosa_available",
        ):
            setattr(utils, name, lambda: False)
            setattr(imports, name, lambda: False)
        from transformers import BertConfig, BertModel

        self.bert = BertModel(
            BertConfig(attn_implementation="sdpa"), add_pooling_layer=False
        )

    def forward(self, ids, mask):
        return self.bert(input_ids=ids, attention_mask=mask, return_dict=False)[0]


class State:
    def __init__(self, name, model):
        self.name = name
        self.width = int(name[6:]) if name.startswith("bucket") else 0
        self.is_dg = name.startswith("dg") or bool(self.width)
        self.settings = {
            "triton.cudagraphs": name not in ("compile", "eager"),
            "triton.dynagraph": self.is_dg,
            "triton.dynagraph_extern_child": True,
            "triton.dynagraph_cublas_bucket_size": self.width,
            "triton.dynagraph_clone_sites": name == "dg-clone",
            "triton.dynagraph_update": "host",
            "triton.dynagraph_verify_shapes": 0,
            "triton.autotune_pointwise": False,
            "compile_threads": 1,
            "force_disable_caches": True,
        }
        self.runners = []
        self.compiles = 0
        self.counts = {
            "captures": 0,
            "parent_captures": 0,
            "cublas_captures": 0,
            "clones": 0,
            "harvests": 0,
            "served": 0,
            "upstream": 0,
        }
        self.harvest_us = 0.0
        self.fallbacks = []
        self.records = {"new": [], "replay": [], "gpu": []}

        def backend(gm, inputs):
            self.compiles += 1
            with ic.patch(self.settings):
                return compile_fx(gm, inputs)

        self.fn = (
            model
            if name == "eager"
            else torch.compile(model, backend=backend, dynamic=True, fullgraph=True)
        )

    def counters(self):
        out = dict(self.counts)
        out["bucket_captures"] = sum(
            r.cublas_bucket_stats["captures"] for r in self.runners
        )
        out["bucket_clones"] = sum(
            r.cublas_bucket_stats["clones"] for r in self.runners
        )
        return out


def describe(values):
    values = sorted(values)
    return {
        "median_us": statistics.median(values),
        "mean_us": statistics.mean(values),
        "p10_us": values[int((len(values) - 1) * 0.1)],
        "p90_us": values[int((len(values) - 1) * 0.9)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=("mlp", "bert"), default="mlp")
    ap.add_argument(
        "--modes",
        default="compile,dg,dg-clone,bucket32,bucket64,bucket128,bucket256,trees-pow2",
    )
    ap.add_argument("--shapes", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.manual_seed(123)
    torch.set_num_threads(1)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.reset()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = (
        nn.Sequential(*(MLPBlock() for _ in range(12)))
        if args.workload == "mlp"
        else Bert()
    )
    model = model.cuda().to(torch.bfloat16).eval()
    modes = args.modes.split(",")
    states = [State(name, model) for name in modes]
    rng = random.Random(19)
    sizes = rng.sample(range(33, 512), args.shapes)
    profile_sizes = [m for m in range(101, 120) if m not in sizes][:3]
    active, harvesting, site = [None], [False], [None]
    original_init, original_call = (
        dg.DynaGraphRunner.__init__,
        dg.DynaGraphRunner.__call__,
    )
    original_harvest, original_clone = dg.DynaGraphRunner._harvest, dg._clone_child
    original_intercepted = dg.DynaGraphRunner._run_intercepted
    original_capture = torch.cuda.CUDAGraph.capture_begin
    original_fallback = dg._fallback

    def init(r, *a, **kw):
        original_init(r, *a, **kw)
        active[0].runners.append(r)

    def call(r, inputs):
        result = original_call(r, inputs)
        key = (
            "upstream"
            if result is None or result is dg.SKIP_SHAPE or result is dg.REBUILD
            else "served"
        )
        active[0].counts[key] += 1
        return result

    def harvest(r, *a, **kw):
        st = active[0]
        st.counts["harvests"] += 1
        start = time.perf_counter()
        harvesting[0] = True
        try:
            return original_harvest(r, *a, **kw)
        finally:
            harvesting[0] = False
            st.harvest_us += (time.perf_counter() - start) * 1e6

    def intercepted(r, inputs, views, on_extern):
        def at_site(i, fn, a, kw):
            previous = site[0]
            site[0] = r.extern_sites[i]
            try:
                return on_extern(i, fn, a, kw)
            finally:
                site[0] = previous

        return original_intercepted(r, inputs, views, at_site)

    def capture(g, *a, **kw):
        st = active[0]
        if st is not None and any(r._capturing for r in st.runners):
            st.counts["parent_captures"] += 1
        elif harvesting[0]:
            st.counts["captures"] += 1
            if site[0] in (
                "mm",
                "addmm",
                "dynagraph_bucket_mm",
                "dynagraph_bucket_addmm",
            ):
                st.counts["cublas_captures"] += 1
        return original_capture(g, *a, **kw)

    def clone(*a, **kw):
        out = original_clone(*a, **kw)
        if out is not None:
            active[0].counts["clones"] += 1
        return out

    def fallback(*a, **kw):
        if active[0] is not None:
            active[0].fallbacks.append(a)
        return original_fallback(*a, **kw)

    @contextlib.contextmanager
    def use(st):
        active[0] = st
        with ic.patch(st.settings):
            yield

    def batch(m):
        if args.workload == "mlp":
            x = torch.randn(m, 768, device="cuda", dtype=torch.bfloat16)
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_static(x, 1)
            return (x,)
        ids = torch.randint(0, 30000, (1, m), device="cuda")
        mask = torch.ones_like(ids)
        torch._dynamo.mark_dynamic(ids, 1)
        torch._dynamo.mark_dynamic(mask, 1)
        torch._dynamo.mark_static(ids, 0)
        torch._dynamo.mark_static(mask, 0)
        return ids, mask

    def padded(b):
        dim = 0 if args.workload == "mlp" else 1
        m = b[0].shape[dim]
        top = 1 << (m - 1).bit_length()
        return tuple(
            torch.nn.functional.pad(x, (0, 0, 0, top - m) if dim == 0 else (0, top - m))
            for x in b
        )

    batches = {m: batch(m) for m in [512, *sizes, *profile_sizes]}
    padded_batches = {m: padded(b) for m, b in batches.items()}
    gpu_sizes = sizes[:8]
    worst_l2 = {st.name: 0.0 for st in states}

    def invoke(st, m):
        torch.compiler.cudagraph_mark_step_begin()
        return st.fn(*(padded_batches[m] if st.name == "trees-pow2" else batches[m]))

    def check(st, m, out, ref):
        if st.name == "trees-pow2":
            out = out[:m] if args.workload == "mlp" else out[:, :m]
        error = (
            (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-8)
        ).item()
        worst_l2[st.name] = max(worst_l2[st.name], error)
        if not math.isfinite(error) or error > 0.03:
            raise AssertionError(f"{st.name} M={m}: relative L2 {error}")

    with (
        torch.inference_mode(),
        patch.object(dg.DynaGraphRunner, "__init__", init),
        patch.object(dg.DynaGraphRunner, "__call__", call),
        patch.object(dg.DynaGraphRunner, "_harvest", harvest),
        patch.object(dg.DynaGraphRunner, "_run_intercepted", intercepted),
        patch.object(dg, "_clone_child", clone),
        patch.object(dg, "_fallback", fallback),
        patch.object(torch.cuda.CUDAGraph, "capture_begin", capture),
    ):
        for st in states:
            start = time.perf_counter()
            with use(st):
                for _ in range(3):
                    invoke(st, 512)
                torch.cuda.synchronize()
            print(
                f"warm {st.name}: {time.perf_counter()-start:.2f}s, compiles={st.compiles}, runners={len(st.runners)}, fallbacks={st.fallbacks}",
                flush=True,
            )
            if st.is_dg and (not st.runners or st.counts["upstream"] or st.fallbacks):
                raise AssertionError(f"{st.name} did not run entirely through DG")
            st.warm_compiles = st.compiles
            st.warm_counters = st.counters()
            st.warm_harvest_us = st.harvest_us

        for step, m in enumerate(sizes):
            ref = model(*batches[m])
            order = states[step % len(states) :] + states[: step % len(states)]
            if step % 2:
                order = order[::-1]
            for st in order:
                with use(st):
                    torch.cuda.synchronize()
                    before = st.counters()
                    cpu_start = time.process_time_ns()
                    start = time.perf_counter()
                    out = invoke(st, m)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1e6
                    cpu_us = (time.process_time_ns() - cpu_start) / 1000
                    after = st.counters()
                    st.records["new"].append(
                        {
                            "m": m,
                            "wall_us": elapsed,
                            "cpu_us": cpu_us,
                            "captures": after["captures"] - before["captures"],
                            "clones": after["clones"] - before["clones"],
                        }
                    )
                    check(st, m, out, ref)
            if (step + 1) % 10 == 0:
                print(f"new shapes: {step+1}/{len(sizes)}", flush=True)
        for st in states:
            st.new_counters = {
                k: v - st.warm_counters[k] for k, v in st.counters().items()
            }
            st.new_harvest_us = st.harvest_us - st.warm_harvest_us
            st.new_compiles = st.compiles - st.warm_compiles

        # Cudagraph trees records on the second visit. Finish those recordings
        # outside replay timing so every mode measures a genuinely warm hit.
        for m in sizes:
            for st in states:
                with use(st):
                    for _ in range(2):
                        invoke(st, m)
        torch.cuda.synchronize()

        for repeat in range(args.repeats):
            for step, m in enumerate(sizes):
                order = (
                    states[(step + repeat) % len(states) :]
                    + states[: (step + repeat) % len(states)]
                )
                if (step + repeat) % 2:
                    order = order[::-1]
                for st in order:
                    with use(st):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        invoke(st, m)
                        torch.cuda.synchronize()
                        st.records["replay"].append(
                            {
                                "m": m,
                                "round": repeat,
                                "wall_us": (time.perf_counter() - start) * 1e6,
                            }
                        )
            print(f"replay round {repeat+1}/{args.repeats}", flush=True)

        # Pure GPU execution of DG regions, with parameters already installed
        # for each shape. Excludes host patching and input staging. All regions
        # must be served, and there are no eager operations in these workloads.
        for m in gpu_sizes:
            for st in states:
                if not st.is_dg:
                    continue
                with use(st):
                    invoke(st, m)
                    torch.cuda.synchronize()
                    stream = torch.cuda.current_stream().cuda_stream
                    cu = dg._cuda()
                    events = []
                    for _ in range(5):
                        begin, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(20):
                            for r in st.runners:
                                rc = cu.cuGraphLaunch(r.ex.exec_h, stream)
                                if rc:
                                    raise RuntimeError(f"cuGraphLaunch: {rc}")
                        end.record()
                        end.synchronize()
                        events.append(begin.elapsed_time(end) * 1000 / 20)
                    st.records["gpu"].append(
                        {"m": m, "gpu_us": statistics.median(events)}
                    )

        for st in states:
            profile = cProfile.Profile()
            with use(st):
                profile.enable()
                for m in profile_sizes:
                    invoke(st, m)
                    torch.cuda.synchronize()
                profile.disable()
            text = io.StringIO()
            pstats.Stats(profile, stream=text).strip_dirs().sort_stats(
                "tottime"
            ).print_stats(10)
            output.with_name(output.stem + "_" + st.name + "_profile.txt").write_text(
                text.getvalue()
            )
            print(f"profile {st.name} written", flush=True)

    result = {
        "workload": args.workload,
        "shapes": sizes,
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "repeats": args.repeats,
        "verify_shapes": 0,
        "gpu_time_scope": "DG graph execution only; excludes host updates and input staging",
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,power.limit,clocks.sm",
                "--format=csv",
            ],
            text=True,
        ),
        "modes": {},
    }
    for st in states:
        summary = {
            "new": describe([x["wall_us"] for x in st.records["new"]]),
            "replay": describe([x["wall_us"] for x in st.records["replay"]]),
            "new_counts": st.new_counters,
            "new_harvest_us": st.new_harvest_us,
            "compiles": st.compiles,
            "new_compiles": st.new_compiles,
            "runners": len(st.runners),
            "fallbacks": st.fallbacks,
            "extern_sites": [r.extern_sites for r in st.runners],
            "worst_relative_l2": worst_l2[st.name],
        }
        if st.records["gpu"]:
            summary["gpu"] = describe([x["gpu_us"] for x in st.records["gpu"]])
        result["modes"][st.name] = {"summary": summary, "samples": st.records}
        print(st.name, json.dumps(summary), flush=True)
    output.write_text(json.dumps(result, indent=2))
    print("wrote", output, flush=True)


if __name__ == "__main__":
    main()
