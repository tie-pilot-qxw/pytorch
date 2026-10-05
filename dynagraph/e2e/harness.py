"""Shared part of the end-to-end workloads: the modes, counters, per-step timing.

Modes
  eager    no compile
  compile  torch.compile(dynamic=True), no cudagraph
  trees    mode="reduce-overhead", upstream cudagraph_trees (records one graph per new shape)
  dg       mode="reduce-overhead" + DynaGraph
  pad      every batch padded to the global max, dynamic=False + reduce-overhead: one static graph (the K=1 baseline)

Each workload provides make() -> (model, step_fn), step_fn(f, batch) -> loss tensor (the training step does its own
backward/optimizer). The harness runs the same batches once per mode, times every step (including the first pass's
compile/record), and reports record count / DynaGraph serve count / fallback tags / recompile count / loss difference vs eager.
"""
import logging
import os
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "16")
import torch
import torch._dynamo
import torch._inductor.config as ic

MODES = ("eager", "compile", "trees", "dg", "pad")
AMP = os.environ.get("AMP") == "1"


def amp():
    """AMP=1: forward runs under bf16 autocast (DeepGEMM only does bf16). eager is wrapped the same way as every compiled mode."""
    import contextlib

    return torch.autocast("cuda", dtype=torch.bfloat16) if AMP else contextlib.nullcontext()


if os.environ.get("GEMM") == "deepgemm":
    import dgemm

    dgemm.install()


class Counters(logging.Handler):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.tags = {}
        self.records = 0
        self.served = 0
        self.other = 0

    def emit(self, r):
        m = r.getMessage()
        if "fallback [" in m:
            t = m.split("[", 1)[1].split("]", 1)[0]
            detail = m.split("]: ", 1)[1][:120] if "]: " in m else ""
            self.tags.setdefault(t, detail)


C = Counters()
_lg = logging.getLogger("torch._inductor.dynagraph")
_lg.setLevel(logging.INFO)
_lg.addHandler(C)
_lg.propagate = False
if os.environ.get("DGLOG"):
    _fh = logging.FileHandler(os.environ["DGLOG"], mode="w")
    _fh.setFormatter(logging.Formatter("%(relativeCreated)d %(message)s"))

    class _Mem(logging.Filter):
        def filter(self, rec):
            if torch.cuda.is_initialized():
                rec.msg = f"[alloc {torch.cuda.memory_allocated() / 2**30:.1f}G reserved {torch.cuda.memory_reserved() / 2**30:.1f}G] " + str(rec.msg)
            return True

    _fh.addFilter(_Mem())
    _lg.addHandler(_fh)
_hooked = False
import collections

TIMES = collections.Counter()
COUNTS = collections.Counter()


def _hook():
    global _hooked
    if _hooked:
        return
    _hooked = True
    from torch._inductor import cudagraph_trees as ct
    from torch._inductor import dynagraph as dg

    orig = ct.CUDAGraphNode.__init__

    def rec(self, *x, **kw):
        C.records += 1
        return orig(self, *x, **kw)

    ct.CUDAGraphNode.__init__ = rec
    oc = dg.DynaGraphRunner.__call__

    def call(self, inputs):
        if os.environ.get("DGSYNC"):
            torch.cuda.synchronize()
        r = oc(self, inputs)
        if os.environ.get("DGSYNC"):
            torch.cuda.synchronize()
        if r is not None and r is not dg.SKIP_SHAPE and r is not dg.REBUILD:
            C.served += 1
        else:
            C.other += 1
        return r

    dg.DynaGraphRunner.__call__ = call
    if os.environ.get("DGTIME"):
        import functools

        TIMES.clear()
        for name in os.environ["DGTIME"].split(","):
            meth = getattr(dg.DynaGraphRunner, name)

            def timed(self, *a, _orig=meth, _name=name, **kw):
                t = time.perf_counter()
                try:
                    return _orig(self, *a, **kw)
                finally:
                    TIMES[_name] += time.perf_counter() - t
                    COUNTS[_name] += 1

            setattr(dg.DynaGraphRunner, name, functools.wraps(meth)(timed))
    if os.environ.get("DGPREP"):
        op = dg.DynaGraphRunner._prepare_inline
        seen = collections.defaultdict(set)

        def prep(self, i, decl, a_, kw, hkey=None):
            k = hkey[0] if hkey else None
            if hkey in seen[(i, k)]:
                print(f"prepare again: site {i} same hkey", flush=True)
            elif seen[(i, k)]:
                old = next(iter(seen[(i, k)]))
                print(f"prepare again: site {i} key same, hkey parts changed "
                      f"{[n for n, x, y in zip(['key', 'lane', 'ptrs', 'deps'], old, hkey) if x != y]}", flush=True)
            seen[(i, k)].add(hkey)
            return op(self, i, decl, a_, kw, hkey)

        dg.DynaGraphRunner._prepare_inline = prep
    if os.environ.get("DGKEXPR"):
        omp = dg.DynaGraphRunner._make_plan

        def mp(self, env):
            if any("sort" in n for n in self.extern_sites):
                for k in self.kernels:
                    if os.environ["DGKEXPR"] in k["name"]:
                        vals = {n: dg._eval_int(e, env) for n, e in k["exprs"].items()}
                        print(f"KEXPR keys {sorted(k)} args {k.get('args') or k.get('argnames')} syms {k.get('syms') or k.get('sym_args')} {k['name'][-40:]} env {dict(sorted(env.items()))} exprs {vals} consts {k.get('consts')} grid {k.get('grid')}", flush=True)
            return omp(self, env)

        dg.DynaGraphRunner._make_plan = mp
    if os.environ.get("DGHARV"):
        oh2 = dg.DynaGraphRunner._harvest

        def harv(self, env, key, hkey, inputs):
            if any("sort" in n for n in self.extern_sites):
                desc = [(i, tuple(x.shape), tuple(x.stride()), x.dtype) if torch.is_tensor(x) else (i, x) for i, x in enumerate(inputs)]
                print(f"HARV runner {id(self) % 1000} env {dict(sorted(env.items()))} inputs {desc}", flush=True)
            return oh2(self, env, key, hkey, inputs)

        dg.DynaGraphRunner._harvest = harv
    if os.environ.get("DGHKEY"):
        oh = dg.DynaGraphRunner._harvest
        last = {}

        def harvest(self, env, key, hkey, inputs):
            prev = last.get(id(self))
            if prev is not None:
                parts = ["key", "lane", "extern-read ptrs", "deps"]
                diff = [p for p, x, y in zip(parts, prev, hkey) if x != y]
                if "extern-read ptrs" in diff:
                    moved = [(j, hex(x), hex(y)) for j, x, y in zip(sorted(self.extern_read), prev[2], hkey[2]) if x != y]
                    diff.append(f"moved {moved[:4]} of {len(moved)}")
                print(f"harvest #{self.harvests} runner {id(self) % 1000}: changed {diff}", flush=True)
            last[id(self)] = hkey
            return oh(self, env, key, hkey, inputs)

        dg.DynaGraphRunner._harvest = harvest


def compiled(model, mode, **kw):
    torch._dynamo.reset()
    _hook()
    C.reset()
    ic.force_disable_caches = True
    if os.environ.get("TREES_HISTORY"):
        ic.triton.cudagraph_trees_history_recording = True
    ic.triton.dynagraph = mode == "dg"
    ic.triton.dynagraph_extern_child = True
    torch._dynamo.utils.counters.clear()
    # Upstream bug worked around: an unspecialized Python float attribute (SchNet's Gaussian
    # smearing coeff) becomes a 0-d input the backward compiles as a CPU tensor while the forward
    # hands it over on CUDA, and under reduce-overhead the backward's C++ kernel dereferences the
    # device pointer (segfault, trees and DynaGraph alike).
    torch._dynamo.config.specialize_float = True
    if os.environ.get("DGNOASSERT"):
        # debug: no device-side bound asserts, so a bad index shows up as a verification mismatch
        ic.assert_indirect_indexing = False
    if os.environ.get("GEMM") == "triton":
        # GEMM goes through Inductor's Triton templates (DynaGraph tier 1), not the cuBLAS extern; same config for every compiled mode
        ic.max_autotune_gemm = True
        ic.max_autotune_gemm_backends = "TRITON"
    if mode == "eager":
        return model
    if mode == "pad":
        return torch.compile(model, dynamic=False, mode="reduce-overhead", **kw)
    if mode == "compile":
        return torch.compile(model, dynamic=True, **kw)
    return torch.compile(model, dynamic=True, mode="reduce-overhead", **kw)


def run(make, batches, modes, warm=20, label="", pad=None):
    """make() -> (model, step). Every mode starts from the same initialization and runs three segments:
      warm   the first `warm` batches (compilation and first recordings all happen here)
      new    the remaining batches, all with unseen shapes -- real training resamples every epoch, so this is the steady state
      replay the `new` segment run again as is -- every shape seen, only a reference for the hit case, never happens in practice
    pad(batches) -> the batches padded to the global max, used only by pad mode; the padded part can change semantics (e.g.
    BatchNorm statistics), so pad mode's loss difference is for reference only."""
    res = {}
    padded = None
    for mode in modes:
        if mode == "pad" and pad is None:
            continue
        if mode == "pad" and padded is None:
            padded = pad(batches)
        model, step = make()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        f = compiled(model, mode)
        seq = padded if mode == "pad" else batches
        segs = {"warm": seq[:warm], "new": seq[warm:], "replay": seq[warm:]}
        per_seg, losses = {}, []
        prof = None
        for name, bs in segs.items():
            if name == os.environ.get("DGPROF", "") and mode == "dg":
                import cProfile

                prof = cProfile.Profile()
                prof.enable()
            ts = []
            if os.environ.get("NOSYNC") and name != "warm":
                # A training loop that does not wait for the GPU every step: the host work of a step
                # overlaps the GPU work of the ones before it. One sync for the whole segment.
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                # clone at once: a graph's output lives in its pool / arena and the next replay overwrites it
                got = [step(f, b).clone() for b in bs]
                torch.cuda.synchronize()
                ts = [(time.perf_counter() - t0) / max(len(bs), 1)] * len(bs)
                if name != "replay":
                    losses += [x.item() for x in got]
                per_seg[name] = ts
                if TIMES and mode == "dg":
                    n = max(len(bs), 1)
                    print(f"   [{name}] per step " + "  ".join(f"{k} {v / n * 1e3:.2f}ms/{COUNTS[k] / n:.1f} calls" for k, v in TIMES.most_common()), flush=True)
                    TIMES.clear()
                    COUNTS.clear()
                continue
            snaps = []
            for b in bs:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                loss = step(f, b)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
                if TIMES:
                    snaps.append(dict(TIMES))
                if name != "replay":
                    losses.append(loss.item())
            per_seg[name] = ts
            if TIMES and mode == "dg":
                n = max(len(bs), 1)
                print(f"   [{name}] per-step mean " + "  ".join(f"{k} {v / n * 1e3:.2f}ms/{COUNTS[k] / n:.1f} calls" for k, v in TIMES.most_common()), flush=True)
                # Per-step medians: a mean is dominated by the few steps that load or JIT a library kernel.
                med = {}
                for k in TIMES:
                    d = sorted(s1.get(k, 0.0) - s0.get(k, 0.0) for s0, s1 in zip([{}] + snaps[:-1], snaps))
                    med[k] = d[len(d) // 2]
                print(f"   [{name}] per-step median " + "  ".join(f"{k} {v * 1e3:.2f}ms" for k, v in sorted(med.items(), key=lambda x: -x[1])), flush=True)
                TIMES.clear()
                COUNTS.clear()
        if prof is not None:
            import io
            import pstats

            prof.disable()
            buf = io.StringIO()
            pstats.Stats(prof, stream=buf).sort_stats(os.environ.get("DGSORT", "tottime")).print_stats(os.environ.get("DGFILTER", ""), 30)
            if os.environ.get("DGCALLEES"):
                pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_callees(os.environ["DGCALLEES"])
            print(buf.getvalue())
        cnt = torch._dynamo.utils.counters
        res[mode] = dict(
            per_seg=per_seg,
            losses=losses,
            records=C.records,
            served=C.served,
            other=C.other,
            tags=dict(C.tags),
            frames=cnt["stats"].get("unique_graphs", 0),
            breaks=sum(cnt["graph_break"].values()),
            peak=torch.cuda.max_memory_reserved() / 2**30,
        )
        _report(label, mode, res[mode], res.get("eager"), res.get("compile"))
    return res


def _med(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def _loss_delta(r, ref):
    n = min(len(ref["losses"]), len(r["losses"]))
    return max(abs(a - b) / max(abs(b), 1e-6) for a, b in zip(r["losses"][:n], ref["losses"][:n]))


def _report(label, mode, r, ref, ref_compile=None):
    ps = r["per_seg"]
    tot = f"warm {sum(ps['warm']):.2f}s"
    med = " / ".join(f"{_med(ps[k]) * 1e3:.2f}" for k in ("new", "replay"))
    mean_new = sum(ps["new"]) / max(len(ps["new"]), 1) * 1e3
    d = ""
    if ref is not None and mode != "eager":
        d = f" loss_reldiff={_loss_delta(r, ref):.2e}"
    if ref_compile is not None and mode in ("trees", "dg"):
        d += f" (vs compile {_loss_delta(r, ref_compile):.2e})"
    print(f"[{label}] {mode:<7} {tot}  new-shape step mean {mean_new:.2f} median/replay median {med} ms  recorded {r['records']}"
          f"  DG served {r['served']}/other {r['other']}  graphs {r['frames']} break {r['breaks']}"
          f"  peak mem {r['peak']:.1f}G{d}", flush=True)
    for t, why in r["tags"].items():
        print(f"      fallback [{t}] {why}", flush=True)
