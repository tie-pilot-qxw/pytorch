#!/usr/bin/env python3
"""A real variable-length training step: transformers Llama (small config), forward + backward + AdamW, seqlen changes every step.

Runs the same shape stream (same seed, same data) with DynaGraph off and on, and compares: record counts, fallback tags, whether per-step losses match, per-step time.
Usage: python probe_train_step.py [--model llama|gpt2] [--steps N] [--time]
"""
import argparse, os, sys, time, logging
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo, torch._inductor.config as ic

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama", choices=("llama", "gpt2"))
ap.add_argument("--steps", type=int, default=0, help="0 = run the shape stream twice")
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--time", action="store_true", help="also run a timing stream (needs an exclusive card)")
ap.add_argument("--only", default="", help="off / on: run only one side")
ap.add_argument("--noise", action="store_true", help="run twice with DG off to measure the baseline loss difference")
ap.add_argument("--dump", default="", help="print the wrapper lines that mention this name (e.g. buf15)")
ap.add_argument("--breakdown", action="store_true", help="per pass, report time in the forward call / backward / optimizer / inside the runner separately")
ap.add_argument("--profile3", action="store_true", help="cProfile runner.__call__ in the third pass only")
a = ap.parse_args()

logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.INFO)
tags: list[str] = []
class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "fallback [" in m:
            tags.append(m.split("[", 1)[1].split("]", 1)[0] + (": " + m.split("]: ", 1)[1][:80] if "]: " in m else ""))
lg.addHandler(_Grab())

SHAPES = [64, 128, 96, 256, 32, 160, 128, 224]
V = 1024

def make_model(kind):
    torch.manual_seed(0)
    if kind == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(vocab_size=V, hidden_size=a.hidden, intermediate_size=a.hidden * 2,
                          num_hidden_layers=a.layers, num_attention_heads=4, num_key_value_heads=4,
                          max_position_embeddings=512, attn_implementation="sdpa", use_cache=False)
        return LlamaForCausalLM(cfg)
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=V, n_embd=a.hidden, n_layer=a.layers, n_head=4, n_positions=512,
                     attn_implementation="sdpa", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False)
    return GPT2LMHeadModel(cfg)

def batch(L, step):
    g = torch.Generator(device="cuda"); g.manual_seed(1000 + step)
    return torch.randint(0, V, (a.batch, L), device="cuda", generator=g)

def run(dynagraph: bool, stream, timed=False):
    torch._dynamo.reset()
    ic.triton.dynagraph = dynagraph
    ic.triton.dynagraph_extern_child = True
    ic.force_disable_caches = True
    tags.clear()
    from torch._inductor import cudagraph_trees as ct
    n_rec = {"v": 0}
    orig = ct.CUDAGraphNode.__init__
    def rec(self, *x, **kw):
        n_rec["v"] += 1
        return orig(self, *x, **kw)
    ct.CUDAGraphNode.__init__ = rec
    try:
        m = make_model(a.model).cuda().train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        losses, times = [], []
        parts = {"fwd": [], "bwd": [], "opt": [], "runner": []}
        runner_t = {"v": 0.0}
        if a.breakdown:
            from torch._inductor import dynagraph as dg
            _oc = dg.DynaGraphRunner.__call__
            def _timed(self, inputs):
                t = time.perf_counter()
                try:
                    return _oc(self, inputs)
                finally:
                    runner_t["v"] += time.perf_counter() - t
            dg.DynaGraphRunner.__call__ = _timed
        prof = None
        if a.profile3 and dynagraph:
            import cProfile
            from torch._inductor import dynagraph as dg
            prof = cProfile.Profile()
            _oc2 = dg.DynaGraphRunner.__call__
            state = {"on": False}
            def _prof_call(self, inputs):
                if not state["on"]:
                    return _oc2(self, inputs)
                prof.enable()
                try:
                    return _oc2(self, inputs)
                finally:
                    prof.disable()
            dg.DynaGraphRunner.__call__ = _prof_call
        for step, L in enumerate(stream):
            if prof is not None:
                state["on"] = step >= 2 * len(SHAPES)
            ids = batch(L, step)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            runner_t["v"] = 0.0
            out = f(input_ids=ids, labels=ids)
            if a.breakdown:
                torch.cuda.synchronize(); t1 = time.perf_counter()
            out.loss.backward()
            if a.breakdown:
                torch.cuda.synchronize(); t2 = time.perf_counter()
            opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); t3 = time.perf_counter(); times.append(t3 - t0)
            if a.breakdown:
                parts["fwd"].append(t1 - t0); parts["bwd"].append(t2 - t1); parts["opt"].append(t3 - t2)
                parts["runner"].append(runner_t["v"])
            losses.append(out.loss.item())
        if prof is not None:
            import pstats, io
            buf = io.StringIO()
            pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(28)
            print(buf.getvalue())
        if a.breakdown:
            P = len(SHAPES)
            for k, v in parts.items():
                meds = [sorted(v[i:i + P])[len(v[i:i + P]) // 2] * 1e3 for i in range(0, len(v), P)]
                print(f"    {k:<7} per-pass median per step {' / '.join(f'{x:.1f}' for x in meds)} ms")
            if a.breakdown and dynagraph:
                dg.DynaGraphRunner.__call__ = _oc
        return n_rec["v"], losses, times, list(tags)
    finally:
        ct.CUDAGraphNode.__init__ = orig

stream = SHAPES * 2 if a.steps == 0 else [SHAPES[i % len(SHAPES)] for i in range(a.steps)]
if a.dump:
    from torch._inductor import dynagraph as dg
    _orig_init = dg.DynaGraphRunner.__init__
    def _spy(self, model, src, device, *a, **kw):
        body = dg._entry_source(src, getattr(model, "__name__", None)) or ""
        if a.dump in body:
            print(f"  ---- {getattr(model, '__name__', '?')}: lines mentioning {a.dump}")
            for l in body.splitlines():
                if a.dump in l:
                    print("    |", l.strip()[:260])
        return _orig_init(self, model, src, device, *a, **kw)
    dg.DynaGraphRunner.__init__ = _spy
if a.noise:
    l1 = run(False, stream)[1]; l2 = run(False, stream)[1]
    print(f"  DG off twice: max per-step loss rel diff {max(abs(x - y) / max(abs(x), 1e-6) for x, y in zip(l1, l2)):.2e}")
    sys.exit(0)
res = {}
for mode in ("off", "on"):
    if a.only and mode != a.only:
        continue
    n, losses, times, tg = run(mode == "on", stream)
    res[mode] = (n, losses, times, tg)
    P = len(SHAPES)
    meds = [sorted(times[i:i + P])[len(times[i:i + P]) // 2] * 1e3 for i in range(0, len(times), P)]
    print(f"  DG={mode:<3} records {n:<3} steps {len(stream)}  per-pass median per step {' / '.join(f'{x:.1f}' for x in meds)} ms  loss[0] {losses[0]:.4f} loss[-1] {losses[-1]:.4f}")
    if tg:
        seen = {}
        for t in tg: seen[t.split(":")[0]] = seen.get(t.split(":")[0], 0) + 1
        print(f"    tags {seen}")
        for t in list(dict.fromkeys(tg))[:8]:
            print(f"      {t}")
if "off" in res and "on" in res:
    lo, ln = res["off"][1], res["on"][1]
    worst = max(abs(x - y) / max(abs(x), 1e-6) for x, y in zip(lo, ln))
    print(f"  max per-step loss rel diff {worst:.2e}  records {res['off'][0]} -> {res['on'][0]}")
    ok = worst < 1e-3 and res["on"][0] == 0
    print("  all passed" if ok else "  FAILED")
    sys.exit(0 if ok else 1)
