#!/usr/bin/env python3
"""How LLM training is "really written", part 2: packed into 1D + flash-attn varlen (CuTe/JIT version, flash_attn.cute), cu_seqlens on the device,
max_seqlen passed in as a host int. transformers' Llama, with a custom attention registered through AttentionInterface.

--pad 0: the total token count varies with the batch; --pad N: the total is padded to N (only the max_seqlen int and the mask contents change).
PYTHONPATH must include ${DG_DEPS:-/workspace/_deps}/fa4site.
"""
import argparse, os, sys, time, logging
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo, torch._inductor.config as ic

ap = argparse.ArgumentParser()
ap.add_argument("--pad", type=int, default=0)
ap.add_argument("--passes", type=int, default=3)
ap.add_argument("--plain", action="store_true", help="also run plain torch.compile (no cudagraphs) as a third reference")
ap.add_argument("--noise", action="store_true", help="DG off twice: the loss drift the nondeterministic backward alone gives")
ap.add_argument("--breakdown", action="store_true", help="per pass: forward call / backward / optimizer / time inside the runner")
ap.add_argument("--profile3", action="store_true", help="cProfile of runner.__call__ over the last pass only")
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--only", default="")
ap.add_argument("--attn", default="fa3_varlen", help="fa3_varlen / fa4_varlen (the varlen paths registered here) or flash_attention_4 (transformers' built-in integration)")
ap.add_argument("--dump-reuse", action="store_true", help="print the in_out_ptr / reuse / return lines in each region's wrapper")
a = ap.parse_args()

logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.INFO)
tags: list[str] = []
class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "fallback [" in m:
            tags.append(m.split("[", 1)[1].split("]", 1)[0] + (": " + m.split("]: ", 1)[1][:90] if "]: " in m else ""))
lg.addHandler(_Grab())

# transformers first, and the CuTe site dir only added to sys.path afterwards: with that dir on the
# path from the start, transformers' import goes on to torchvision, whose binary does not load
# against the in-tree torch build. Added first so its `flash_attn` (the stub package holding
# `flash_attn.cute`) wins over the container's flash_attn 2.7.4, whose binary does not load either.
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: F401
sys.path.insert(0, os.environ.get("FA4_SITE", os.environ.get("DG_DEPS", "/workspace/_deps") + "/fa4site"))
from flash_attn.cute.interface import flash_attn_varlen_func  # noqa: E402

def _thd(t):
    # (B=1, H, T, D) -> (T, H, D)
    return t.transpose(1, 2).reshape(-1, t.shape[1], t.shape[3])

def fa4_varlen(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    r = flash_attn_varlen_func(
        _thd(query), _thd(key), _thd(value),
        cu_seqlens_q=kwargs["cu_seq_lens_q"], cu_seqlens_k=kwargs["cu_seq_lens_k"],
        max_seqlen_q=kwargs["max_length_q"], max_seqlen_k=kwargs["max_length_k"],
        softmax_scale=scaling, causal=True,
    )
    out = r[0] if isinstance(r, tuple) else r
    return out.unsqueeze(0), None      # (B=1, T, H, D)

def fa3_varlen(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    # standard FA3 (hopper, the flash_attn_3 package, built ourselves)
    import flash_attn_interface as fa3
    r = fa3.flash_attn_varlen_func(
        _thd(query), _thd(key), _thd(value),
        kwargs["cu_seq_lens_q"], kwargs["cu_seq_lens_k"],
        kwargs["max_length_q"], kwargs["max_length_k"],
        softmax_scale=scaling, causal=True,
    )
    out = r[0] if isinstance(r, tuple) else r
    return out.unsqueeze(0), None

def register_attn():
    # After the model class import: `transformers.modeling_utils` imported on its own drags in
    # torchvision, whose binary does not load against the in-tree torch build.
    from transformers import LlamaForCausalLM  # noqa: F401
    import transformers.modeling_utils as mu
    if "fa4_varlen" not in mu.ALL_ATTENTION_FUNCTIONS:
        mu.AttentionInterface.register("fa4_varlen", fa4_varlen)
    if "fa3_varlen" not in mu.ALL_ATTENTION_FUNCTIONS:
        mu.AttentionInterface.register("fa3_varlen", fa3_varlen)

V = 1024
DOCS = [[40, 56, 120], [200, 16, 64, 30], [96, 96], [24, 180, 44, 8, 60], [150, 70], [32, 32, 32, 32], [88, 110, 12], [64]]

def make_model():
    torch.manual_seed(0)
    from transformers import LlamaConfig, LlamaForCausalLM
    register_attn()
    cfg = LlamaConfig(vocab_size=V, hidden_size=a.hidden, intermediate_size=a.hidden * 2,
                      num_hidden_layers=a.layers, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=2048, attn_implementation=a.attn, use_cache=False,
                      torch_dtype=torch.bfloat16)
    return LlamaForCausalLM(cfg).to(torch.bfloat16)

def batch(step):
    docs = list(DOCS[step % len(DOCS)])
    g = torch.Generator(device="cuda"); g.manual_seed(1000 + step)
    T = sum(docs)
    if a.pad:
        assert a.pad >= T, (a.pad, T)
        if a.pad > T:
            docs.append(a.pad - T)
    ids = torch.randint(0, V, (1, sum(docs)), device="cuda", generator=g)
    pos = torch.cat([torch.arange(n) for n in docs]).cuda()[None]
    labels = ids.clone()
    if a.pad and a.pad > T:
        labels[:, T:] = -100
    cu = torch.tensor([0] + list(torch.tensor(docs).cumsum(0)), dtype=torch.int32, device="cuda")
    return ids, pos, labels, cu, max(docs)

def run(dynagraph: bool, cudagraphs: bool = True):
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
        m = make_model().cuda().train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        f = torch.compile(m, dynamic=True, mode="reduce-overhead" if cudagraphs else "default")
        losses, times = [], []
        parts = {"fwd": [], "bwd": [], "opt": [], "runner": []}
        runner_t = {"v": 0.0, "n": 0}
        prof = None
        state = {"on": False}
        if (a.breakdown or a.profile3) and dynagraph:
            import cProfile
            from torch._inductor import dynagraph as dg
            _oc = dg.DynaGraphRunner.__call__
            prof = cProfile.Profile() if a.profile3 else None
            def _timed(self, inputs):
                t = time.perf_counter()
                if prof is not None and state["on"]:
                    prof.enable()
                try:
                    return _oc(self, inputs)
                finally:
                    if prof is not None and state["on"]:
                        prof.disable()
                    runner_t["v"] += time.perf_counter() - t
                    runner_t["n"] += 1
            dg.DynaGraphRunner.__call__ = _timed
        for step in range(a.passes * len(DOCS)):
            state["on"] = step >= (a.passes - 1) * len(DOCS)
            ids, pos, labels, cu, mx = batch(step)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            runner_t["v"] = 0.0; runner_t["n"] = 0
            out = f(input_ids=ids, position_ids=pos, labels=labels,
                    cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=mx, max_length_k=mx)
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
            P = len(DOCS)
            for k, v in parts.items():
                meds = [sorted(v[i:i + P])[len(v[i:i + P]) // 2] * 1e3 for i in range(0, len(v), P)]
                print(f"    {k:<7} median step per pass {' / '.join(f'{x:.1f}' for x in meds)} ms")
            if dynagraph:
                print(f"    runner calls per step {runner_t['n']}")
        if (a.breakdown or a.profile3) and dynagraph:
            dg.DynaGraphRunner.__call__ = _oc
        return n_rec["v"], losses, times, list(tags)
    finally:
        ct.CUDAGraphNode.__init__ = orig

if a.dump_reuse:
    from torch._inductor import dynagraph as dg
    _orig_init = dg.DynaGraphRunner.__init__
    def _spy(self, model, src, device, *a, **kw):
        body = dg._entry_source(src, getattr(model, "__name__", None)) or ""
        name = getattr(model, "__name__", "?")
        head = re.search(r"^\s*(.*)= args\s*$", body, re.MULTILINE)
        print(f"  ---- {name}: args {head.group(1).strip()[:150] if head else '?'}")
        for l in body.splitlines():
            t = l.strip()
            if "reuse" in t or "in_out_ptr" in t or t.startswith("return") or ("= reinterpret_tensor(" in t and "arg" in t):
                print("    |", t[:200])
        return _orig_init(self, model, src, device, *a, **kw)
    dg.DynaGraphRunner.__init__ = _spy
import re
res = {}
P = len(DOCS)
for mode in (("plain", "off", "on") if a.plain else ("off", "on")):
    if a.only and mode != a.only:
        continue
    n, losses, times, tg = run(mode == "on" and not a.noise, cudagraphs=mode != "plain")
    res[mode] = (n, losses, times, tg)
    meds = [sorted(times[i:i + P])[len(times[i:i + P]) // 2] * 1e3 for i in range(0, len(times), P)]
    print(f"  pad={a.pad} DG={mode:<3} recordings {n:<3} median step per pass {' / '.join(f'{x:.1f}' for x in meds)} ms  loss[-1] {losses[-1]:.4f}")
    if tg:
        seen = {}
        for t in tg: seen[t.split(":")[0]] = seen.get(t.split(":")[0], 0) + 1
        print(f"    tags {seen}")
        for t in list(dict.fromkeys(tg))[:6]:
            print(f"      {t}")
if "plain" in res:
    for other in ("off", "on"):
        if other in res:
            d = [abs(x - y) / max(abs(x), 1e-6) for x, y in zip(res["plain"][1], res[other][1])]
            print(f"  plain vs {other}: max per-step loss rel diff {max(d):.2e}  per step: " + " ".join(f"{x:.0e}" for x in d))
if "off" in res and "on" in res:
    lo, ln = res["off"][1], res["on"][1]
    diffs = [abs(x - y) / max(abs(x), 1e-6) for x, y in zip(lo, ln)]
    worst = max(diffs)
    print(f"  max per-step loss rel diff {worst:.2e}  recordings {res['off'][0]} -> {res['on'][0]}")
    if worst > 1e-5:
        print("  per step: " + " ".join(f"{d:.0e}" for d in diffs))
    ok = worst < 1e-2
    print("  all passed" if ok else "  FAILED")
    sys.exit(0 if ok else 1)
