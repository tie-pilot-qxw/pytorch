#!/usr/bin/env python3
"""LLM training "as really written": samples packed into a 1-D token stream, positions reset per document, attention via flex_attention (torch-native varlen).

Two modes: --pad 0 (total token count varies with the batch) and --pad 1024 (total padded to a fixed value; only the mask contents change).
Runs the same stream with DG off and on, comparing record counts, fallback tags, per-step loss, and per-pass per-step time.
"""
import argparse, os, sys, time, logging
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo, torch._inductor.config as ic

ap = argparse.ArgumentParser()
ap.add_argument("--pad", type=int, default=0, help="0 = total varies with the batch; N = pad to N tokens")
ap.add_argument("--passes", type=int, default=3)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--only", default="")
ap.add_argument("--noise", action="store_true", help="DG off twice: the drift the backward alone gives")
ap.add_argument("--plain", action="store_true", help="also run plain torch.compile (no cudagraphs) as a third reference")
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

V = 1024
# Document lengths per batch (a fixed pseudo-random stream, same on both sides)
DOCS = [[40, 56, 120], [200, 16, 64, 30], [96, 96], [24, 180, 44, 8, 60], [150, 70], [32, 32, 32, 32], [88, 110, 12], [64]]

def make_model():
    torch.manual_seed(0)
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=V, hidden_size=a.hidden, intermediate_size=a.hidden * 2,
                      num_hidden_layers=a.layers, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=2048, attn_implementation="flex_attention", use_cache=False)
    return LlamaForCausalLM(cfg)

def batch(step):
    docs = list(DOCS[step % len(DOCS)])
    g = torch.Generator(device="cuda"); g.manual_seed(1000 + step)
    T = sum(docs)
    if a.pad:
        assert a.pad >= T, (a.pad, T)
        if a.pad > T:
            docs.append(a.pad - T)   # add one padding document
    ids = torch.randint(0, V, (1, sum(docs)), device="cuda", generator=g)
    pos = torch.cat([torch.arange(n) for n in docs]).cuda()[None]
    labels = ids.clone()
    if a.pad and a.pad > T:
        labels[:, T:] = -100
    return ids, pos, labels

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
        for step in range(a.passes * len(DOCS)):
            ids, pos, labels = batch(step)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = f(input_ids=ids, position_ids=pos, labels=labels)
            out.loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
            losses.append(out.loss.item())
        return n_rec["v"], losses, times, list(tags)
    finally:
        ct.CUDAGraphNode.__init__ = orig

res = {}
P = len(DOCS)
for mode in (("plain", "off", "on") if a.plain else ("off", "on")):
    if a.only and mode != a.only:
        continue
    n, losses, times, tg = run(mode == "on" and not a.noise, cudagraphs=mode != "plain")
    res[mode] = (n, losses, times, tg)
    meds = [sorted(times[i:i + P])[len(times[i:i + P]) // 2] * 1e3 for i in range(0, len(times), P)]
    print(f"  pad={a.pad} DG={mode:<3} records {n:<3} per-pass median per step {' / '.join(f'{x:.1f}' for x in meds)} ms  loss[-1] {losses[-1]:.4f}")
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
            print(f"  plain vs {other}: max per-step loss rel diff {max(d):.2e}")
if "off" in res and "on" in res:
    lo, ln = res["off"][1], res["on"][1]
    worst = max(abs(x - y) / max(abs(x), 1e-6) for x, y in zip(lo, ln))
    print(f"  max per-step loss rel diff {worst:.2e}  records {res['off'][0]} -> {res['on'][0]}")
    ok = worst < 1e-4
    print("  all passed" if ok else "  FAILED")
    sys.exit(0 if ok else 1)
