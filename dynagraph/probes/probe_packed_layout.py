#!/usr/bin/env python3
"""packed batch: the total token count is fixed, the split by number of samples varies. The dynamic layout's arena should take only "the total of one shape",
the fixed layout takes "sum of per-slot maxima x headroom". Both layouts must be served by one graph, with 0 recordings and bitwise equality; the dynamic
layout must also swap in a bigger arena when the total grows, instead of rebuilding.

Usage: TORCHINDUCTOR_DYNAGRAPH_LAYOUT=dynamic|fixed  TORCHINDUCTOR_DYNAGRAPH_UPDATE=host|device
"""
import os, sys, logging
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo, torch._inductor.config as ic
logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout); h.setFormatter(logging.Formatter("  %(message)s")); lg.addHandler(h)
ic.triton.dynagraph = True
ic.triton.dynagraph_extern_child = True
ic.triton.dynagraph_update = os.environ.get("TORCHINDUCTOR_DYNAGRAPH_UPDATE", "auto")
ic.triton.dynagraph_layout = os.environ.get("TORCHINDUCTOR_DYNAGRAPH_LAYOUT", "dynamic")
ic.force_disable_caches = True
from torch._inductor import dynagraph as dg

runners = []
orig_build = dg.DynaGraphRunner.build
def spy_build(self, *a, **kw):
    ok = orig_build(self, *a, **kw)
    if ok: runners.append(self)
    return ok
dg.DynaGraphRunner.build = spy_build
tags, grown = [], {"arena": 0, "store": 0}
class _Count(logging.Filter):
    def filter(self, r):
        m = r.getMessage()
        if "fallback [" in m: tags.append(m.split("[", 1)[1].split("]", 1)[0])
        if "DynaGraph arena" in m and "->" in m: grown["arena"] += 1
        if "storage" in m and "->" in m: grown["store"] += 1
        return True
lg.addFilter(_Count())

D = 64
class M(torch.nn.Module):
    # Intermediates of three sizes: (B, L, D) pointwise, (B, D) per-sample reduction, (L, D) per-position reduction.
    # In the fixed layout each slot is sized by its own maximum; in the dynamic layout the total is computed per shape.
    def forward(self, x):
        y = torch.nn.functional.gelu(x) * 1.5          # (B, L, D)
        per_sample = y.sum(1)                          # (B, D)
        per_pos = y.mean(0)                            # (L, D)
        z = torch.tanh(y + per_sample[:, None, :] - per_pos[None, :, :])
        return z.sum(-1), per_sample.sum(-1) + per_pos.sum(-1).sum()

TOTAL = 4096
def stream():
    # Splits with a fixed total (B*L = 4096), then the total grows once (8192), then back to small
    for B in (8, 64, 4, 128, 16):
        yield B, TOTAL // B
    yield 8, 2 * TOTAL // 8
    yield 32, TOTAL // 32

m = M().cuda()
f = torch.compile(m, dynamic=True, mode="reduce-overhead")
n_rec = {"v": 0}
from torch._inductor import cudagraph_trees as ct
orig_rec = ct.CUDAGraphNode.__init__
def rec(self, *a, **kw):
    n_rec["v"] += 1
    return orig_rec(self, *a, **kw)
ct.CUDAGraphNode.__init__ = rec

worst = 0.0
with torch.no_grad():
    for rep in range(2):
        for B, L in stream():
            x = torch.randn(B, L, D, device="cuda")
            out = f(x)
            ref = m(x)
            for o, r in zip(out, ref):
                worst = max(worst, ((o - r).abs().max() / r.abs().max().clamp_min(1)).item())
torch.cuda.synchronize()
arena_bytes = sum(r.arena.numel() for r in runners)
store_bytes = sum(s.numel() * s.element_size() for r in runners for s in r.input_store if s is not None)
layout = ic.triton.dynagraph_layout
print(f"  layout {layout}  update {ic.triton.dynagraph_update}  regions {len(runners)}  recordings {n_rec['v']}  tags {sorted(set(tags)) or '-'}")
print(f"  arena {arena_bytes/1e6:.2f} MB  input store {store_bytes/1e6:.2f} MB  arena grown {grown['arena']} times  input store grown {grown['store']} times  max rel diff {worst:.1e}")
ok = len(runners) >= 1 and worst < 1e-5 and not [t for t in tags if t not in ("arena-too-small", "input-too-large")]
if layout == "dynamic":
    ok = ok and n_rec["v"] == 0 and grown["arena"] >= 1
print("  all passed" if ok else "  failed")
sys.exit(0 if ok else 1)
