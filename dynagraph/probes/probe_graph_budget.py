"""Revisit shapes whose main graph was dropped over the graph budget.

fp32 addmm goes to cuBLAS (a tier-2 extern call), and cuBLAS picks a different kernel set (node
topology) for small and large M. Two such sites in one region need one main graph per combination
of their topologies (4 here), so with a budget of 2 graphs are dropped (LRU). Later passes revisit
shapes that were harvested but whose graph is gone; those must be captured again, not crash
(this used to raise KeyError in DynaGraphRunner.__call__).
"""
import logging
import os

os.environ.setdefault("TORCHINDUCTOR_DYNAGRAPH_MAX_GRAPHS", "2")
import torch
import torch._inductor.config as ic

msgs = []


class Grab(logging.Handler):
    def emit(self, r):
        msgs.append(r.getMessage())


lg = logging.getLogger("torch._inductor.dynagraph")
lg.setLevel(logging.INFO)
lg.addHandler(Grab())

ic.triton.dynagraph = True
ic.triton.dynagraph_extern_child = True
torch.backends.cuda.matmul.allow_tf32 = False

K, N = 4096, 1024
w = torch.randn(K, N, device="cuda")
b = torch.randn(N, device="cuda")
w2 = torch.randn(K, N, device="cuda")


def f(x, y):
    # Two cuBLAS sites whose M comes from different inputs: their topologies combine, so the
    # number of main graphs (one per combination) can exceed the budget while each site stays
    # within its own.
    return torch.addmm(b, x, w).relu().sum(0) + torch.addmm(b, y, w2).relu().sum(0)


cf = torch.compile(f, dynamic=True, mode="reduce-overhead")
Ms = [(8, 8), (4096, 4096), (8, 4096), (4096, 8)] * 3
worst = 0.0
for i, (m1, m2) in enumerate(Ms):
    torch.compiler.cudagraph_mark_step_begin()
    x = torch.randn(m1, K, device="cuda")
    y = torch.randn(m2, K, device="cuda")
    got = cf(x, y).clone()
    ref = f(x, y)
    worst = max(worst, ((got - ref).abs() / ref.abs().clamp_min(1)).max().item())
torch.cuda.synchronize()
drops = sum("dropped graph" in m for m in msgs)
fallbacks = [m for m in msgs if "fallback [" in m]
print(f"calls {len(Ms)}  dropped graphs {drops}  fallbacks {len(fallbacks)}  max rel diff vs eager {worst:.2e}")
for m in fallbacks[:5]:
    print("  ", m[:160])
ok = worst < 1e-4 and drops > 0 and not fallbacks
print("  all passed" if ok else "  FAILED (expected graph drops, no fallbacks, and results matching eager)")
raise SystemExit(0 if ok else 1)
