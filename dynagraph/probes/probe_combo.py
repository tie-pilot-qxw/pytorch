#!/usr/bin/env python3
"""Regions with a combo kernel (horizontal fusion, config.combo_kernels): the grid is the sum of the sub-kernels' block counts and changes with the shape.

Three independent pointwise ops are fused into one kernel under combo_kernels=True, with xnumel_0/1/2 passed as runtime arguments,
grid_type = SequentialComboKernelGrid. This checks that both the host and device update paths can recompute this kind of summed grid per shape.
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
ic.force_disable_caches = True
ic.combo_kernels = True
ic.benchmark_combo_kernel = False
from torch._inductor import dynagraph as dg
seen = {"combo": 0, "fallback": 0}
class _Count(logging.Filter):
    def filter(self, r):
        if "fallback [" in r.getMessage():
            seen["fallback"] += 1
        return True
lg.addFilter(_Count())
orig = dg.DynaGraphRunner.__init__
def spy(self, model, src, device, *a, **kw):
    seen["combo"] += src.count("combo_grid_meta")
    for l in src.splitlines():
        if "'grid_type'" in l:
            i = l.find("'grid_type'"); print("    grid_type:", l[i:i+60])
    return orig(self, model, src, device, *a, **kw)
dg.DynaGraphRunner.__init__ = spy

class Three(torch.nn.Module):   # three 1D pointwise ops, different numel
    def forward(self, a, b, c):
        return torch.sigmoid(a) * 2, torch.relu(b) + 1, torch.tanh(c) - 1
class Mixed(torch.nn.Module):   # two that need 2D tiles (transpose): the grid's y axis takes the max over the sub-kernels
    def forward(self, a, b):
        return a.t().contiguous() + 1, b.t().contiguous() * 3

def run(name, mk, shapes):
    torch._dynamo.reset()
    m = globals()[name]().cuda()
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")
    worst = 0.0
    with torch.no_grad():
        for L in shapes:
            args = mk(L)
            out = f(*args)
            ref = m(*args)
            for o, r in zip(out, ref):
                # compared with eager: sigmoid/tanh implementations differ, so float32 rounding differences are allowed
                worst = max(worst, ((o - r).abs().max() / r.abs().max().clamp_min(1)).item())
    return worst

results = []
mk3 = lambda L: (torch.randn(L, 64, device="cuda"), torch.randn(L, 32, device="cuda"), torch.randn(L, 16, device="cuda"))
mk2 = lambda L: (torch.randn(L, 96, device="cuda"), torch.randn(L, 40, device="cuda"))
for name, mk in (("Three", mk3), ("Mixed", mk2)):
    print(f"== {name}")
    seen["combo"] = 0; seen["fallback"] = 0
    w = run(name, mk, (256, 256, 100, 300, 17, 256))
    ok = w < 1e-5 and seen["combo"] > 0 and seen["fallback"] == 0
    results.append(ok)
    print(f"    combo kernels seen {seen['combo']}  fallbacks {seen['fallback']}  worst rel diff {w:.1e}  {'ok' if ok else 'FAIL'}")
print("== pass=%d fail=%d" % (sum(results), len(results) - sum(results)))
sys.exit(0 if all(results) else 1)
