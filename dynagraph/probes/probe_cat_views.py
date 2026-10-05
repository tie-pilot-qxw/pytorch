#!/usr/bin/env python3
"""Slice views from torch.cat: Inductor has each producer kernel write straight into `reinterpret_tensor(buf, ..., offset)` of the concat buffer,
and the offset may be symbolic (channel offset x spatial size). A pointer patch that drops the offset writes to the wrong place (inception hit this).
Batch and spatial dims are both dynamic, and both paths run; plus an "offset view of an input" (a slice of the second half of x's channels).

    UPDATE=host|device python probe_cat_views.py
"""
from __future__ import annotations
import logging, os, sys
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct

class CatModel(torch.nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(c, 1, 1))
        self.w2 = torch.nn.Parameter(torch.randn(c, 1, 1))
    def forward(self, x):
        a = torch.relu(x * self.w1)
        b = torch.sigmoid(x * self.w2) - 0.5
        c = torch.cat([a, b, x[:, 8:]], dim=1)          # three slices; the third is an offset view of the input
        return c * 1.5 + torch.cat([b, a], dim=1).mean(dim=(1, 2, 3), keepdim=True)

class Grab(logging.Handler):
    def __init__(self): super().__init__(); self.msgs = []
    def emit(self, rec): self.msgs.append(rec.getMessage())

def main() -> int:
    torch._dynamo.reset()
    ic.triton.dynagraph = True
    ic.triton.dynagraph_extern_child = True
    mode = os.environ.get("UPDATE", "auto"); ic.triton.dynagraph_update = mode
    m = CatModel().cuda().eval()
    grab = Grab(); lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.INFO); lg.addHandler(grab)
    n_rec = {"v": 0}; orig = ct.CUDAGraphTreeManager.record_function
    def spy(self, *a, **kw): n_rec["v"] += 1; return orig(self, *a, **kw)
    ct.CUDAGraphTreeManager.record_function = spy
    try:
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        worst = 0.0
        with torch.no_grad():
            for B, H in ((8, 32), (3, 20), (5, 32), (2, 12), (8, 32)):
                g = torch.Generator(device="cuda"); g.manual_seed(B * 100 + H)
                x = torch.randn(B, 16, H, H, device="cuda", generator=g)
                got = f(x); got = f(x)
                ref = m(x)
                worst = max(worst, ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item())
    finally:
        ct.CUDAGraphTreeManager.record_function = orig; lg.removeHandler(grab)
    tags = sorted({t.split("[", 1)[1].split("]", 1)[0] for t in grab.msgs if t.startswith("DynaGraph fallback [")})
    detail = [t[:140] for t in grab.msgs if t.startswith("DynaGraph fallback [") or t.startswith("DynaGraph mismatch")][:2]
    ok = not tags and n_rec["v"] == 0 and worst <= 1e-5
    print(f"    mode={mode}  recordings {n_rec['v']}  rel diff {worst:.1e}  fallbacks {','.join(tags) or '-'}  {'OK' if ok else 'FAIL'}")
    for d in (detail if not ok else []): print("     ", d)
    print("\n  " + ("all passed" if ok else "1 check failed"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
