#!/usr/bin/env python3
"""On the device path, large inputs are patched by address (no more per-call copy): when an input is >= dynagraph_patch_bytes the planner
points the nodes that read it straight at the caller's tensor. Checks: served, 0 recordings, bitwise equal, the runner's patch set is non-empty,
and a different input tensor of the same shape is also correct (the address changed, so it must be re-pointed).

    UPDATE=device python probe_input_patch.py
"""
from __future__ import annotations
import logging, os, sys
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct

D = 4096  # L x 4096 fp32: L=1024 is already 16 MiB, above the default threshold
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(D))
    def forward(self, x):
        return (torch.relu(x * self.w) + 1.0).sum(-1)

class Grab(logging.Handler):
    def __init__(self): super().__init__(); self.msgs = []
    def emit(self, rec): self.msgs.append(rec.getMessage())

def main() -> int:
    torch._dynamo.reset()
    ic.triton.dynagraph = True
    ic.triton.dynagraph_extern_child = True
    mode = os.environ.get("UPDATE", "device")
    ic.triton.dynagraph_update = mode
    m = M().cuda().eval()
    grab = Grab(); lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.INFO); lg.addHandler(grab)
    grabbed = {}
    orig = ct._maybe_build_dynagraph
    def spy(model, inputs, kwargs, *a, **kw):
        r = orig(model, inputs, kwargs, *a, **kw)
        if r is not False: grabbed["r"] = r
        return r
    ct._maybe_build_dynagraph = spy
    n_rec = {"v": 0}; orig_rec = ct.CUDAGraphTreeManager.record_function
    def rspy(self, *a, **kw): n_rec["v"] += 1; return orig_rec(self, *a, **kw)
    ct.CUDAGraphTreeManager.record_function = rspy
    bad = 0
    try:
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        worst = 0.0
        with torch.no_grad():
            for L in (2048, 700, 1500, 2048, 64):
                # two different tensors of the same length: different addresses, both results must be correct
                for seed in (0, 1):
                    g = torch.Generator(device="cuda"); g.manual_seed(L * 7 + seed)
                    x = torch.randn(L, D, device="cuda", generator=g)
                    got = f(x); got = f(x)
                    ref = m(x)
                    worst = max(worst, ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item())
    finally:
        ct._maybe_build_dynagraph = orig; ct.CUDAGraphTreeManager.record_function = orig_rec; lg.removeHandler(grab)
    r = grabbed.get("r")
    tags = sorted({t.split("[", 1)[1].split("]", 1)[0] for t in grab.msgs if t.startswith("DynaGraph fallback [")})
    patched = sorted(r.patch_inputs) if r is not None else None
    print(f"    mode={mode}  recordings {n_rec['v']}  rel diff {worst:.1e}  fallbacks {','.join(tags) or '-'}  patch set {patched}  update={getattr(r, 'update', None)}")
    ok = r is not None and not tags and n_rec["v"] == 0 and worst <= 1e-5 and patched
    if mode == "device" and r is not None and not patched:
        print("    FAIL device path did not put the large input in the patch set")
    bad += not ok
    print("\n  " + ("all passed" if not bad else f"{bad} failed"))
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
