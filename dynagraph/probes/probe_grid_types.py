#!/usr/bin/env python3
"""Grid types not yet modelled: split scan (cumsum over a long dim), cooperative reduction (few rows, long reductions),
mix-order reduction (one tensor reduced in both directions), bmm template (BatchMatmulGrid3D).
One small model each, with the dynamic dim on a different grid axis in each. Checks whether it is served, fallback tags, numerics, recording count.

    UPDATE=host|device python probe_grid_types.py
"""
from __future__ import annotations
import logging, os, sys
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct

class Grab(logging.Handler):
    def __init__(self): super().__init__(); self.msgs = []
    def emit(self, rec): self.msgs.append(rec.getMessage())

def run(name, make, shapes, make_input, setup=None, expect_types=(), device_ok_tags=()):
    torch._dynamo.reset()
    ic.triton.dynagraph = True
    ic.triton.dynagraph_extern_child = True
    ic.triton.cooperative_reductions = False
    ic.triton.force_cooperative_reductions = False
    ic.max_autotune_gemm = False
    if setup: setup()
    m = make()
    grab = Grab(); lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.INFO); lg.addHandler(grab)
    n_rec = {"v": 0}; orig = ct.CUDAGraphTreeManager.record_function
    def spy(self, *a, **kw): n_rec["v"] += 1; return orig(self, *a, **kw)
    ct.CUDAGraphTreeManager.record_function = spy
    types_seen = set()
    orig_extract = None
    try:
        from torch._inductor import dynagraph as dg
        orig_extract = dg.extract_kernel_table
        # *a/**kw: this wraps a torch-internal function whose signature grows.
        def extract_spy(*a, **kw):
            out = orig_extract(*a, **kw)
            if out and out[0]:
                for k in out[0]: types_seen.add(k.get("grid_type"))
            return out
        dg.extract_kernel_table = extract_spy
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        worst = 0.0
        with torch.no_grad():
            for L in shapes:
                x = make_input(L)
                for _ in range(2):
                    got = f(*x) if isinstance(x, tuple) else f(x)
                ref = m(*x) if isinstance(x, tuple) else m(x)
                worst = max(worst, ((got.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-6)).item())
    except Exception as exc:
        print(f"    {name:<14} ERR {type(exc).__name__}: {str(exc)[:110]}")
        return 1
    finally:
        ct.CUDAGraphTreeManager.record_function = orig
        lg.removeHandler(grab)
        if orig_extract: dg.extract_kernel_table = orig_extract
    tags = sorted({t.split("[", 1)[1].split("]", 1)[0] for t in grab.msgs if t.startswith("DynaGraph fallback [")})
    detail = [t[:100] for t in grab.msgs if t.startswith("DynaGraph fallback [")][:1]
    hit = [t for t in expect_types if t in types_seen]
    ok = not tags and n_rec["v"] == 0 and worst <= 2e-2 and (not expect_types or hit)
    if not ok and os.environ.get("UPDATE", "auto") == "device" and tags and set(tags) <= set(device_ok_tags):
        # Forced device path: this kernel is not reachable by a planner (no
        # static-launcher handle), and auto routes such a region to the host.
        ok = worst <= 2e-2
        print(f"    {name:<14} falls back as expected when the device path is forced {tags} (auto takes the host path)")
    print(f"    {name:<14} grid {sorted(t for t in types_seen if t)}  recordings {n_rec['v']}  rel diff {worst:.1e}  "
          f"{','.join(tags) or '-'} {'ok' if ok else 'FAIL'} {detail[0] if detail and not ok else ''}")
    if expect_types and not hit:
        print(f"      (this model did not generate {expect_types}; the probe missed its target)")
    return 0 if ok else 1

def main() -> int:
    bad = 0
    shapes = [1024, 300, 777, 64]
    class Cumsum(torch.nn.Module):
        def forward(self, x): return torch.cumsum(x, dim=1) * 0.5
    bad += run("split_scan", lambda: Cumsum().cuda(), [8, 3, 5, 2],
               lambda L: torch.randn(L, 1 << 20, device="cuda"), expect_types=("SplitScanGrid",))
    class LongSum(torch.nn.Module):
        def forward(self, x): return (x * 2).sum(dim=1)
    def coop(): ic.triton.cooperative_reductions = True; ic.triton.force_cooperative_reductions = True
    bad += run("cooperative", lambda: LongSum().cuda(), [8, 3, 5, 2],
               lambda L: torch.randn(L, 1 << 20, device="cuda"), setup=coop, expect_types=("CooperativeReductionGrid",),
               device_ok_tags=("handle-mismatch",))
    class MixOrder(torch.nn.Module):
        def forward(self, x): return x.sum(dim=0) + x.sum(dim=1).mean()
    def mix():
        if hasattr(ic.triton, "mix_order_reduction"): ic.triton.mix_order_reduction = True
    bad += run("mix_order", lambda: MixOrder().cuda(), shapes,
               lambda L: torch.randn(L, 2048, device="cuda"), setup=mix, expect_types=())
    class Bmm(torch.nn.Module):
        def forward(self, a, b): return torch.bmm(a, b).relu()
    def tri(): ic.max_autotune_gemm = True; ic.max_autotune_gemm_backends = "TRITON"
    bad += run("bmm_triton", lambda: Bmm().cuda(), [64, 20, 33, 7],
               lambda B: (torch.randn(B, 64, 64, device="cuda"), torch.randn(B, 64, 64, device="cuda")),
               setup=tri, expect_types=())
    print("\n  " + ("all passed" if not bad else f"{bad} item(s) failed"))
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
