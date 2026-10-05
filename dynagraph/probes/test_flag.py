#!/usr/bin/env python3
"""Final acceptance: with TORCHINDUCTOR_DYNAGRAPH=1, torch.compile gets it out of the box.

Criteria:
  1. results are correct for several shapes (compared with eager, tolerating algorithmic differences)
  2. only one graph is recorded (judged by counter or log), not one per shape
"""
import os, sys

SHAPES = tuple(int(v) for v in os.environ.get("DG_SHAPES", "512,256,333,64,7").split(","))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")


def run(dynagraph: bool):
    import torch, torch._inductor.config as ic
    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    # GEMM must be routed away from cuBLAS: extern_kernels.mm does not go through the static launcher,
    # that node has no handle and the planner cannot patch it -- the handle-count check correctly makes the whole graph fall back.
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"

    from torch._dynamo.utils import counters
    counters.clear()
    n_cgify = {"n": 0}
    import torch._inductor.compile_fx as cfx
    o_cg = cfx.cudagraphify
    def spy_cg(*a, **kw):
        n_cgify["n"] += 1
        return o_cg(*a, **kw)
    cfx.cudagraphify = spy_cg
    n_record = {"n": 0}
    import torch._inductor.cudagraph_trees as ct
    orig = ct.CUDAGraphTreeManager.record_function
    def spy(self, *a, **kw):
        n_record["n"] += 1
        return orig(self, *a, **kw)
    ct.CUDAGraphTreeManager.record_function = spy

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.l = torch.nn.Linear(128, 128)
        def forward(self, x):
            h = torch.relu(self.l(x))
            return h - h.mean(dim=-1, keepdim=True)

    # Both runs must get the same weights and the same inputs, otherwise the outputs cannot be compared bit for bit.
    torch.manual_seed(0)
    m = M().cuda().eval()
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")

    outs = {}
    try:
        torch.manual_seed(1)
        for M_ in SHAPES:
            x = torch.randn(M_, 128, device="cuda")
            with torch.no_grad():
                # Call each shape twice: the first time cudagraph_trees sees a FunctionID
                # it only does an eager warmup, and it records for real on the second call. With a single call it records
                # no graph, the control group becomes "not using cudagraph at all", and the comparison is meaningless.
                f(x)
                outs[M_] = (f(x).float().cpu().clone(),
                            m(x).float().cpu().clone())
    finally:
        ct.CUDAGraphTreeManager.record_function = orig
        cfx.cudagraphify = o_cg
    skips = dict(counters["inductor"])
    return n_record["n"], outs, n_cgify["n"], skips


def main():
    import torch
    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1

    res, rec = {}, {}
    for flag in (False, True):
        n, outs, ncg, skips = run(flag)
        res[flag] = outs
        rec[flag] = n
        print(f"\n  dynagraph={flag}")
        print(f"    cudagraphify called {ncg} times; recordings {n}")
        interesting = {k: v for k, v in skips.items()
                       if "cudagraph" in k or "skip" in k or "partition" in k}
        print(f"    relevant counters: {interesting or '(empty)'}")

    # The reference is the same compile path with dynagraph off, not eager: GEMM is routed to Triton,
    # which differs algorithmically from eager's cuBLAS anyway, and comparing against eager would read the algorithmic difference as a correctness problem.
    #
    # But the two sides should not be bit-identical either: the control group re-records a graph per shape and can pick an autotune
    # config for each; DynaGraph has one graph, with the config pinned to the one selected at recording time. A different XBLOCK
    # gives a different reduction order, so one ULP of difference is correct. So the criterion is "no farther from eager than
    # the control group", not "identical to the control group".
    print("\n  compare with dynagraph=False (criterion: no farther from eager than the control group):")
    bad = 0
    for M_ in SHAPES:
        (a, ea), (b, eb) = res[False][M_], res[True][M_]
        if a.shape != b.shape:
            print(f"    M={M_} shape {tuple(b.shape)} != {tuple(a.shape)}  FAIL")
            bad += 1; continue
        # The eager references of the two runs must be bit-identical; otherwise it is a seeding problem, not DynaGraph's fault.
        seed_ok = (ea - eb).abs().max().item() == 0
        scale = max(ea.abs().max().item(), 1e-9)
        ctl = (a - ea).abs().max().item() / scale
        dyn = (b - eb).abs().max().item() / scale
        ok = seed_ok and dyn <= max(ctl * 1.5, 1e-6)
        print(f"    M={M_} ctl<->dyna {(a - b).abs().max().item():.2e}"
              f" | ctl<->eager {ctl:.1e} | dyna<->eager {dyn:.1e}"
              f" | same seed {seed_ok}" + ("  OK" if ok else "  FAIL"))
        bad += not ok

    # The whole point: the control group records one graph per shape, DynaGraph serves them all with one.
    print(f"\n  recordings {rec[False]} -> {rec[True]} ({len(SHAPES)} shapes)")
    if rec[True] != 0 or rec[False] != len(SHAPES):
        print("  FAIL: recording count mismatch: DynaGraph should record nothing, the control group once per shape")
        bad += 1
    print("\n  " + ("all passed" if not bad else f"{bad} checks failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
