#!/usr/bin/env python3
"""Probe (class B: must fall back): when GEMM stays in cuBLAS, DynaGraph must refuse the whole graph.

With max_autotune_gemm off, Linear / matmul compile to extern_kernels.mm -- that is cuBLAS,
which does not go through StaticCudaLauncher, so capture gets no device-updatable node handle and the planner cannot
reach the node. If one graph really served it, the node would stay forever at the shape it was recorded at: no
error, just wrong data. So the criteria are

  1. dynagraph=True must record one graph per shape just like the control (= it really fell back)
  2. after falling back the results are still correct (the reference is the control, not eager)

On its own, "same recording count" would also pass "if DynaGraph did nothing at all", so here we also require
that the DynaGraph path was really reached: grab the wrapper source, confirm it really contains extern_kernels,
and record whether _capture finished and how many Triton kernels the kernel table has.

Both topologies are tried because they hit different guards:
  mm_then_triton -- the extern mm output feeds a Triton kernel (whose pointer the arena rewrites)
  triton_then_mm -- a Triton kernel output feeds the extern mm, and the mm result is the graph's return value
In the second, neither the input nor the output of the extern node is a Triton argument the planner rewrites; it is the
harder of the two to catch, so it must be tried on its own.

Measured conclusion (see the comment at the end of the file): the handle-count check does not catch extern kernels.
"""
import os, sys, io, re, logging

SHAPES = tuple(int(v) for v in os.environ.get("DG_SHAPES", "256,128,77,32").split(","))
KINDS = ("mm_then_triton", "triton_then_mm")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

# DynaGraph logs its fallback reasons at INFO; without capturing them you cannot see which guard fired.
LOGBUF = io.StringIO()


def attach_log():
    h = logging.StreamHandler(LOGBUF)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    for name in ("torch._inductor.cudagraph_trees", "torch._inductor.dynagraph"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(h)


def make_model(kind, torch):
    class MMFirst(torch.nn.Module):
        """extern mm first: its output is read by the Triton kernel after it."""
        def __init__(self):
            super().__init__(); self.l = torch.nn.Linear(128, 128)
        def forward(self, x):
            h = torch.relu(self.l(x))
            return h - h.mean(dim=-1, keepdim=True)

    class MMLast(torch.nn.Module):
        """extern mm last: the graph's return value is the very buffer cuBLAS writes."""
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(128, 128) * 0.05)
        def forward(self, x):
            y = torch.relu(x) * 1.5
            return y @ self.w

    return (MMFirst if kind == "mm_then_triton" else MMLast)()


def run(dynagraph: bool, kind: str):
    import torch, torch._inductor.config as ic
    torch._dynamo.reset()
    LOGBUF.seek(0); LOGBUF.truncate(0)
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    # This probe wants extern_kernels: unlike the other probes, gemm autotune is turned off here
    # so GEMM stays in cuBLAS instead of being routed to a Triton template.
    ic.max_autotune_gemm = False

    n_record = {"n": 0}
    import torch._inductor.cudagraph_trees as ct
    orig = ct.CUDAGraphTreeManager.record_function
    def spy(self, *a, **kw):
        n_record["n"] += 1
        return orig(self, *a, **kw)
    ct.CUDAGraphTreeManager.record_function = spy

    # Grab the wrapper source DynaGraph actually reads: it is the only evidence that "this path was reached, and
    # the graph it saw really contains extern_kernels". Without it, "every shape recorded a graph" could also
    # just mean the flag never took effect. _capture's return value also tells us whether the handle-count check passed.
    import torch._inductor.dynagraph as dg
    seen = {"src": None, "n_kernels": None, "capture_ok": None, "built": 0}
    o_src = dg._wrapper_source
    def spy_src(model):
        s = o_src(model)
        if s and seen["src"] is None:
            seen["src"] = s
        return s
    o_cap = dg.DynaGraphRunner._capture
    def spy_cap(self):
        seen["n_kernels"] = len(self.kernels or [])
        seen["built"] += 1
        r = o_cap(self)
        seen["capture_ok"] = r
        return r
    if dynagraph:
        dg._wrapper_source = spy_src
        dg.DynaGraphRunner._capture = spy_cap

    # Both runs must get the same weights and the same inputs, otherwise the outputs cannot be compared bitwise.
    torch.manual_seed(0)
    m = make_model(kind, torch).cuda().eval()
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")

    outs = {}
    try:
        torch.manual_seed(1)
        for M_ in SHAPES:
            x = torch.randn(M_, 128, device="cuda")
            with torch.no_grad():
                # Call each shape twice: the first time cudagraph_trees sees a FunctionID it only does
                # an eager warmup and records on the second call. With one call the control records no graph at all,
                # and "still one graph per shape after fallback" has nothing to compare against.
                f(x)
                outs[M_] = (f(x).float().cpu().clone(),
                            m(x).float().cpu().clone())
    finally:
        ct.CUDAGraphTreeManager.record_function = orig
        dg._wrapper_source = o_src
        dg.DynaGraphRunner._capture = o_cap
    tags = re.findall(r"DynaGraph fallback \[([\w-]+)\][^\n]*", LOGBUF.getvalue())
    lines = [ln for ln in LOGBUF.getvalue().splitlines() if "DynaGraph" in ln]
    return n_record["n"], outs, seen, tags, lines


def check(kind):
    print(f"\n=== model {kind} ===")
    res, rec = {}, {}
    seen, tags, lines = {}, [], []
    for flag in (False, True):
        n, outs, sn, tg, ln = run(flag, kind)
        res[flag], rec[flag] = outs, n
        if flag:
            seen, tags, lines = sn, tg, ln
        print(f"  dynagraph={flag}: recorded {n} times")

    bad = 0

    # ---- target path check: does this graph really contain extern_kernels ----
    src = seen.get("src")
    extern = sorted(set(re.findall(r"extern_kernels\.\w+", src))) if src else []
    print("  Target path check:")
    print(f"    wrapper read: {bool(src)} | extern calls: {extern or '(none)'}")
    print(f"    Triton kernels in the kernel table: {seen.get('n_kernels')}"
          f" | _capture returned: {seen.get('capture_ok')}"
          f" | times build reached _capture: {seen.get('built')}")
    if not (src and extern):
        print("    FAIL missed the target path: either DynaGraph never ran, or GEMM did not land on cuBLAS")
        bad += 1

    print(f"  fallback reason tags: {tags or '(no fallback log at all)'}")
    for ln in lines[:10]:
        print("    " + ln)
    # Record which guard actually blocked it. The handle-count check (handle-mismatch) is the one the design claims
    # blocks "kernels that did not go through the static launcher", so whether it fires matters a lot.
    print(f"  blocked by the handle-count check: {'handle-mismatch' in tags}")

    # ---- correctness: the reference is the same compile path with dynagraph off, not eager ----
    for M_ in SHAPES:
        (a, ea), (b, eb) = res[False][M_], res[True][M_]
        if a.shape != b.shape:
            print(f"    M={M_} shape {tuple(b.shape)} != {tuple(a.shape)}  FAIL")
            bad += 1; continue
        # The eager references of the two runs must be bitwise identical; otherwise it is a seeding problem, not DynaGraph's fault.
        seed_ok = (ea - eb).abs().max().item() == 0
        scale = max(ea.abs().max().item(), 1e-9)
        ctl = (a - ea).abs().max().item() / scale
        dyn = (b - eb).abs().max().item() / scale
        ok = seed_ok and dyn <= max(ctl * 1.5, 1e-6)
        print(f"    M={M_} ctl<->dyna {(a - b).abs().max().item():.2e}"
              f" | ctl<->eager {ctl:.1e} | dyna<->eager {dyn:.1e}"
              f" | same seed {seed_ok}" + ("  OK" if ok else "  FAIL"))
        bad += not ok

    # ---- core criterion: must fall back, i.e. the recording count equals the control's ----
    print(f"  recording count {rec[False]} -> {rec[True]} ({len(SHAPES)} shapes)")
    if rec[False] != len(SHAPES):
        print("  FAIL the control did not record one graph per shape; the experiment itself is invalid")
        bad += 1
    if rec[True] == 0:
        print("  FAIL DynaGraph served this graph: the cuBLAS node cannot be patched, so this is silently wrong data")
        bad += 1
    elif rec[True] != rec[False]:
        print(f"  FAIL fell back, but the recording count differs from the control ({rec[True]} != {rec[False]})")
        bad += 1
    print("  " + (f"{kind}: passed" if not bad else f"{kind}: {bad} failed"))
    return bad, tags


def main():
    import torch
    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1
    attach_log()
    bad, all_tags = 0, {}
    for kind in KINDS:
        b, tags = check(kind)
        bad += b
        all_tags[kind] = tags
    print("\n  Fallback reasons per topology:")
    for k, t in all_tags.items():
        print(f"    {k}: {t or '(none)'}")
    if not any("extern-launch" in t for t in all_tags.values()):
        # If this structural check does not fire, it is not doing its job -- what blocks extern kernels then falls back to
        # the empirical numeric comparison of the self-check replay; see the measurement record at the end of the file.
        print("  FAIL extern-launch did not fire: the structural check is broken, only the numeric comparison still blocks it")
        bad += 1
    print("\n  " + ("all passed" if not bad else f"{bad} failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

# ---------------------------------------------------------------- measurement record
# This probe changed the implementation, so the record has two parts.
#
# [First run, 2026-09-18, CUDA_VISIBLE_DEVICES=3] For both topologies:
#   recorded 4 -> 4 (really fell back), results match the control,
#   fallback tag = selfcheck-mismatch, _capture returned True, 1 Triton kernel in the kernel table.
#
# In other words, the "handle count mismatch" check the task expected never fired once, and it is
# structurally blind to extern kernels: extern_kernels.mm is not a CachingAutotuner, so it never enters the
# kernel table; nor does it go through StaticCudaLauncher, so it never enters the handle table. Both sides are short by one,
# len(handles) == len(kernels) still holds, and _capture happily returns True.
# (The comment in dynagraph.py says "a count mismatch means some kernel did not go through the static launcher";
#   that direction holds, but the converse does not: when a kernel skips the static launcher the counts can still match.)
#
# What actually blocked it was the self-check replay at the recorded shape: the arena relayout moved the Triton kernel's pointer
# argument (topology one) / the read address of the output buffer (topology two) away from the address cuBLAS writes to,
# so the replay result disagreed with what the wrapper computed directly, and it fell back.
# This means the extern kernel was blocked back then by an empirical numeric comparison, not by a structural
# decision. No wrong data was observed, but the nature of the guard differed from what the code comment claimed.
#
# [Rerun after the fix, same day] dynagraph.py gained unreachable_launch(): it scans the entry function
# and, on seeing extern_kernels. / torch.ops. / aten. / .item(), refuses with extern-launch,
# blocking it already at the unusable_reason() stage, so capture never happens. Rerun results:
#   both topologies tag ['extern-launch'], recorded 4 -> 4, results still bitwise identical to the control.
# The criterion in main() above was inverted accordingly: it fails only when extern-launch does not fire.
