#!/usr/bin/env python3
"""Two mutually independent symbolic dims: (B, L, D), with B and L varying independently and D fixed.

Why a separate probe: test_arena_e2e.py already showed that arena re-layout can serve a shape space
where "the total is fixed and the split varies", but it **drives the runner directly** -- it calls
extract_kernel_table / generate_planner / cuLaunchKernel itself, bypassing
cudagraph_trees. What actually ships is the public switch ic.triton.dynagraph, so this runs
the same thing through the public torch.compile path.

This probe targets multiple symbols specifically:
  * ctx is a 1D array indexed by sym_index, and each of the two symbols must land in its own slot.
    If the planner swapped S(0)/S(1), the grids and the kernels' numel arguments would all be swapped;
  * output sizes are computed by the runner in __call__ with _eval_int(env), and the two outputs
    are (B, D) and (L, D). So **every shape deliberately has B != L**: as soon as the symbols are
    crossed, the output shape is immediately wrong, without waiting for the numeric comparison.

How the shapes are chosen: the first shape must be the one with the largest B*L. _maybe_build_dynagraph builds
the runner on the **first** shape it sees; the runner clones that call's inputs into
static_inputs, and every later __call__ does
`dst.view(-1)[:src.numel()].copy_(...)`; a later shape with more elements then
raises RuntimeError directly (reproduced on this machine: (8,32) then (64,256); dynagraph=False is fine,
dynagraph=True fails with "size of tensor a (32768) must match ... b (2097152)").
The dynagraph.py module docstring says "the graph must be recorded at the largest shape of the range", and bucket_of()
can compute that upper bound, but nobody on the public path calls it. This is a real defect in the implementation,
not what this probe tests, so here we go along with it by putting the largest first, and record it separately.
Other than that B and L vary fully independently, including a
(16,256) -> (8,512) pair where "one shrinks while the other grows at the same time",
and L=512 also exceeds the recording-time L=256 -- no single recorded shape can cover all of them.
"""
import os, sys

# (B, L); D fixed at 128. The first has the largest B*L; see the docstring for why.
SHAPES = ((64, 256), (64, 32), (16, 256), (8, 512), (32, 96))
D = 128
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

_OUT = os.environ.get("DG_OUT", "/tmp/dynagraph_out")
os.makedirs(_OUT, exist_ok=True)
DUMP = os.path.join(_OUT, "multi_symbol_planner.cu")
# Dump the planner source to disk as evidence that "the multi-symbol path was really taken":
# its dg_eval table shows both S(0) and S(1) being used.
os.environ["TORCHINDUCTOR_DYNAGRAPH_DUMP"] = DUMP


def run(dynagraph: bool):
    import torch, torch._inductor.config as ic
    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    # GEMM must be routed away from cuBLAS: extern_kernels.mm does not go through the static launcher,
    # that node has no handle, the handle-count check makes the whole graph fall back, and the probe tests nothing.
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

    # The runner-building step itself does not log (usable() false and an env length mismatch both silently
    # return False), so intercept it here: when we get a runner, copy out the symbol table it recognized,
    # which is first-hand evidence that "multiple symbols really were modelled"; if we get none, we know it fell back.
    o_build = ct._maybe_build_dynagraph
    info = {"built": None, "symbols": None, "sym_from_input": None,
            "slots": None, "outputs": None}
    def spy_build(model, inputs, kwargs, *a, **kw):
        r = o_build(model, inputs, kwargs, *a, **kw)
        if info["built"] is None:
            info["built"] = r is not False
            if r is not False:
                info["symbols"] = list(getattr(r, "symbols", []) or [])
                info["sym_from_input"] = dict(getattr(r, "sym_from_input", {}) or {})
                info["slots"] = getattr(r, "n_slots", None)
                info["outputs"] = list(getattr(r, "outputs", []) or [])
        return r
    ct._maybe_build_dynagraph = spy_build

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.l = torch.nn.Linear(D, D)
        def forward(self, x):            # x: (B, L, D)
            h = torch.relu(self.l(x))    # the mm's M dim is B*L: two symbols multiplied in one grid
            # Each of the two reductions eats one symbol, leaving different output sizes:
            # (B, D) follows only B, (L, D) follows only L. In the arena one of these grows while the other shrinks.
            return h.sum(dim=1), h.sum(dim=0)

    # Both runs must get the same weights and the same inputs, otherwise the outputs cannot be compared bit for bit.
    torch.manual_seed(0)
    m = M().cuda().eval()
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")

    outs = {}
    try:
        torch.manual_seed(1)
        for B, L in SHAPES:
            x = torch.randn(B, L, D, device="cuda")
            with torch.no_grad():
                # Call each shape twice: the first time cudagraph_trees sees a FunctionID
                # it only does an eager warmup, and it records for real on the second call. With a single call
                # the control group records no graph, and "control group = really running cudagraph" no longer holds.
                f(x)
                got = f(x)
                ref = m(x)
                outs[(B, L)] = (
                    tuple(t.detach().float().cpu().clone() for t in got),
                    tuple(t.detach().float().cpu().clone() for t in ref),
                )
    finally:
        ct.CUDAGraphTreeManager.record_function = orig
        ct._maybe_build_dynagraph = o_build
        cfx.cudagraphify = o_cg
    skips = dict(counters["inductor"])
    return n_record["n"], outs, n_cgify["n"], skips, info


def dump_evidence():
    """Read evidence from the dumped planner source: how many S(i) actually appear in the symbol table."""
    if not os.path.exists(DUMP):
        return None
    txt = open(DUMP).read()
    import re
    used = sorted(set(re.findall(r"\bS\((\d+)\)", txt)) |
                  set(re.findall(r"ctx\[\((\d+)\)\]", txt)))
    exprs = [ln.strip() for ln in txt.splitlines()
             if re.match(r"\s*case \d+: return ", ln)]
    grids = [ln.strip() for ln in txt.splitlines() if "gx=" in ln]
    return used, exprs, grids


def main():
    import logging
    # If it falls back, at least the reason must be visible (the exception branch logs at INFO).
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("torch._inductor.cudagraph_trees").setLevel(logging.INFO)
    logging.getLogger("torch._inductor.dynagraph").setLevel(logging.INFO)

    import torch
    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1
    if os.path.exists(DUMP):
        os.remove(DUMP)   # do not let leftovers from the previous run pose as this run's evidence

    res, rec, info = {}, {}, {}
    for flag in (False, True):
        n, outs, ncg, skips, inf = run(flag)
        res[flag] = outs
        rec[flag] = n
        info[flag] = inf
        print(f"\n  dynagraph={flag}")
        print(f"    cudagraphify called {ncg} times; recordings {n}")
        interesting = {k: v for k, v in skips.items()
                       if "cudagraph" in k or "skip" in k or "partition" in k}
        print(f"    relevant counters: {interesting or '(empty)'}")
        if flag:
            print(f"    runner built: {inf['built']}")
            print(f"    symbols: {inf['symbols']}  symbol<-input index: {inf['sym_from_input']}")
            print(f"    arena slots: {inf['slots']}  output buffers: {inf['outputs']}")

    bad = 0

    # Hard evidence of multiple symbols: there must be more than one, and all must be symints taken from the inputs.
    syms = info[True]["symbols"] or []
    if not info[True]["built"]:
        print("\n  FAIL: DynaGraph was not built (it fell back), so this probe missed the target path")
        bad += 1
    elif len(syms) < 2 or len(info[True]["sym_from_input"] or {}) < 2:
        print(f"\n  FAIL: only {len(syms)} symbol(s) recognized, the multi-symbol path was not hit")
        bad += 1

    ev = dump_evidence()
    if ev is None:
        print("  (planner not dumped: the build never reached the codegen step)")
    else:
        used, exprs, grids = ev
        print(f"\n  ctx slots used in the planner: S({'), S('.join(used)})"
              f"  -- {len(used)} in total")
        for e in exprs:
            print(f"    expr    {e}")
        for g in grids:
            print(f"    grid    {g}")
        if len(used) < 2:
            print("  FAIL: the planner references only one symbol slot, multi-symbol ctx indexing was not really used")
            bad += 1

    # The reference is the same compile path with dynagraph off, not eager: GEMM is routed to
    # Triton, which differs algorithmically from eager's cuBLAS anyway, and comparing against eager would read
    # the algorithmic difference as a correctness problem. The two sides should not be bit-identical either: the control
    # group re-records a graph per shape and can pick an autotune config for each; DynaGraph has one graph, with the config
    # pinned to the one selected at recording time, and a different reduction order giving one ULP of difference is correct. So the criterion is
    # "no farther from eager than the control group".
    print("\n  compare with dynagraph=False (criterion: exact shape match + no farther from eager than the control group):")
    for B, L in SHAPES:
        (a, ea), (b, eb) = res[False][(B, L)], res[True][(B, L)]
        want = ((B, D), (L, D))
        shp = tuple(tuple(t.shape) for t in b)
        if shp != want:
            print(f"    B={B:<4} L={L:<5} output shape {shp} != {want}  FAIL"
                  f"  <- typical symptom of crossed symbols")
            bad += 1; continue
        line, ok_all = [], True
        for j, (aj, bj, eaj, ebj) in enumerate(zip(a, b, ea, eb)):
            # The eager references of the two runs must be bit-identical; otherwise it is a seeding problem, not DynaGraph's fault.
            seed_ok = (eaj - ebj).abs().max().item() == 0
            scale = max(eaj.abs().max().item(), 1e-9)
            ctl = (aj - eaj).abs().max().item() / scale
            dyn = (bj - ebj).abs().max().item() / scale
            ok = seed_ok and dyn <= max(ctl * 1.5, 1e-6)
            ok_all &= ok
            line.append(f"out{j}: ctl<->dyna {(aj - bj).abs().max().item():.1e}"
                        f" ctl<->eager {ctl:.1e} dyna<->eager {dyn:.1e} same seed {seed_ok}")
        print(f"    B={B:<4} L={L:<5} " + " | ".join(line) + ("  OK" if ok_all else "  FAIL"))
        bad += not ok_all

    # For this probe to mean anything, the shape space must really contain shapes "the recorded graph cannot cover".
    # Recording happens at the first shape (64,256), where out1 is (256,128); at (8,512)
    # out1 becomes (512,128), larger than at recording time. If every shape stayed within the recorded shape
    # in every dim, the recorded graph's original buffers alone would fit them, arena re-layout would never
    # be forced, and the "one graph serves the whole shape space" conclusion would have to be discounted.
    cb, cl = SHAPES[0]
    grew = [(B, L) for B, L in SHAPES if B > cb or L > cl]
    print(f"\n  recorded shape (B={cb}, L={cl}); shapes exceeding it in some dim: {grew or '(none)'}")
    if not grew:
        print("  FAIL: no shape exceeds the recorded shape in any dim; these constants miss the probe's design goal")
        bad += 1

    # The whole point: the control group records one graph per (B,L), DynaGraph serves them all with one.
    print(f"\n  recordings {rec[False]} -> {rec[True]} ({len(SHAPES)} shapes)")
    if rec[True] != 0 or rec[False] != len(SHAPES):
        print("  FAIL: recording count mismatch: DynaGraph should record nothing, the control group once per shape")
        bad += 1

    print("\n  " + ("all passed" if not bad else f"{bad} checks failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
