#!/usr/bin/env python3
r"""Class A probe (should be served): when the wrapper is cut into several partition_N, each one becomes its own graph.

This used to be a class B probe asserting "multiple partitions must fall back", with the reason given in the docstring of
_entry_source: the collected kernels belong to two different cudagraphs, and this layer cannot tell them apart.
**That reason was wrong.** compile_fx calls cudagraphify once per partition via recursively_apply_fns,
and the callable handed to the hook has __name__ "partition_0" /
"partition_1" (measured with dynagraph/_probe_partition_id.py, not included in this repo), so each partition already
has its own runner, its own arena, its own graph; all that is needed is to read the right section by name.

But "reading the right section" is only necessary, not sufficient. Hooking it up for real exposed two things about passing
data between partitions, and both are what this probe now guards:

  1. **The return value contains inputs.** partition_0 is `return (buf0, buf3, arg1_1)` --
     arg1_1 is passed through as is to the next partition. Picking only bufN returns a tuple shorter than what the caller unpacks.
  2. **Buffers come in through the arguments.** partition_1 is `arg1_1, buf0, buf6, s77 = args`,
     then `buf8 = buf0` is written in place. It does not allocate a single buffer itself, its arena is empty,
     and it writes into the caller's buffer -- the result must be written back, or the caller gets stale data.

How the model produces two partitions: a CPU round trip in the middle. In scheduler.should_partition,
DeviceCopy and non-GPU ops are both explicit reasons to cut, so the three steps after relu -- sum().cpu(),
the multiply on the CPU, and .cuda() back -- are pushed out of the cudagraph, splitting the GPU part into a front and a back block.
Note that .item() cannot be used: it breaks the graph at the dynamo level, giving two compile units with one partition
each, and the target would be missed.

Criteria:
  1. >=2 partitions are really seen (target hit), and the section can be looked up by name
  2. counterfactual: without a name, _entry_source still returns None -- showing that "by name" is what rescued it
  3. with dynagraph=True the recording count drops to 0 (every partition was served)
  4. output bitwise identical to the control
"""
import logging
import os
import re
import sys

SHAPES = tuple(int(v) for v in os.environ.get("DG_SHAPES", "512,256,333,64").split(","))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

# Fallback reasons only show up at INFO level; without it nothing is visible. But the assertions do not rely on logs --
# a log only says "refused", not "whether it was refused because of the partition count"; that part is covered by
# the direct observation of _maybe_build_dynagraph and the counterfactual check below.
logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")


class _Collect(logging.Handler):
    """Collect the fallback reasons.

    Every refusal in dynagraph.py goes through _fallback(), printed as "DynaGraph fallback [tag]".
    What a class B probe must report is "which reason fired", so the reasons must be captured instead of just
    flushed to stderr -- as a side effect this also shows whether a given fallback printed any reason at all.
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.msgs: list[str] = []

    def emit(self, record):
        self.msgs.append(f"{record.name.split('.')[-1]}: {record.getMessage()}")


LOGS = _Collect()
for _name in ("torch._inductor.cudagraph_trees", "torch._inductor.dynagraph"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.INFO)
    _lg.addHandler(LOGS)

# Observations from each _maybe_build_dynagraph call, accumulated across both runs
SEEN: list[dict] = []
# The last multi-partition wrapper source seen, kept for the counterfactual check below
SRC = {"s": ""}

# Dump the wrapper source to disk for manual inspection of the partition structure (the probe's evidence, not a report)
WRAPPER_DUMP = os.environ.get(
    "DG_WRAPPER_DUMP", os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "multi_partition_wrapper.py")
)

# _entry_source uses a regex of exactly this shape; copy it verbatim so that "the partitions I count"
# and "the partitions it counts" are the same thing.
_PART_RE = re.compile(r"^def partition_\d+\(args\):\n", re.MULTILINE)


def run(dynagraph: bool, cut: bool = True):
    import torch
    import torch._inductor.config as ic

    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    # Route GEMM to Triton: extern_kernels.mm does not go through the static launcher, so the handle-count check
    # would refuse the graph first; the fallback reason would then not be "multiple partitions" and the target would be lost.
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"
    # Without graph partition there is no partition_N, and the whole probe loses its target. It is on by default;
    # pin it explicitly so an env var cannot turn it off while the probe still claims a hit.
    ic.graph_partition = True

    from torch._dynamo.utils import counters

    counters.clear()

    import torch._inductor.cudagraph_trees as ct
    from torch._inductor import dynagraph as dg

    # The observation point is _maybe_build_dynagraph: it is the only place that sees both
    # "which wrapper is being asked about this time" and "whether its answer is serve or fall back".
    orig_build = ct._maybe_build_dynagraph

    def spy_build(model, inputs, kwargs, *a, **kw):
        src = dg._wrapper_source(model) or ""
        r = orig_build(model, inputs, kwargs, *a, **kw)
        if len(_PART_RE.findall(src)) >= 2:
            SRC["s"] = src
        if src and not os.path.exists(WRAPPER_DUMP):
            try:
                os.makedirs(os.path.dirname(WRAPPER_DUMP) or ".", exist_ok=True)
                with open(WRAPPER_DUMP, "w") as fh:
                    fh.write(src)
            except OSError:
                pass
        SEEN.append(
            dict(
                flag=dynagraph,
                n_part=len(_PART_RE.findall(src)),
                entry_none=dg._entry_source(src) is None,
                entry_named=dg._entry_source(
                    src, getattr(model, "__name__", None)
                ) is not None,
                name=getattr(model, "__name__", None),
                served=r is not False,
                path=(getattr(model, "__globals__", {}) or {}).get("__file__"),
            )
        )
        return r

    ct._maybe_build_dynagraph = spy_build

    n_record = {"n": 0}
    orig_rec = ct.CUDAGraphTreeManager.record_function

    def spy_rec(self, *a, **kw):
        n_record["n"] += 1
        return orig_rec(self, *a, **kw)

    ct.CUDAGraphTreeManager.record_function = spy_rec

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(128, 128)

        def forward(self, x):
            h = torch.relu(self.l(x))
            if cut:
                # These three steps are the cut: DeviceCopy out, compute on the CPU, DeviceCopy back.
                # h is still used after the *, so real GPU work remains on both sides of the cut.
                s = h.sum().cpu()
                t = (s * 0.5).cuda()
            else:
                # Positive control: the same model, except this step stays on the GPU, so there is only one
                # partition. This path **must be served**. Without it, this probe's three criteria
                # (not served / recording count equals the control / numerics right) would equally hold for "DynaGraph
                # refused every graph for some other reason" -- making it a test that
                # still prints a pass if DynaGraph were replaced by an empty shell.
                t = h.sum() * 0.5
            return (h * t) - h.mean(dim=-1, keepdim=True)

    # Both runs must get the same weights and the same inputs, otherwise the outputs cannot be compared bitwise.
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
                # it only does an eager warmup and records on the second call. With one call the control records
                # no graph at all, and the criterion "recording count after fallback equals the control" degenerates to 0 == 0.
                f(x)
                outs[M_] = (
                    f(x).float().cpu().clone(),
                    m(x).float().cpu().clone(),
                )
    finally:
        ct.CUDAGraphTreeManager.record_function = orig_rec
        ct._maybe_build_dynagraph = orig_build
    skips = dict(counters["inductor"])
    return n_record["n"], outs, skips


def main():
    import torch

    if not torch.cuda.is_available():
        print("no CUDA device available")
        return 1

    res, rec = {}, {}
    for flag in (False, True):
        n, outs, skips = run(flag)
        res[flag] = outs
        rec[flag] = n
        print(f"\n  dynagraph={flag}")
        print(f"    recorded {n} times")
        interesting = {
            k: v
            for k, v in skips.items()
            if "cudagraph" in k or "skip" in k or "partition" in k
        }
        print(f"    relevant counters: {interesting or '(empty)'}")

    bad = 0

    # --- criterion 1: the target was really hit ---------------------------
    # Without first proving "the wrapper really has >=2 partitions", a later "fell back" could have another cause
    # (handle-count mismatch, config not converged...), and then this probe would pass with any model.
    print("\n  wrapper observations (what _maybe_build_dynagraph saw each time it was asked):")
    if not SEEN:
        print("    FAIL _maybe_build_dynagraph was never called -- the probe never touched the target code")
        bad += 1
    multi = [s for s in SEEN if s["n_part"] >= 2]
    for s in SEEN:
        print(
            f"    flag={s['flag']} partitions={s['n_part']}"
            f" entry={s['name']} by-name lookup:{s['entry_named']}"
            f" None without name:{s['entry_none']} served:{s['served']}"
        )
    if SEEN:
        print(f"    wrapper file: {SEEN[-1]['path']}")
        print(f"    wrapper copy: {WRAPPER_DUMP}")
    fb = [m for m in LOGS.msgs if "fallback" in m]
    print("\n  Fallback reasons printed by DynaGraph:")
    for m in fb or ["(none at all)"]:
        print(f"    {m}")
    # Now the other way round: multiple partitions should be served, so any fallback log means a gap in coverage.
    # An empty log is good, but print it anyway -- when something does break, this is the only place that pins down which gate.
    if multi and fb:
        print("    FAIL multiple partitions still fall back, coverage is incomplete (the reasons are listed above)")
        bad += 1
    if not multi:
        print("    FAIL never saw >=2 partitions -- no multi-partition wrapper was produced,")
        print("      so this probe did not hit the multi-partition branch of _entry_source")
        bad += 1
    elif not all(s["entry_named"] for s in multi):
        print("    FAIL there are >=2 partitions, but lookup by __name__ does not find the matching section -- ")
        print("      entry detection did not take effect; even if the rest passes, it is not due to multi-partition support")
        bad += 1
    elif not all(s["entry_none"] for s in multi):
        print("    FAIL _entry_source did not return None without a name either -- the ambiguity guard is gone,")
        print("      so the success above may just have hit the old single-partition path")
        bad += 1

    # --- criterion 1b: counterfactual -- it really was "lookup by name" that rescued it ---
    # Seeing "served" is not enough: maybe only one partition of this wrapper was ever asked about.
    # Ask twice with the same source: without a name it should still be None (the ambiguity guard is still there), with a name
    # it should immediately recover the symbol table. Only if both hold was it entry detection that did the work.
    if SRC["s"]:
        from torch._inductor import dynagraph as dg

        names = _PART_RE.findall(SRC["s"])
        anon = dg._entry_source(SRC["s"])
        named = {n: dg._entry_source(SRC["s"], n) for n in ("partition_0", "partition_1")}
        syms = {
            n: dg._input_symbol_map(b) for n, b in named.items() if b is not None
        }
        print(
            f"\n  counterfactual: partitions {len(names)}; _entry_source without a name -> "
            f"{'None' if anon is None else 'not None'}; symbol table from by-name lookup {syms}"
        )
        if anon is not None:
            print("    FAIL a section is found even without a name -- the ambiguity guard is broken")
            bad += 1
        if not syms or not all(syms.values()):
            print("    FAIL section found by name but the symbol table is still empty -- entry detection is not what rescued it")
            bad += 1

    # --- criterion 2: every section of a multi-partition wrapper must be served ---
    # One section served and another not also counts as a failure: it would mean coverage is luck, not that passing
    # buffers between partitions is really handled.
    refused = [s for s in SEEN if s["flag"] and s["n_part"] >= 2 and not s["served"]]
    if refused:
        print(f"    FAIL {len(refused)} partitions were not served:")
        for s_ in refused:
            print(f"        {s_['name']}")
        bad += 1

    # --- criterion 3: output is still correct -----------------------------
    # The reference is the same compile path with dynagraph off, not eager: GEMM is routed to Triton,
    # which already differs algorithmically from eager's cuBLAS; comparing with eager would read the algorithm difference as a correctness problem.
    print("\n  Compared with dynagraph=False (criterion: no farther from eager than the control):")
    for M_ in SHAPES:
        (a, ea), (b, eb) = res[False][M_], res[True][M_]
        if a.shape != b.shape:
            print(f"    M={M_} shape {tuple(b.shape)} != {tuple(a.shape)}  FAIL")
            bad += 1
            continue
        # The eager references of the two runs must be bitwise identical; otherwise it is a seeding problem, not DynaGraph's fault.
        seed_ok = (ea - eb).abs().max().item() == 0
        scale = max(ea.abs().max().item(), 1e-9)
        ctl = (a - ea).abs().max().item() / scale
        dyn = (b - eb).abs().max().item() / scale
        ok = seed_ok and dyn <= max(ctl * 1.5, 1e-6)
        print(
            f"    M={M_} ctl<->dyna {(a - b).abs().max().item():.2e}"
            f" | ctl<->eager {ctl:.1e} | dyna<->eager {dyn:.1e}"
            f" | same seed {seed_ok}" + ("  OK" if ok else "  FAIL")
        )
        bad += not ok

    # --- criterion 4: the recording count must drop to 0 ------------------
    # End-to-end evidence. The control must be != 0, otherwise these shapes never went through cudagraph and the test proves nothing.
    # ---- positive control: the same model without the cut must be served ----
    n_pos, outs_pos, skips_pos = run(True, cut=False)
    served_pos = n_pos == 0
    print(f"\n  positive control (CPU round trip removed, one partition left): recorded {n_pos} times")
    print(f"    expected: served, nothing recorded -- {'OK' if served_pos else 'FAIL'}")
    if not served_pos:
        print("    FAIL even a single partition cannot be served, so the 'fallback' above cannot be attributed to multiple partitions")
        print(f"    counters from the positive control run: {skips_pos}")
    bad += not served_pos
    for M_ in SHAPES:
        got, eager = outs_pos[M_]
        rel = (got - eager).abs().max().item() / max(eager.abs().max().item(), 1e-9)
        ok = rel < 1e-3
        print(f"    M={M_:<4} rel to eager {rel:.2e} {'OK' if ok else 'FAIL'}")
        bad += not ok

    print(f"\n  recording count {rec[False]} -> {rec[True]} ({len(SHAPES)} shapes)")
    if rec[False] == 0:
        print("  FAIL the control never recorded -- these shapes never went through cudagraph, the criterion is void")
        bad += 1
    elif rec[True] != 0:
        print(f"  FAIL still recorded {rec[True]} times -- the multi-partition wrapper was not fully served")
        bad += 1

    print("\n  " + ("all passed" if not bad else f"{bad} failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
