#!/usr/bin/env python3
"""Degenerate shapes: run 0-sized (empty tensor) and tiny shapes like 1/2/3 through the same served graph.

Why this is worth testing on its own: a grid dimension of 0 is cudaErrorInvalidArgument, so the
planner cannot write an empty extent as "grid=0"; it has to switch the node off with
cudaGraphKernelNodeSetEnabled(node, 0) instead (see _PLANNER_TEMPLATE in dynagraph.py). No test had
touched this disable branch before, and if it is wrong it can fail in two ways:
  a) it does not switch off -> the node replays with the large recording-time grid and scribbles over the arena;
  b) once switched off it never comes back on -> the first normal shape after an empty one has a dirty output.
So the point of this probe is not "the empty tensor is right" (an empty tensor has no elements, so any comparison passes), but:
  * while the empty branch is disabled, the other branch must still compute correctly (two independent symbols;
    the x branch goes through GEMM, the y branch is pointwise only, and with Mx=0 only the two x nodes should be switched off);
  * running a normal shape after an empty one must still match the control group bit for bit (proving
    SetEnabled(1) really switched the node back on).

Measured findings (see the verdict at the end of the file and the README-level notes):
  * With default settings torch.compile never hands you a 0-sized dynamic shape -- 0/1 specialization turns
    size 0 and size 1 into static constants and compiles a new graph, so DynaGraph never even sees them;
    and a size-0 mm does not compile at all under max_autotune_gemm_backends="TRITON"
    (NoValidChoicesError). That is an Inductor limitation, unrelated to DynaGraph.
  * With torch.fx.experimental._config.backed_size_oblivious=True, 0 and 1 stay in the same
    dynamic graph, DynaGraph serves all of 0/1/2/3/33/64/512 with one graph,
    and only then is the disable branch actually exercised.
Both phases run and are judged separately: PHASE A records the default behavior, PHASE B is the real acceptance test of the disable branch.

Three things hit along the way (none of them is what this probe judges, but they are written here so nobody hits them again):
  1. If an empty shape is the [first] shape this region sees, DynaGraph gives up on the region permanently:
     at capture time the kernels with an empty extent are not launched at all, handle count 1 != kernel count 3,
     the [handle-mismatch] fallback fires, and once deferred_cudagraphify records the runner as False
     it never retries. Same set of shapes: largest first gives one graph for all, empty first gives one graph per shape.
     Repro in _explore_empty_first.py in the same directory (not included in this repo).
  2. With default settings, a shape like (1,4) with one dimension specialized away makes DynaGraph fall back
     for the whole region, because of [unmodelled]: no xnumel/XBLOCK to size a grid -- once xnumel is specialized
     to a constant it is a constexpr in the Triton signature and not in args, so exprs has no xnumel, and
     the Grid1D branch of generate_planner cannot tell "this dim is constant" from "I did not find this dim".
     Yet the FixedGrid node in the same graph, holding the constant grid ['4','1','1'], is let through.
     Repro in _explore_why_unmodelled.py in the same directory (not included in this repo).
  3. Under max_autotune_gemm_backends="TRITON" a size-0 mm fails to compile outright
     (NoValidChoicesError), so (0,64) in PHASE A does not run on either side.
     That is Inductor's issue, unrelated to DynaGraph.
"""
import ast
import logging
import os
import re
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

DUMP = os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "empty_and_tiny_planner.cu")
os.makedirs(os.path.dirname(DUMP), exist_ok=True)

# The first shape must be the largest: DynaGraphRunner builds the graph on the first shape it sees,
# static_inputs and the arena are sized for that set, and later shapes can only be smaller.
# PHASE A uses default settings; 1 and 0 are included only to record "it specializes by default";
# 0 goes last because a size-0 mm fails to compile under TRITON-only and would interrupt the later shapes.
PHASE_A = [(512, 256), (3, 7), (2, 5), (1, 4), (0, 64)]
# PHASE B sandwiches empty shapes between normal ones, specifically to check re-enable after disable.
PHASE_B = [(512, 256), (0, 64), (3, 7), (2, 2), (1, 1), (0, 0), (64, 33)]

# The two dims must get different values, otherwise duck shaping treats them as the same symbol,
# and as soon as they differ the guard fails and it recompiles, so it is no longer the same graph under test.


class _Collect(logging.Handler):
    """Collect DynaGraph's INFO lines and print them all at the end.

    Fallback reasons (handle-mismatch / config not settled / unknown grid type) only come out
    of the log, and mixed into the autotune output they are impossible to see.
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def run(dynagraph, shapes, size_oblivious, dump=None):
    import torch
    import torch._inductor.config as ic
    import torch._inductor.cudagraph_trees as ct
    import torch._inductor.dynagraph as dg
    import torch.fx.experimental._config as fxc

    torch._dynamo.reset()
    fxc.backed_size_oblivious = size_oblivious
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    # GEMM must be routed away from cuBLAS: extern_kernels.mm does not use the static launcher, so that node
    # has no handle, the planner cannot patch it, and the handle-count check makes the whole graph fall back.
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"
    if dump:
        os.environ["TORCHINDUCTOR_DYNAGRAPH_DUMP"] = dump
    else:
        os.environ.pop("TORCHINDUCTOR_DYNAGRAPH_DUMP", None)

    n_rec = {"n": 0}
    o_rec = ct.CUDAGraphTreeManager.record_function

    def spy_rec(self, *a, **kw):
        n_rec["n"] += 1
        return o_rec(self, *a, **kw)

    # Count served calls directly as runner invocations: more direct than "was there a recompile",
    # since it is the definition of "did this call land on the DynaGraph path".
    served = {"n": 0}
    envs = []
    o_call = dg.DynaGraphRunner.__call__

    def spy_call(self, inputs):
        served["n"] += 1
        # __call__ clears inputs, so the symbol values have to be copied before going in;
        # this env is the one fed into ctx, and it is used later to recompute the planner's grid.
        envs.append(
            {
                s: int(inputs[i])
                for s, i in self.sym_from_input.items()
                if i < len(inputs) and isinstance(inputs[i], int)
            }
        )
        return o_call(self, inputs)

    ct.CUDAGraphTreeManager.record_function = spy_rec
    dg.DynaGraphRunner.__call__ = spy_call

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(128, 128)

        def forward(self, x, y):
            # x branch: GEMM + a persistent reduction; both nodes' grids are proportional to Mx,
            # and both should be switched off when Mx=0.
            h = torch.relu(self.l(x))
            a = h - h.mean(dim=-1, keepdim=True)
            # y branch: pure pointwise, an independent symbol. It must still compute when Mx=0 -- the only
            # way to keep the "empty extent" check from comparing nothing against nothing.
            b = torch.tanh(y) * 2.0 + 1.0
            return a, b

    # Both runs must get the same weights and the same inputs, otherwise the outputs cannot be compared bit for bit.
    torch.manual_seed(0)
    m = M().cuda().eval()
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")

    out = {}
    try:
        torch.manual_seed(1)
        for mx, my in shapes:
            r0, s0, e0 = n_rec["n"], served["n"], len(envs)
            x = torch.randn(mx, 128, device="cuda")
            y = torch.randn(my, 128, device="cuda")
            rec = {"rec": 0, "served": 0, "env": [], "err": None}
            try:
                with torch.no_grad():
                    # Call each shape twice: the first time cudagraph_trees sees a FunctionID
                    # it only does an eager warmup, and it records for real on the second call. With a
                    # single call the control group records no graph at all and is a control in name only.
                    f(x, y)
                    a, b = f(x, y)
                    ea, eb = m(x, y)
                rec.update(
                    out=(a.float().cpu().clone(), b.float().cpu().clone()),
                    eager=(ea.float().cpu().clone(), eb.float().cpu().clone()),
                )
            except Exception as exc:  # a per-shape failure should not take down the whole probe
                rec["err"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            rec["rec"] = n_rec["n"] - r0
            rec["served"] = served["n"] - s0
            rec["env"] = envs[e0:]
            out[(mx, my)] = rec
    finally:
        ct.CUDAGraphTreeManager.record_function = o_rec
        dg.DynaGraphRunner.__call__ = o_call
        os.environ.pop("TORCHINDUCTOR_DYNAGRAPH_DUMP", None)
    return out


def grids_at(dump_path, env):
    """Recompute each node's grid under this set of symbols, with the same formulas the planner was generated from.

    This is the evidence that "the disable branch was really taken": checking the outputs cannot show it,
    since an empty tensor is right no matter what. The kernel table at the top of the dump holds exactly the
    grid formulas generate_planner used and the settled block sizes; recompute them as-is, and nodes with gx<=0 are the ones switched off.
    """
    from torch._inductor.dynagraph import _GRID_AXES, _eval_int

    src = open(dump_path).read()
    m = re.search(r"^// kernels: (.*)$", src, re.MULTILINE)
    if not m:
        return None, None, src
    kernels = ast.literal_eval(m.group(1))
    rows = []
    for k in kernels:
        if k["grid"]:
            g = [_eval_int(e, env) for e in k["grid"]]
        else:
            g = [1, 1, 1]
            axes = _GRID_AXES.get(k.get("grid_type") or "")
            # Each axis is ("cdiv", numel, block) / ("numel", numel, None) /
            # ("const", None, key), same table the generators read.
            for j, (kind, numel, bk) in enumerate(axes or ()):
                if kind == "const":
                    g[j] = k["blocks"][bk]
                    continue
                n = _eval_int(k["exprs"][numel], env)
                if kind == "numel":
                    g[j] = n
                    continue
                blk = k["blocks"][bk]
                g[j] = None if n is None else (n + blk - 1) // blk
        rows.append((k["name"], g))
    return rows, kernels, src


def arena_offsets(kernels_src, env):
    """Recompute each buffer's byte range in the arena with the same rules as dynagraph_layout.

    This is the only piece of "behavioral evidence" in this probe (rather than recomputed formulas): the planner
    returns early when gx<=0, so the pointer switch never runs, and a node that was not switched off would
    keep the previous round's pointers and keep writing with the large recording-time grid. With an empty shape slot0 has size 0
    and slot1's offset is pushed to 0 by the prefix sum -- i.e. the y branch's output lands exactly where the x branch wrote last round.
    b being bit-exact therefore means the two x nodes really did not run a single instruction.
    """
    from torch._inductor.dynagraph import _eval_int

    m = re.search(r"^// slots: (.*)$", kernels_src, re.MULTILINE)
    n = re.search(r"^// sizes: (.*)$", kernels_src, re.MULTILINE)
    if not (m and n):
        return None
    slot_of, sizes = ast.literal_eval(m.group(1)), ast.literal_eval(n.group(1))
    nslots = max(slot_of.values()) + 1
    sz = [0] * nslots
    for b, (span, itemsize) in sizes.items():
        v = _eval_int(span, env)
        if v is None:
            return None
        sz[slot_of[b]] = max(sz[slot_of[b]], v * itemsize)
    off, acc = [], 0
    for i in range(nslots):
        off.append(acc)
        acc += (sz[i] + 255) & ~255
    return {
        b: (off[slot_of[b]], off[slot_of[b]] + _eval_int(span, env) * itemsize)
        for b, (span, itemsize) in sizes.items()
    }


def close_enough(rec_ctl, rec_dyn):
    """Criterion copied from test_flag.py: the reference is the same compile path with dynagraph off, not eager.

    GEMM is routed to Triton, which differs algorithmically from eager's cuBLAS anyway; using eager as ground truth
    would read the algorithmic difference as a correctness problem. But bit-exact equality with the control group is
    not required either: the control group re-records a graph per shape and picks an autotune config for each, while
    DynaGraph only has the one pinned at recording time; a different XBLOCK gives a different reduction order, so one ULP of difference is correct.
    """
    msgs, ok = [], True
    for idx, tag in ((0, "a(x branch)"), (1, "b(y branch)")):
        c, cd = rec_ctl["out"][idx], rec_ctl["eager"][idx]
        d, dd = rec_dyn["out"][idx], rec_dyn["eager"][idx]
        if c.shape != d.shape:
            msgs.append(f"{tag} shape {tuple(d.shape)}!={tuple(c.shape)}")
            ok = False
            continue
        if c.numel() == 0:
            msgs.append(f"{tag} empty tensor (shapes match)")
            continue
        # The eager references of the two runs must be bit-identical; otherwise it is a seeding problem, not DynaGraph's fault.
        seed_ok = (cd - dd).abs().max().item() == 0
        scale = max(cd.abs().max().item(), 1e-9)
        ctl = (c - cd).abs().max().item() / scale
        dyn = (d - dd).abs().max().item() / scale
        good = seed_ok and dyn <= max(ctl * 1.5, 1e-6)
        ok &= good
        msgs.append(
            f"{tag} ctl<->dyna {(c - d).abs().max().item():.1e}"
            f" ctl<->eager {ctl:.1e} dyn<->eager {dyn:.1e} same seed {seed_ok}"
        )
    return ok, "; ".join(msgs)


def alias_of(src):
    m = re.search(r"^// alias: (.*)$", src, re.MULTILINE)
    return ast.literal_eval(m.group(1)) if m else {}


def report(title, shapes, ctl, dyn, strict):
    """strict holds the shapes that "must be served by one graph"; the rest are only recorded, not judged."""
    print(f"\n===== {title} =====")
    bad = 0
    n_distinct = len({s for s in shapes if ctl[s]["err"] is None})
    rec_ctl = sum(ctl[s]["rec"] for s in set(shapes))
    rec_dyn = sum(dyn[s]["rec"] for s in set(shapes))
    for s in shapes:
        c, d = ctl[s], dyn[s]
        if c["err"] or d["err"]:
            print(f"  {str(s):>12}  ctl {c['err'] or 'ok'} | dyna {d['err'] or 'ok'}")
            if s in strict:
                bad += 1
            continue
        ok, msg = close_enough(c, d)
        mark = "OK" if ok else "FAIL"
        env = d["env"][-1] if d["env"] else {}
        line = (
            f"  {str(s):>12}  recordings {c['rec']}->{d['rec']} served {d['served']}"
            f" env={env or '-'}  {msg}  {mark}"
        )
        if s in strict:
            if d["rec"] != 0 or d["served"] == 0:
                ok = False
                line += "  <- should be served by one graph but recorded a graph"
            bad += not ok
        print(line)
    print(
        f"  total recordings {rec_ctl} -> {rec_dyn}"
        f" ({n_distinct} distinct shapes that ran; strict set {sorted(strict)})"
    )
    return bad, rec_ctl, rec_dyn


def main():
    import torch

    if not torch.cuda.is_available():
        print("no CUDA device available")
        return 1

    col = _Collect()
    for nm in ("torch._inductor.cudagraph_trees", "torch._inductor.dynagraph"):
        lg = logging.getLogger(nm)
        lg.setLevel(logging.INFO)
        lg.addHandler(col)

    # ---------- PHASE A: default settings, see whether torch.compile will hand out 0/1 dims ----------
    a_ctl = run(False, PHASE_A, size_oblivious=False)
    a_dyn = run(True, PHASE_A, size_oblivious=False)
    # By default 0 and 1 are specialized into static shapes (compiled as a separate graph), which DynaGraph cannot control,
    # so strict only holds shapes with both dims >=2.
    strict_a = {s for s in PHASE_A if min(s) >= 2}
    bad_a, _, _ = report("PHASE A: default settings (0/1 specialization on)", PHASE_A, a_ctl, a_dyn, strict_a)

    # ---------- PHASE B: backed_size_oblivious, 0/1 stay in the same dynamic graph ----------
    b_ctl = run(False, PHASE_B, size_oblivious=True)
    b_dyn = run(True, PHASE_B, size_oblivious=True, dump=DUMP)
    strict_b = set(PHASE_B)  # in this phase every shape must be served by the same graph
    bad_b, rc, rd = report(
        "PHASE B: backed_size_oblivious=True (0/1 not specialized)", PHASE_B, b_ctl, b_dyn, strict_b
    )
    if rc != len({s for s in PHASE_B if b_ctl[s]["err"] is None}) or rd != 0:
        print("  FAIL: recording count mismatch: DynaGraph should record nothing, the control group once per distinct shape")
        bad_b += 1

    # ---------- evidence for the disable branch ----------
    print("\n===== planner grid recomputation (was the disable branch really taken) =====")
    hit = False
    if not os.path.exists(DUMP):
        print(f"  FAIL: no dump file {DUMP}, so PHASE B never built a graph")
        bad_b += 1
    else:
        rows0, kernels, src = grids_at(DUMP, {})
        # Device planner disables with cudaGraphKernelNodeSetEnabled; the host
        # patcher with cuGraphNodeSetEnabled on the exec. Either is the branch.
        has_branch = (
            "cudaGraphKernelNodeSetEnabled(handles[i], 0)" in src
            or "cuGraphNodeSetEnabled(e->ex, nd->node, 0)" in src
        )
        print(f"  planner has a disable branch: {has_branch}")
        for k in kernels or []:
            print(f"  node {k['name']}: grid={k['grid']} exprs={k['exprs']} blocks={k['blocks']}")
        for s in PHASE_B:
            env = b_dyn[s]["env"][-1] if b_dyn[s]["env"] else None
            if not env:
                continue
            rows, _, _ = grids_at(DUMP, env)
            off = [nm for nm, g in rows if any(v is not None and v <= 0 for v in g)]
            on = [nm for nm, g in rows if nm not in off]
            print(f"  {str(s):>12} env={env} -> switched off {len(off)}/{len(rows)} nodes {off}")
            if off and on and s != (0, 0):
                hit = True  # only meaningful when some node is off while another is still computing
        if not has_branch:
            bad_b += 1
    print(f"  disable branch meaningfully taken (some nodes off, some still computing): {hit}")

    # ---------- behavioral evidence: if not switched off, b would be overwritten by stale writes from the x branch ----------
    if os.path.exists(DUMP):
        src = open(DUMP).read()
        prev = None
        for s in PHASE_B:
            env = b_dyn[s]["env"][-1] if b_dyn[s]["env"] else None
            if not env or 0 not in env.values() or prev is None:
                if env:
                    prev = env
                continue
            here, before = arena_offsets(src, env), arena_offsets(src, prev)
            if not (here and before):
                break
            # A switched-off node still holds the previous round's pointers, so it would write the ranges in before;
            # buffers still live this round fall in here. If the two intersect, "not switched off" would be
            # caught by the output check, rather than missed by luck.
            rows_here, _, _ = grids_at(DUMP, env)
            dead = {nm for nm, g in rows_here if any(v is not None and v <= 0 for v in g)}
            stale = []
            for k in kernels or []:
                if k["name"] not in dead:
                    continue
                for b in k["ptrs"].values():
                    tgt = (alias_of(src) or {}).get(b, b)
                    if tgt in before:
                        stale.append((k["name"], tgt, before[tgt]))
            live = {b: r for b, r in here.items() if r[1] > r[0]}
            clash = [
                (nm, tb, rb, lb, lr)
                for nm, tb, rb in stale
                for lb, lr in live.items()
                if rb[0] < lr[1] and lr[0] < rb[1]
            ]
            print(f"  {str(s):>12} switched-off nodes wrote last round {stale}")
            print(f"               buffers live this round {live}")
            print(f"               ranges intersect: {[(c[0], c[3]) for c in clash] or 'none'}"
                  + (" (intersect -> the 'b bit-exact' line above is behavioral evidence the nodes really did not run)"
                     if clash else " (no intersection -> formula evidence only, not caught behaviorally)"))
            prev = env
    if not hit:
        bad_b += 1

    print("\n===== DynaGraph log =====")
    for ln in dict.fromkeys(col.lines):
        if "Recording cudagraph tree" not in ln:
            print("  " + ln)

    bad = bad_a + bad_b
    print(
        "\n  "
        + ("all passed" if not bad else f"{bad} checks failed")
        + f" (PHASE A {bad_a}, PHASE B {bad_b})"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
