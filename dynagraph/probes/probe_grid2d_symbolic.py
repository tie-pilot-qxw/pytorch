#!/usr/bin/env python3
"""Does the planner get it right when the **y range of a Grid2D is also driven by a symbol**?

Why another probe: probe_grid2d.py does make Inductor emit grid_type=Grid2D,
but its ynumel is a static 64, so what gets generated is `gy = ceil(64/YBLOCK)` -- a constant.
In other words, the y row newly added to `_GRID_AXES` has **never been driven by a symbol**:
even if the `gy` formula were wrong, as long as it always evaluates to the recording-time value, nothing would show.

Here both dims are symbolic: x has shape (M, N), y has shape (N, M), and the transpose in `x + y.t()`
forces Inductor into 2D tiling, with M and N varying independently.

Criteria (all are required):
  1. **grid_type really is Grid2D/Grid3D**, read from the kernel table dumped to disk;
  2. **the gy expression really references ctx** (i.e. `S(i)`), not a constant --
     this is exactly what probe_grid2d lacks; without it the first check is only a "nominal hit";
  3. the control group records one graph per shape, dynagraph=True **records none**
     (without this, "numbers match" would still hold if DynaGraph degenerated into always falling back);
  4. every shape's output is bit-identical to the control group.

Shapes go in descending element count: the static input buffer only keeps headroom-times slack, and ascending
order would first hit the `input-too-large` retirement, which is a different path (tested separately by test_verify.py).

No timing; runs on a shared card.
"""
from __future__ import annotations

import logging
import os
import re
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

# (M, N), descending by M*N. The two dims vary independently, including a pair where "one shrinks while the other grows".
SHAPES = ((256, 192), (256, 64), (64, 192), (32, 320), (96, 48))

_OUT = os.environ.get("DG_OUT", "/tmp/dynagraph_out")
os.makedirs(_OUT, exist_ok=True)
DUMP = os.path.join(_OUT, "grid2d_symbolic_planner.cu")


def run(dynagraph: bool):
    import torch
    import torch._inductor.config as ic
    import torch._inductor.cudagraph_trees as ct

    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"

    msgs = []

    class Grab(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    grab = Grab()
    lg = logging.getLogger("torch._inductor.dynagraph")
    lg.setLevel(logging.INFO)
    lg.addHandler(grab)

    n_record = {"v": 0}
    orig = ct.CUDAGraphTreeManager.record_function

    def spy(self, *a, **kw):
        n_record["v"] += 1
        return orig(self, *a, **kw)

    ct.CUDAGraphTreeManager.record_function = spy

    class M(torch.nn.Module):
        def forward(self, a, b):
            # The transpose forces 2D tiling: if the elementwise op could be flattened to 1D, Inductor would
            # give Grid1D, and the y row would be missed again.
            return (a + b.t()).relu() * 2.0

    m = M().cuda().eval()
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")

    outs = {}
    try:
        torch.manual_seed(1)
        for M_, N_ in SHAPES:
            a = torch.randn(M_, N_, device="cuda")
            b = torch.randn(N_, M_, device="cuda")
            with torch.no_grad():
                # Twice: the first time cudagraph_trees sees a FunctionID it only does an eager
                # warmup, and it records for real on the second call. With a single call the control group records
                # no graph, and the "recordings 5 -> 0" criterion degenerates into 0 == 0.
                f(a, b)
                outs[(M_, N_)] = (
                    f(a, b).float().cpu().clone(),
                    m(a, b).float().cpu().clone(),
                )
    finally:
        ct.CUDAGraphTreeManager.record_function = orig
        lg.removeHandler(grab)

    tags = [t.split("[", 1)[1].split("]", 1)[0]
            for t in msgs if t.startswith("DynaGraph fallback [")]
    return n_record["v"], outs, tags


def inspect_dump() -> tuple[bool, bool, str]:
    """(is there 2D/3D tiling, is the y range symbol-driven, explanation).

    Match the full set of names: what Inductor actually emits is `Grid2DWithYZOverflow` -- when the number of
    blocks along y exceeds 65535 it folds into z. That is the common form of 2D tiling; plain `Grid2D` is actually rare.

    "y is symbol-driven" also cannot just look for `gy=dg_eval(...)`: in the folded form y goes through `raw`
    (`raw = ceil(ynumel/YBLOCK)`, then folded by `div`), so follow every dg_eval that appears in a case body
    back to its expression. Without this, the previous check is only a nominal hit --
    a wrong formula would not show when gy always equals the recording-time value.
    """
    if not os.path.exists(DUMP):
        return False, False, "planner was not dumped to disk"
    src = open(DUMP).read()
    types = re.findall(r"'grid_type':\s*'(\w+)'", src)
    has2d = any("Grid2D" in t or "Grid3D" in t for t in types)

    def expr_of(i: str) -> str:
        m = re.search(rf"case {i}:\s*return\s*(.+?);", src)
        return m.group(1).strip() if m else ""

    sym_y, detail = False, []
    if os.environ.get("TORCHINDUCTOR_DYNAGRAPH_UPDATE") == "host":
        # The host-path patcher inlines the expressions directly into the grid assignment (`S(i)` is a symbol),
        # with no dg_eval table lookup; just check whether that line itself references a symbol.
        for line in src.splitlines():
            if re.search(r"\b(raw|gy)\s*=\s*dg_floordiv\(", line) and "S(" in line:
                sym_y = True
                detail.append(line.strip()[:90])
        return (has2d, sym_y,
                f"grid_type={types} | " + ("; ".join(detail) or "y range does not reference a symbol"))
    for line in src.splitlines():
        # Only look at the case bodies of the grid switch: they are the only place gx/gy/gz are written.
        if "gy" not in line and "raw" not in line:
            continue
        # Folded form: raw is computed from ynumel; plain form: gy is computed directly from ynumel.
        for kind, pat in (("raw", r"raw\s*=\s*dg_floordiv\(dg_eval\((\d+),ctx\)"),
                          ("gy", r"gy\s*=\s*dg_floordiv\(dg_eval\((\d+),ctx\)")):
            for i in re.findall(pat, line):
                e = expr_of(i)
                if "S(" in e:
                    sym_y = True
                    detail.append(f"{kind} uses dg_eval({i}) = {e}")
    return (has2d, sym_y,
            f"grid_type={types} | " + ("; ".join(detail) or "y range does not reference ctx"))


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("no CUDA device available")
        return 1

    # If the dump is not deleted we might read stale evidence from a previous run, and the "hit" would be fake.
    if os.path.exists(DUMP):
        os.remove(DUMP)

    res, rec, tg = {}, {}, {}
    for flag in (False, True):
        if flag:
            os.environ["TORCHINDUCTOR_DYNAGRAPH_DUMP"] = DUMP
        n, outs, tags = run(flag)
        res[flag], rec[flag], tg[flag] = outs, n, tags
        print(f"\n  dynagraph={flag}  recordings {n}  fallback tags {tags or '(none)'}")
    os.environ.pop("TORCHINDUCTOR_DYNAGRAPH_DUMP", None)

    bad = 0
    has2d, sym_gy, detail = inspect_dump()
    print(f"\n  evidence: {detail}")
    print(f"    hit 2D/3D tiling -- {'OK' if has2d else 'FAIL missed the target'}")
    print(f"    y range driven by a symbol -- {'OK' if sym_gy else 'FAIL y is constant, that path is still unverified'}")
    bad += (not has2d) + (not sym_gy)

    ok = rec[False] == len(SHAPES) and rec[True] == 0
    print(f"\n  recordings {rec[False]} -> {rec[True]} ({len(SHAPES)} shapes)"
          f" -- {'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"    fallback tags: {tg[True] or '(none)'}")
    bad += not ok

    print("\n  bitwise comparison against the control group:")
    for key in SHAPES:
        a, ea = res[False][key]
        b, eb = res[True][key]
        if a.shape != b.shape:
            print(f"    {key} shape {tuple(b.shape)} != {tuple(a.shape)}  FAIL")
            bad += 1
            continue
        d = (a - b).abs().max().item()
        scale = max(ea.abs().max().item(), 1e-9)
        print(f"    M={key[0]:<4} N={key[1]:<4} ctl<->dyna {d:.2e}"
              f" | ctl<->eager {(a - ea).abs().max().item() / scale:.1e}"
              f"  {'OK' if d == 0 else 'FAIL'}")
        bad += d != 0

    print("\n  " + ("all passed" if not bad else f"{bad} checks failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
