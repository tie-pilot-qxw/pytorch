#!/usr/bin/env python3
"""probe_grid2d.py -- exercise the 2D tiled pointwise path (Inductor's Grid2D).

The _GRID_AXES table in dynagraph.py just grew two rows, Grid2D/Grid3D, but before this nothing
had ever run them: every graph verified so far had only Grid1D (plain pointwise/reduction) and
FixedGrid (mm templates). The sole purpose of this probe is to make Inductor actually emit a kernel whose grid_type is
Grid2D, and to show that DynaGraph computes it correctly and does not fall back the whole graph because of it.

Criteria (KIND A, "should be served"):
  1. the control group (dynagraph=False) records one graph per shape, dynagraph=True records none;
  2. the dynagraph=True outputs match the control group on every shape;
  3. **the graph really contains a Grid2D/Grid3D kernel** -- without this, passing the first two
     only means Grid1D ran once more, which is meaningless for this goal. There are two independent sources of proof:
       a. hook CachingAutotuner.run and read the inductor_meta of the kernels actually launched;
       b. the planner written out by TORCHINDUCTOR_DYNAGRAPH_DUMP: its header comment is the
          kernel table DynaGraph extracted itself, including the grid_type field.
     b is evidence from DynaGraph's point of view: it shows the Grid2D kernel's grid was computed by
     _GRID_AXES, rather than bypassed by another branch (e.g. FixedGrid reading grid_0/1/2 directly).

Why the model looks like this (both points are forced, not arbitrary choices):
  * The transpose must **not** follow a GEMM output. max_autotune's Triton mm template would absorb the following
    pointwise into the template kernel as an epilogue; that kernel is FixedGrid, and the 2D tile
    is gone. So the whole graph here is pure pointwise; the GEMM config is set per the contract but goes unused.
  * The y-axis numel must be **static**. dynamic=True marks every input dimension as dynamic, and
    whenever Inductor cannot prove y_grid <= 65535 (see triton.py:needs_yz_grid_overflow)
    it switches to Grid2DWithYZOverflow -- that grid formula has y/z overflow folding, _GRID_AXES
    does not have it, and DynaGraph (correctly) falls back the whole graph. Measured: a transpose with both dims dynamic always gets
    Grid2DWithYZOverflow. Here a (64,) parameter is broadcast to pin the input's second dim
    to 64, so the y axis can be static, leaving the x axis for the dynamic M.

Does the probe have teeth (verified, not inferred): break torch._inductor.dynagraph._GRID_AXES and run
run(True) again, and this probe's criteria correctly fail --
  * delete the "Grid2D" row     -> 5 recordings, log [unmodelled]: grid type Grid2D is not modelled
  * swap the 2D x/y axes        -> 5 recordings, log [selfcheck-mismatch]: at {'s77': 512}
The second shows a wrong axis order does not turn into wrong numbers: DynaGraph's own _replays_match catches it
on the recording shape, then the whole graph falls back, so the "0 recordings" criterion fails.

What it cannot reach (stated clearly so readers are not misled):
  * Grid3D was not hit. Tried permute + contiguous on (M,8,8) and (M,32,16), even with
    ic.triton.max_tiles=3 on; Inductor still collapses three dims into two and always gives Grid2D.
    So the Grid3D row of _GRID_AXES has still never been run by anything.
  * The recording shape here is the largest, 512; later shapes only get smaller, and the xnumel argument is
    patched, so "extra launches in the x direction" are absorbed by the kernel's own mask. That is, equal numerics on
    smaller shapes do not by themselves prove the 2D grid formula correct; what really guards it are the two negative
    controls above and DynaGraph's self-check at recording time.
"""
import logging
import os
import re
import sys

SHAPES = tuple(int(v) for v in os.environ.get("DG_SHAPES", "512,256,129,64,8").split(","))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "grid2d_planner.cu")

# When DynaGraph falls back it says only one line at INFO level. A KIND A probe should not fall back at all, but if it really
# does, it must be able to say which rule blocked it, otherwise the report is reduced to just "not served".
_LOGS: list[str] = []


class _Collect(logging.Handler):
    def emit(self, record):
        _LOGS.append(f"{record.name}: {record.getMessage()}")


def _setup_logging():
    h = _Collect()
    for name in ("torch._inductor.cudagraph_trees", "torch._inductor.dynagraph"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(h)


def run(dynagraph: bool):
    import torch, torch._inductor.config as ic
    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"

    n_record = {"n": 0}
    import torch._inductor.cudagraph_trees as ct
    orig_rec = ct.CUDAGraphTreeManager.record_function
    def spy_rec(self, *a, **kw):
        n_record["n"] += 1
        return orig_rec(self, *a, **kw)
    ct.CUDAGraphTreeManager.record_function = spy_rec

    # Only kernels that are actually launched go through .run(). Hooking __init__ misses some (autotune
    # candidates are built in another module, and in practice it also missed ones compiled in workers), so hook run.
    from torch._inductor.runtime import triton_heuristics as th
    grids: dict[str, str] = {}
    orig_run = th.CachingAutotuner.run
    def spy_run(self, *a, **kw):
        meta = self.inductor_meta or {}
        nm = meta.get("kernel_name") or "?"
        if nm not in grids:
            grids[nm] = str(meta.get("grid_type"))
        return orig_run(self, *a, **kw)
    th.CachingAutotuner.run = spy_run

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Broadcast to (M, 64): this also specializes the symbol of the input's second dim to 64,
            # i.e. makes the 2D tile's y axis static. See the file header.
            self.b = torch.nn.Parameter(torch.randn(64))

        def forward(self, x):                     # (M, 64)
            h = torch.relu(x * self.b + 1.0)      # (M, 64)
            # .contiguous() is the pivot of the whole probe: it forces Inductor to materialize a contiguous
            # (64, M) buffer read with transposed strides, so this pointwise becomes 2D.
            # Without it Inductor folds the transpose into the indexing and falls back to a 1D grid.
            return h.t().contiguous() * 2.0 - 1.0  # (64, M)

    # Both runs must get the same weights and the same inputs, otherwise the outputs cannot be compared bitwise.
    torch.manual_seed(0)
    m = M().cuda().eval()
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")

    outs = {}
    try:
        torch.manual_seed(1)
        for M_ in SHAPES:
            x = torch.randn(M_, 64, device="cuda")
            with torch.no_grad():
                # Call each shape twice: the first time cudagraph_trees sees a FunctionID
                # it only does an eager warmup, and it really records on the second. With one call the control group records
                # no graph at all, and the "recording count" criterion becomes useless.
                f(x)
                outs[M_] = (f(x).float().cpu().clone(),
                            m(x).float().cpu().clone())
    finally:
        ct.CUDAGraphTreeManager.record_function = orig_rec
        th.CachingAutotuner.run = orig_run
    return n_record["n"], outs, grids


def dumped_grid_types():
    """Read the kernel table at the head of the planner dump, return [(kernel name, grid_type)].

    This is what DynaGraph itself sees: grid_type Grid2D in the table means this kernel's
    grid was computed by the 2D branch of _GRID_AXES. The dump is written only after the planner is generated successfully,
    so a missing file == it never got that far.
    """
    if not os.path.exists(DUMP):
        return None
    with open(DUMP) as fh:
        head = fh.readline()
    names = re.findall(r"'name': '([^']*)'", head)
    types = re.findall(r"'grid_type': '?(\w+)'?", head)
    return list(zip(names, types)) if len(names) == len(types) else [
        (f"#{i}", t) for i, t in enumerate(types)
    ]


def main():
    import torch
    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1
    _setup_logging()
    os.makedirs(os.path.dirname(DUMP), exist_ok=True)
    if os.path.exists(DUMP):
        os.remove(DUMP)

    res, rec, grid = {}, {}, {}
    for flag in (False, True):
        # The dump only means something on the dynagraph side (the control group never builds a planner).
        if flag:
            os.environ["TORCHINDUCTOR_DYNAGRAPH_DUMP"] = DUMP
        n, outs, grids = run(flag)
        os.environ.pop("TORCHINDUCTOR_DYNAGRAPH_DUMP", None)
        res[flag], rec[flag], grid[flag] = outs, n, grids
        print(f"\n  dynagraph={flag}")
        print(f"    {n} recordings")
        for k, v in sorted(grids.items()):
            print(f"    kernel {k}: grid_type={v}")

    bad = 0

    # Criterion 3 first: if it fails, passing the two below cannot prove the 2D row of code was ever run.
    tiled = {k: v for k, v in grid[True].items() if v in ("Grid2D", "Grid3D")}
    table = dumped_grid_types()
    print("\n  target path (Grid2D/Grid3D):")
    print(f"    tiled kernels launched: {tiled or '(none at all)'}")
    print(f"    kernel table in the planner dump: {table if table is not None else '(no dump file)'}")
    in_table = [t for t in (table or []) if t[1] in ("Grid2D", "Grid3D")]
    if not tiled:
        print("    FAIL Inductor did not emit a tiled kernel this time; the probe missed its target")
        bad += 1
    elif not in_table:
        print("    FAIL the tiled kernel is not in DynaGraph's kernel table: it is not in the graph being served")
        bad += 1
    else:
        print(f"    ok DynaGraph computed the grid for {len(in_table)} tiled kernel(s) via _GRID_AXES")

    # The reference is the same compile path with dynagraph off, not eager. The criterion is "no farther from eager than
    # the control group": the two sides may pick different autotune configs, so a slight difference in summation/block order is expected.
    print("\n  vs dynagraph=False (criterion: no farther from eager than the control group):")
    for M_ in SHAPES:
        (a, ea), (b, eb) = res[False][M_], res[True][M_]
        if a.shape != b.shape:
            print(f"    M={M_} shape {tuple(b.shape)} != {tuple(a.shape)}  FAIL")
            bad += 1; continue
        seed_ok = (ea - eb).abs().max().item() == 0
        scale = max(ea.abs().max().item(), 1e-9)
        ctl = (a - ea).abs().max().item() / scale
        dyn = (b - eb).abs().max().item() / scale
        ok = seed_ok and dyn <= max(ctl * 1.5, 1e-6)
        print(f"    M={M_} ctl<->dyna {(a - b).abs().max().item():.2e}"
              f" | ctl<->eager {ctl:.1e} | dyna<->eager {dyn:.1e}"
              f" | same seed {seed_ok}" + ("  ok" if ok else "  FAIL"))
        bad += not ok

    print(f"\n  recordings {rec[False]} -> {rec[True]} ({len(SHAPES)} shapes)")
    if rec[True] != 0 or rec[False] != len(SHAPES):
        print("  FAIL recording count mismatch: DynaGraph should record none, the control group should record once per shape")
        bad += 1

    if _LOGS:
        print("\n  logs (all fallback reasons are here):")
        for line in _LOGS:
            print(f"    {line}")

    print("\n  " + ("all passed" if not bad else f"{bad} item(s) failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
