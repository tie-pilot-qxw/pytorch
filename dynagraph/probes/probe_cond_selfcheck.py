#!/usr/bin/env python3
"""The branch self-check itself must be checked: swapped wiring must be caught.

At capture time the two branches are wired as the two bodies of a conditional node, and the selector is set on the device; the data only
takes one of them, so an ordinary self-check passes even with the bodies swapped. `_branches_match` uses the forcing slot in ctx
to make the graph take each branch in turn, and the reference is the same wrapper with the selector forced to the same branch.

Three runs here: the clean one must pass; inverting the forcing semantics on the graph side must be caught; removing the
forcing on the eager reference side must also be caught (otherwise both sides could be inert together and "all passed" would be fake).
"""
import logging, os, subprocess, sys

CASE = "cond_pointwise"


def _run_child(mode):
    tags = []

    class _Grab(logging.Handler):
        def emit(self, r):
            m = r.getMessage()
            if "DynaGraph fallback [" in m:
                tags.append(m.split("[", 1)[1].split("]", 1)[0])

    logging.basicConfig(level=logging.WARNING)
    lg = logging.getLogger("torch._inductor.dynagraph")
    lg.setLevel(logging.INFO)
    lg.addHandler(_Grab())

    from torch._inductor import dynagraph as dg

    if mode == "swap_graph":
        # "force to branch b" becomes "force to the other branch"
        orig = dg._planner_u_source
        dg._planner_u_source = lambda *a, **kw: orig(*a, **kw).replace(
            "= pin - 1;", "= 2 - pin;"
        )
    elif mode == "drop_eager":
        # Only the eager reference of the branch self-check step ignores the forcing (warmup must still warm up per branch,
        # otherwise it degrades into unsettled-config and this is no longer what is being tested)
        cls = next(
            getattr(dg, n)
            for n in dir(dg)
            if isinstance(getattr(dg, n), type)
            and hasattr(getattr(dg, n), "_branches_match")
        )
        real = cls._branches_match

        def patched(self, env):
            g = self.model.__globals__
            keep = g.get("__dg_sel")
            g["__dg_sel"] = lambda i, real_v: real_v
            try:
                return real(self, env)
            finally:
                g["__dg_sel"] = keep

        cls._branches_match = patched

    sys.argv = ["x", "--cases", CASE, "--only", "on"]
    import probe_unbacked_device as P

    P.main()
    print("TAGS " + ",".join(sorted(set(tags))), flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "--child":
    _run_child(sys.argv[2])
    raise SystemExit(0)

bad = 0
for mode, want in (("clean", False), ("swap_graph", True), ("drop_eager", True)):
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", mode],
        capture_output=True, text=True, timeout=1800,
    )
    line = next((l for l in r.stdout.splitlines() if l.startswith("TAGS ")), "TAGS ")
    got = "cond-branch-mismatch" in line
    ok = got == want
    what = "swap was caught" if want else "clean run has no false positive"
    print(f"  {mode:<11} tags {line[5:] or '-':<24} {what}  {'ok' if ok else 'FAIL'}")
    bad += not ok
print("  all passed: both sides of the branch self-check are in effect" if not bad else f"  {bad} item(s) failed")
sys.exit(1 if bad else 0)
