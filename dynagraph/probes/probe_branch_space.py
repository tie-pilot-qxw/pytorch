#!/usr/bin/env python3
"""Declared branch space: the budget follows what the op declares, and a value outside the space must be reported.

`torch.utils._capture_deps` used to express only "did it change" (an equality relation). An op's capture
switches among a set of kernels it can name itself; equality only lets the consumer know "the old one is stale",
not "what it can turn into" -- so every branch pays for a capture the first time it shows up,
and how many bodies a site may grow can only be guessed by the consumer as a constant (today `dynagraph_max_graphs`,
whose comment justifies it with "cuDNN conv has three tiers, cuBLAS two").

With `branches=` added:
  1. a site's body budget is set by the number of declared branches; undeclared sites still use that constant;
  2. if the resolver returns a value outside the declaration, it is treated as "the declaration is false" and reported,
     instead of turning into a budget overrun nobody can explain.
"""
import logging
import sys


def main() -> int:
    from torch._inductor.dynagraph import DynaGraphRunner, _max_graphs
    from torch.utils import _capture_deps

    bad = 0

    def check(what, got, want):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(f"    {what:<44} {got!r:<22} {'OK' if ok else f'**want {want!r}**'}")

    _capture_deps.register("probe::silent", ("g",), lambda g: g)
    _capture_deps.register("probe::named", ("g",), lambda g: g, branches=("a", "b", "c"))

    print("  1. Per-site body budget")
    # `_site_budget` only reads `site_deps`, so there is no need to build a whole runner.
    fake = DynaGraphRunner.__new__(DynaGraphRunner)
    fake.site_deps = [None, ("probe::silent", ("x",)), ("probe::named", ("x",))]
    check("undeclared site -> constant", DynaGraphRunner._site_budget(fake, 0), (_max_graphs(), False))
    check("declared but no space given -> constant", DynaGraphRunner._site_budget(fake, 1), (_max_graphs(), False))
    check("three declared branches -> 3", DynaGraphRunner._site_budget(fake, 2), (3, True))

    print("\n  2. Returning a value outside the declaration")
    fake.site_deps = [("probe::named", ("a",))]
    fake._undeclared = {}
    rec = []

    class _Grab(logging.Handler):
        def emit(self, r):
            if r.levelno >= logging.WARNING:
                rec.append(r.getMessage())

    lg = logging.getLogger("torch._inductor.dynagraph")
    h = _Grab()
    lg.addHandler(h)
    lg.setLevel(logging.WARNING)
    try:
        k1 = DynaGraphRunner._deps_key(fake)
        check("in-space value is silent", len(rec), 0)
        check("value goes into the key", k1, (("probe::named", ("a",), "a"),))
        fake.site_deps = [("probe::named", ("zzz",))]
        DynaGraphRunner._deps_key(fake)
        check("out-of-space value warns once", len(rec), 1)
        DynaGraphRunner._deps_key(fake)
        check("same value does not warn again", len(rec), 1)
        if rec:
            print(f"      -> {rec[0]}")
    finally:
        lg.removeHandler(h)

    print("\n  3. Old three-argument registration is not broken")
    d = _capture_deps.lookup("probe::silent")
    check("branches is None", d.branches, None)
    check("index access is still the resolver", d[1]("q"), "q")
    check("unpacking two fields still works", tuple(d[:2])[0], ("g",))

    print("\n  " + ("all passed" if not bad else f"{bad} failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
