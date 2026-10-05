"""Why sharing did not take effect: count _bind_plan returning None vs. succeeding, and _rebind calls. Usage: same as esm.py."""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils import _capture_launch as cl

C = collections.Counter()
ob, orb = cl._bind_plan, cl._rebind


def bp(t, old):
    r = ob(t, old)
    C["plan_none" if r is None else "plan_ok"] += 1
    if r is None and C["plan_none"] <= 2:
        ext = [cl._extent(x) for x in old]
        for L in t:
            for j, b in enumerate(L.args):
                for w in range(0, len(b) - 7, 8):
                    v = int.from_bytes(b[w:w + 8], "little")
                    if v > 1 << 40:
                        print(f"  arg {j} word {w}: {hex(v)} in extent {[lo <= v < hi for lo, hi in ext]}", flush=True)
        print("  extents", [(hex(lo), hi - lo) for lo, hi in ext], flush=True)
    return r


def rb(*a):
    C["rebind"] += 1
    return orb(*a)


cl._bind_plan, cl._rebind = bp, rb
import atexit

atexit.register(lambda: print("SHARE", dict(C), flush=True))
sys.argv = ["esm.py"] + sys.argv[1:]
import esm

esm.main()
