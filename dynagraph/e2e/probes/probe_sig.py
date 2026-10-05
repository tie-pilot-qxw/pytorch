"""Why grouping gets few hits: record each site's signature on every region's first call, and print the fields that differ between sites with the same op and the same normalized text but different signatures."""
import atexit
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch._inductor import dynagraph as dg

orig = dg.DynaGraphRunner._site_signature
seen = {}


def sig(self, i, env, offsets, args):
    r = orig(self, i, env, offsets, args)
    key = (id(self), tuple(sorted(env.items())))
    d = seen.setdefault(key, {})
    if i not in d:
        d[i] = (r[0] if r else None, self.site_args[i][:140])
    return r


dg.DynaGraphRunner._site_signature = sig


def report():
    for key, d in list(seen.items())[:40]:
        sigs = [s for s, _ in d.values()]
        print(f"region {key[0] % 1000}: {len(d)} sites, {len(set(sigs))} distinct sigs, {sum(s is None for s in sigs)} None", flush=True)
        by = collections.defaultdict(list)
        for i, (s, txt) in d.items():
            if s is not None:
                by[(s[0], s[1])].append((i, s, txt))
        shown = 0
        for k, lst in by.items():
            if len(lst) < 2 or len({x[1] for x in lst}) == 1:
                continue
            a, b = lst[0], lst[-1]
            for pa, pb in zip(a[1][2], b[1][2]):
                if pa != pb:
                    print(f"  site {a[0]} vs {b[0]}: {pa}  !=  {pb}", flush=True)
            print(f"    {a[2]}\n    {b[2]}", flush=True)
            shown += 1
            if shown >= 4:
                break
        nt = collections.Counter((s[0], s[1]) for s in sigs if s)
        print(f"  norm-text groups {len(nt)}", flush=True)


atexit.register(report)
sys.argv = ["esm.py"] + sys.argv[1:]
import esm

esm.main()
