"""C++ runtime hit stats: after the run, list (hits, misses, entries) for every DynaGraphRunner. Usage is the same as esm.py."""
import atexit
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # e2e/, for esm and harness


def report():
    from torch._inductor import dynagraph as dg

    for o in gc.get_objects():
        if isinstance(o, dg.DynaGraphRunner):
            rt = o._rt
            print(f"runner {o.mode:9s} kernels {len(o.kernels):3d} inline {len(o.inline_sites or {}):3d} "
                  f"rt {"-" if rt is None else ("off" if rt is False else (rt.stats(), rt.size(), rt.n_execs()))} prog {o._rt_prog} "
                  f"extern_read {len(o.extern_read)} of which static {len(set(o.extern_read) & set(o.static_idxs))} "
                  f"child_sites {len(o.child_sites)} harvest keys {len(o._inline_ops)}", flush=True)


atexit.register(report)
sys.argv = ["esm.py"] + sys.argv[1:]
import esm

esm.main()
