"""Minimal prototype, step one: in DG mode, how much DG has to handle in one new-shape step, and how much host time is spent outside DG.

Per step: region calls, inline sites, Triton kernel nodes, output tensor count; host launch time (no sync)
and DG __call__'s share of it. Usage: same as gputime.py, with MODE fixed to dg.
"""
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
name = sys.argv[1]
sys.argv = [name + ".py"] + sys.argv[2:]
import harness
import torch

cap = {}
harness.run = lambda make, batches, modes, **kw: cap.update(make=make, batches=batches)
importlib.import_module(name).main()
model, step = cap["make"]()
f = harness.compiled(model, "dg")
from torch._inductor import dynagraph as dg

stat = {"calls": 0, "inline": 0, "kernels": 0, "outs": 0, "dg": 0.0}
oc = dg.DynaGraphRunner.__call__


def call(self, inputs):
    t = time.perf_counter()
    key = tuple(v for v in inputs if isinstance(v, int))
    r = oc(self, inputs)
    stat["dg"] += time.perf_counter() - t
    if os.environ.get("DGSYNC"):
        try:
            torch.cuda.synchronize()
        except Exception as e:
            ex = self.ex
            print(f"FAULT after call {stat['calls']} region kernels {len(self.kernels)} inline {len(self.inline_sites)} "
                  f"ints {key} execs {list(self.execs)} ex inline_cluster "
                  f"{sorted((i, tuple(c)) for i, c in ex.inline_cluster.items())[:12]}", flush=True)
            raise
    stat["calls"] += 1
    stat["inline"] += len(self.inline_sites)
    stat["kernels"] += len(self.kernels)
    stat["outs"] += len(r) if isinstance(r, (list, tuple)) else 0
    return r


dg.DynaGraphRunner.__call__ = call
bs = cap["batches"]
for b in bs[:10]:
    step(f, b)
torch.cuda.synchronize()
for k in stat:
    stat[k] = 0
host = 0.0
n = len(bs) - 10
for b in bs[10:]:
    t0 = time.perf_counter()
    step(f, b).clone()
    host += time.perf_counter() - t0
torch.cuda.synchronize()
print(f"[{name} counts] per new-shape step: region calls {stat['calls'] / n:.1f}, inline sites {stat['inline'] / n:.0f}, "
      f"Triton kernel nodes {stat['kernels'] / n:.0f}, output tensors {stat['outs'] / n:.0f}; "
      f"host launch {host / n * 1e3:.2f} ms, of which DG __call__ {stat['dg'] / n * 1e3:.2f} ms, "
      f"outside DG {(host - stat['dg']) / n * 1e3:.2f} ms")
