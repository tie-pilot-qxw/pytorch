"""Minimal prototype, step 3: split the host time of one training step by phase (no sync), to see how much is left outside DG.

ESM: forward (until f(...) returns), backward (until loss.backward() returns), optimizer (step + zero_grad):
the host time of each, and DG __call__'s share of it. Usage: MODE=compile|dg python e2e/proto_phases.py esm [args]
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
mod = importlib.import_module(name)
mod.main()
mode = os.environ.get("MODE", "compile")
m = mod.make_model()
opt = torch.optim.AdamW(m.parameters(), lr=4e-4)
f = harness.compiled(m, mode)
from torch._inductor import dynagraph as dg

dgt = {"t": 0.0}
oc = dg.DynaGraphRunner.__call__


def call(self, inputs):
    t = time.perf_counter()
    try:
        return oc(self, inputs)
    finally:
        dgt["t"] += time.perf_counter() - t


dg.DynaGraphRunner.__call__ = call
from torch._inductor import output_code as oc_mod

wr = {"t": 0.0, "n": 0}
_ocall = oc_mod.CompiledFxGraph.__call__


def wcall(self, inputs):
    # The Inductor wrapper's `call` (launching the region's kernels); what DynaGraph replaces.
    t = time.perf_counter()
    try:
        return _ocall(self, inputs)
    finally:
        wr["t"] += time.perf_counter() - t
        wr["n"] += 1


oc_mod.CompiledFxGraph.__call__ = wcall
ph = {"fwd": 0.0, "bwd": 0.0, "opt": 0.0}
dgp = {"fwd": 0.0, "bwd": 0.0}


def step(b, rec):
    x, mask, y = b
    t0 = time.perf_counter(); d0 = dgt["t"]
    with harness.amp():
        loss = f(input_ids=x, attention_mask=mask, labels=y).loss
    t1 = time.perf_counter(); d1 = dgt["t"]
    loss.backward()
    t2 = time.perf_counter(); d2 = dgt["t"]
    opt.step()
    opt.zero_grad(set_to_none=True)
    t3 = time.perf_counter()
    if rec:
        ph["fwd"] += t1 - t0; ph["bwd"] += t2 - t1; ph["opt"] += t3 - t2
        dgp["fwd"] += d1 - d0; dgp["bwd"] += d2 - d1
    return loss.detach().clone()


bs = cap["batches"]
for b in bs[:10]:
    step(b, False)
torch.cuda.synchronize()
n = len(bs) - 10
wr["t"] = 0.0
wr["n"] = 0
t0 = time.perf_counter()
for b in bs[10:]:
    step(b, True)
torch.cuda.synchronize()
wall = (time.perf_counter() - t0) / n * 1e3
print(f"[{name} {mode}] new-shape step (no sync) wall {wall:.2f} ms; host: forward {ph['fwd'] / n * 1e3:.2f}"
      f" (of which DG {dgp['fwd'] / n * 1e3:.2f}), backward {ph['bwd'] / n * 1e3:.2f} (of which DG {dgp['bwd'] / n * 1e3:.2f}), "
      f"optimizer {ph['opt'] / n * 1e3:.2f} ms; Inductor compiled-region calls {wr['n'] / n:.1f} totaling {wr['t'] / n * 1e3:.2f} ms, "
      f"other host {(ph['fwd'] + ph['bwd'] + ph['opt'] - wr['t']) / n * 1e3:.2f} ms")
