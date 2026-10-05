#!/usr/bin/env python3
"""Q5: can Inductor's shared-mem ever depend on the RUNTIME shape?
Also: what happens to a persistent reduction when the REDUCTION dim is dynamic?"""
from __future__ import annotations
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
import torch, torch._inductor.config as ic
ic.force_disable_caches = True
from torch._inductor.runtime import triton_heuristics as th

LAUNCHES = []
orig_run = th.CachingAutotuner.run
def run(self, *args, stream, **kw):
    r = orig_run(self, *args, stream=stream, **kw)
    if self.launchers:
        L = self.launchers[0]
        LAUNCHES.append((self.inductor_meta.get("kernel_name") or self.fn.__name__,
                         str(L.config), L.shared,
                         [a for a in args if isinstance(a, int)]))
    return r
th.CachingAutotuner.run = run

NCOMPILE = [0]
orig_ml = th.StaticTritonCompileResult.make_launcher
def ml(self):
    NCOMPILE[0] += 1
    return orig_ml(self)
th.StaticTritonCompileResult.make_launcher = ml

print("### A: dynamic BATCH dim, reduction dim static (512)")
def fa(x): return torch.softmax(x, -1) + x.sum(-1, keepdim=True)
ca = torch.compile(fa, dynamic=True)
for n in (37, 41, 97, 193):
    LAUNCHES.clear()
    ca(torch.randn(n, 512, device="cuda")); torch.cuda.synchronize()
    print(f"  n={n:4d} compiles_so_far={NCOMPILE[0]:3d} " +
          " | ".join(f"{k.split('_fused_')[-1]}: shared={s} cfg=({c})" for k,c,s,ii in LAUNCHES))

print("### B: dynamic REDUCTION dim")
def fb(x): return torch.softmax(x, -1) + x.sum(-1, keepdim=True)
cb = torch.compile(fb, dynamic=True)
for r in (512, 777, 1024, 2048):
    LAUNCHES.clear()
    before = NCOMPILE[0]
    cb(torch.randn(64, r, device="cuda")); torch.cuda.synchronize()
    print(f"  r={r:5d} new_compiles={NCOMPILE[0]-before:3d} " +
          " | ".join(f"{k.split('_fused_')[-1]}: shared={s} cfg=({c}) ints={ii}" for k,c,s,ii in LAUNCHES))
