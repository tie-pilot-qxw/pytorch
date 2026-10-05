"""
Does turning dynamic off bring cpu_ops down to zero?

The signature of the ReLU6 bug: it only appears under dynamic=True (a module's float attribute gets symbolized
and materialized as a CPU tensor). If other models' cpu_ops share this signature, they are other instances of the same bug
and fall in the "fixable upstream" category; if they remain under dynamic=False, they are structural.
"""
import sys, os
from collections import Counter
import torch
import torch._inductor.config as ic, torch._dynamo.config as dc
import torch._inductor.scheduler as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models

ic.force_disable_caches = True
seen = {}
orig = S.Scheduler.should_partition


def patched(self, node, *a, **kw):
    out = orig(self, node, *a, **kw)
    if isinstance(out, str):
        try:
            n = node.get_name()
        except Exception:
            n = str(id(node))
        seen.setdefault(n, out)
    return out


S.Scheduler.should_partition = patched

for spec in sys.argv[1:]:
    row = []
    for dyn in (True, False):
        dc.capture_dynamic_output_shape_ops = dyn
        dc.capture_scalar_outputs = dyn
        seen.clear(); torch._dynamo.reset()
        try:
            with models.device_ctx():
                m, a, k = models.build(spec)
            torch.compile(m, dynamic=dyn, mode="reduce-overhead")(*a, **k)
            tag = "ok"
        except Exception as e:
            tag = type(e).__name__
        row.append((dyn, tag, len(seen), dict(Counter(seen.values()))))
    print(f"\n{spec}")
    for dyn, tag, n, c in row:
        print(f"  dynamic={str(dyn):<6} {tag:<28} non_gpu_nodes={n:<4} {c}")
    a_n, b_n = row[0][2], row[1][2]
    verdict = ("**all from symbolic materialization under dynamic shapes -> same class as ReLU6, fixable upstream**"
               if a_n > 0 and b_n == 0 else
               "still present with dynamic off -> structural" if b_n > 0 else "0 on both sides")
    print(f"  => {verdict}")
