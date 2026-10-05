"""
What operators are those 71 cpu_ops, really?

Standalone script that does its own patching and **does not import instrument.py**, so it cannot disturb the full survey that is currently running.
"""
import sys, os
from collections import Counter
import torch
import torch._inductor.config as ic, torch._dynamo.config as dc
import torch._inductor.scheduler as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models

ic.force_disable_caches = True
dc.capture_dynamic_output_shape_ops = True
dc.capture_scalar_outputs = True

seen = {}
orig = S.Scheduler.should_partition


def patched(self, node, *a, **kw):
    out = orig(self, node, *a, **kw)
    if isinstance(out, str) and "ops" in out and "unbacked" not in out:
        try:
            name = node.get_name()
        except Exception:
            name = str(id(node))
        if name in seen:
            return out
        ir = getattr(node, "node", None)
        info = {
            "reason": out,
            "ir_type": type(ir).__name__ if ir is not None else "None(Fused)",
            "device": str(getattr(node, "get_device", lambda: None)()),
        }
        op = getattr(ir, "op_overload", None)
        if op is not None:
            info["op"] = str(op)
        origins = getattr(ir, "origins", None) or getattr(
            getattr(ir, "origin_node", None), "target", None)
        if origins:
            try:
                info["origin"] = ",".join(sorted({str(getattr(o, "target", o))
                                                  for o in origins})[:3])
            except Exception:
                info["origin"] = str(origins)[:60]
        seen[name] = info
    return out


S.Scheduler.should_partition = patched

spec = sys.argv[1] if len(sys.argv) > 1 else "tv:mobilenet_v2"
with models.device_ctx():
    m, a, k = models.build(spec)
try:
    torch.compile(m, dynamic=True, mode="reduce-overhead")(*a, **k)
except Exception as e:
    print("(run phase:", type(e).__name__, ")")

print(f"\n{spec}: {len(seen)} non-GPU / copy nodes\n")
for key in ("reason", "ir_type", "op", "origin", "device"):
    c = Counter(v.get(key, "-") for v in seen.values())
    print(f"by {key}:")
    for val, n in c.most_common(8):
        print(f"  {n:>4}  {str(val)[:96]}")
    print()
