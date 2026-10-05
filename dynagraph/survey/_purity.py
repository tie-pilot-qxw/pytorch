"""
Design-assumption check: are the hundred-or-so CPU nodes in the detection models an independent compute chain that **depends only on shapes**?

If so, the planner kernel can compute them directly without waiting on any device-side result --
the cleanest case for the DynaGraph mechanism. If their inputs mix in values of GPU tensors, it is
not a pure shape computation.
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

cpu_nodes = {}      # name -> (set of buffer names it reads)
all_dev = {}        # buffer name -> device
orig = S.Scheduler.should_partition


def patched(self, node, *a, **kw):
    out = orig(self, node, *a, **kw)
    try:
        name = node.get_name()
    except Exception:
        return out
    dev = None
    try:
        dev = str(node.get_device())
    except Exception:
        pass
    all_dev[name] = dev
    if isinstance(out, str) and out.endswith("ops") and "unbacked" not in out:
        try:
            reads = {d.name for d in node.read_writes.reads}
        except Exception:
            reads = set()
        cpu_nodes.setdefault(name, reads)
    return out


S.Scheduler.should_partition = patched

spec = sys.argv[1] if len(sys.argv) > 1 else "tvdet:ssd300_vgg16"
with models.device_ctx():
    m, a, k = models.build(spec)
try:
    torch.compile(m, dynamic=True, mode="reduce-overhead")(*a, **k)
except Exception as e:
    print("(run phase:", type(e).__name__, ")")

print(f"\n{spec}: {len(cpu_nodes)} non-GPU nodes")
names = set(cpu_nodes)
inside = outside_gpu = outside_unknown = 0
ext_samples = Counter()
for n, reads in cpu_nodes.items():
    for r in reads:
        if r in names:
            inside += 1
        else:
            d = all_dev.get(r)
            if d and "cuda" in d:
                outside_gpu += 1
                ext_samples[r[:24]] += 1
            else:
                outside_unknown += 1
                ext_samples[f"{r[:24]}(dev={d})"] += 1

print(f"  reads from inside the same CPU chain : {inside}")
print(f"  reads from **buffers on the GPU** : {outside_gpu}   <= this is the number that matters")
print(f"  reads from outside the graph / unknown (mostly inputs or constants) : {outside_unknown}")
if ext_samples:
    print("  external read samples:")
    for k2, v in ext_samples.most_common(8):
        print(f"    {v:>4}  {k2}")
