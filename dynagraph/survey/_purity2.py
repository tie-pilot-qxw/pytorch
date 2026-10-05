"""
A more accurate purity check: do the inputs of those hundred-or-so CPU nodes include any tensors on the GPU?

The previous version only recorded the device of nodes visited by should_partition and could not
see the device of intermediate buffers, so its "reads from GPU = 0" does not count. This version
looks up every buffer's device directly from V.graph.
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

result = {}
orig_gp = S.Scheduler.graph_partition


def patched_gp(self, *a, **kw):
    """At partition time, grab the buffer devices of the whole graph and the reads of the CPU nodes in one go."""
    from torch._inductor.virtualized import V
    try:
        dev_of = {}
        for name in list(getattr(V.graph, "name_to_buffer", {}) or {}):
            try:
                dev_of[name] = str(V.graph.get_buffer(name).get_device())
            except Exception:
                pass
        gi = getattr(V.graph, "graph_inputs", None)
        if isinstance(gi, dict):
            items = gi.items()
        elif isinstance(gi, (list, tuple)):
            names = getattr(V.graph, "graph_input_names", []) or []
            items = zip(names, gi)
        else:
            items = []
        for name, box in items:
            try:
                dev_of[str(name)] = str(box.get_device())
            except Exception:
                pass

        cpu_reads = Counter()
        n_cpu = 0
        for node in self.nodes:
            try:
                d = str(node.get_device())
            except Exception:
                continue
            if "cpu" not in d:
                continue
            n_cpu += 1
            for r in node.read_writes.reads:
                rd = dev_of.get(r.name, "?")
                cpu_reads["cuda" if "cuda" in rd else ("cpu" if "cpu" in rd else rd)] += 1
        if n_cpu and "n_cpu" not in result:
            result.update(n_cpu=n_cpu, reads=dict(cpu_reads), n_buf=len(dev_of))
    except Exception as e:
        result.setdefault("err", f"{type(e).__name__}: {e}")
    return orig_gp(self, *a, **kw)


S.Scheduler.graph_partition = patched_gp

spec = sys.argv[1] if len(sys.argv) > 1 else "tvdet:ssd300_vgg16"
if spec.startswith("extra:"):
    import models_extra
    with models.device_ctx():
        m, a, k = models_extra.build(spec[6:])
    m = m.cuda()
    a = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in a)
else:
    with models.device_ctx():
        m, a, k = models.build(spec)
try:
    torch.compile(m, dynamic=True, mode="reduce-overhead")(*a, **k)
except Exception as e:
    print("(run phase:", type(e).__name__, ")")

print(f"\n{spec}")
print(f"  CPU scheduler nodes: {result.get('n_cpu')}   (total buffers in graph {result.get('n_buf')})")
print(f"  these CPU nodes read from (by source): {result.get('reads')}")
print("  -> the 'cuda' entry is what matters: >0 means the CPU computation depends on GPU data, not a pure shape computation")
if result.get("err"):
    print("  (exception while grabbing:", result["err"], ")")
