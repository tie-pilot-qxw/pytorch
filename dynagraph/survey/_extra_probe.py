"""Run the models_extra probes once to see how many pieces the truly data-dependent patterns get split into."""
import sys, os
import torch
import torch._inductor.config as ic, torch._dynamo.config as dc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_extra
from instrument import install, reset, snapshot

ic.force_disable_caches = True
install()

for cap in (True, False):
    dc.capture_dynamic_output_shape_ops = cap
    dc.capture_scalar_outputs = cap
    print(f"\n########## data-dependent capture = {cap}")
    for name in models_extra.ALL:
        torch._dynamo.reset(); reset()
        try:
            with torch.device("cuda"):
                m, a, k = models_extra.build(name)
            m = m.cuda()
            a = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in a)
            torch.compile(m, dynamic=True, mode="reduce-overhead")(*a, **k)
            tag = "ok"
        except Exception as e:
            tag = type(e).__name__
        s = snapshot()
        print(f"  {name:<10} {tag:<30} graph_breaks={s['dynamo_graph_breaks']:<3} "
              f"non_graphable_nodes={s['n_nodes_not_cudagraphable']:<4} "
              f"partitions={s['n_partitions_observed']} {s['partition_reason_counts']}")
