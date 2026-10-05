"""Does the hook itself disturb what it measures? Compare compile time and partition results with / without the hook."""
import sys, os, time
import torch
import torch._inductor.config as ic, torch._dynamo.config as dc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models

ic.force_disable_caches = True
dc.capture_dynamic_output_shape_ops = True
dc.capture_scalar_outputs = True

spec = sys.argv[1] if len(sys.argv) > 1 else "tv:resnet18"
use_hook = len(sys.argv) > 2 and sys.argv[2] == "hook"

n_parts = []
if use_hook:
    from instrument import install, reset, snapshot
    install(); reset()
else:
    # Without the hook we still need the partition count as a baseline: wrap only graph_partition, leave should_partition alone
    import torch._inductor.scheduler as S
    _o = S.Scheduler.graph_partition

    def _p(self, *a, **kw):
        out = _o(self, *a, **kw)
        try:
            n_parts.append(len(out[0]))
        except Exception:
            pass
        return out
    S.Scheduler.graph_partition = _p

with models.device_ctx():
    m, a, k = models.build(spec)
t0 = time.time()
try:
    torch.compile(m, dynamic=True, mode="reduce-overhead")(*a, **k)
except Exception as e:
    print("(run phase:", type(e).__name__, ")")
dt = time.time() - t0

if use_hook:
    s = snapshot()
    print(f"RESULT hook=1 wall={dt:.1f}s parts={s['n_partitions_observed']} "
          f"nodes={s['n_nodes_not_cudagraphable']} calls={s['should_partition_raw_calls']}")
else:
    print(f"RESULT hook=0 wall={dt:.1f}s parts={n_parts}")
