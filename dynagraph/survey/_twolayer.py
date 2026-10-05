"""Dynamo graph breaks and Inductor partitions are two separate layers; measure each one."""
import torch, torch._inductor.config as ic
from instrument import install, reset, snapshot
from models import build
from torch._dynamo.utils import counters
ic.force_disable_caches = True
install()

for spec in ["builtin:plain", "builtin:datadep", "builtin:nonzero"]:
    torch._dynamo.reset(); counters.clear(); reset()
    m, inp = build(spec)
    # layer 1: how many graphs Dynamo split into, and why
    expl = torch._dynamo.explain(m)(*inp)
    gb = [str(r) for r in getattr(expl, "break_reasons", [])]
    print(f"\n=== {spec}")
    print(f"  Dynamo: graph_count={expl.graph_count} break_count={expl.graph_break_count} op_count={expl.op_count}")
    for r in gb[:4]:
        print(f"    break: {r[:160]}")

    # layer 2: how many segments Inductor partitioned into in cudagraph mode
    torch._dynamo.reset(); counters.clear(); reset()
    m, inp = build(spec)
    try:
        torch.compile(m, dynamic=True, mode="reduce-overhead")(*inp)
        s = snapshot()
        cg = {k: v for k, v in counters["inductor"].items() if "cudagraph" in k or "partition" in k}
        print(f"  Inductor: partition_true={s['n_partition_true']} counters={cg}")
    except Exception as e:
        print(f"  Inductor: FAILED {type(e).__name__}: {str(e)[:150]}")
