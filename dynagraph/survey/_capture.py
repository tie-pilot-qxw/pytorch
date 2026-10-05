"""Only after the unbacked-capture switches are turned on does data dependence reach Inductor's should_partition."""
import torch, torch._inductor.config as ic, torch._dynamo.config as dc
from instrument import install, reset, snapshot
from models import build
from torch._dynamo.utils import counters
ic.force_disable_caches = True
install()

for cap in [False, True]:
    dc.capture_dynamic_output_shape_ops = cap
    dc.capture_scalar_outputs = cap
    print(f"\n########## capture_dynamic_output_shape_ops={cap}")
    for spec in ["builtin:plain", "builtin:datadep", "builtin:nonzero"]:
        torch._dynamo.reset(); counters.clear(); reset()
        m, inp = build(spec)
        expl = torch._dynamo.explain(m)(*inp)
        torch._dynamo.reset(); counters.clear(); reset()
        m, inp = build(spec)
        try:
            torch.compile(m, dynamic=True, mode="reduce-overhead")(*inp)
            s = snapshot(); err = ""
        except Exception as e:
            s = snapshot(); err = f"  !! {type(e).__name__}: {str(e)[:100]}"
        allc = dict(counters["inductor"])
        cg = {k: v for k, v in allc.items() if "cudagraph" in k or "partition" in k}
        print(f"  {spec:<18} dynamo_graphs={expl.graph_count} breaks={expl.graph_break_count} "
              f"| partition_true={s['n_partition_true']} cg_counters={cg}{err}")
        if allc and not cg:
            print(f"      (all inductor counters: {list(allc)[:8]})")
