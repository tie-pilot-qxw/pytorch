import torch
from instrument import install, reset, snapshot
from models import build
from torch._dynamo.utils import counters
import torch._inductor.config as _ic
_ic.force_disable_caches = True
install()
for mode in [None, "reduce-overhead"]:
    for spec in ["builtin:plain", "builtin:datadep"]:
        torch._dynamo.reset(); counters.clear(); reset()
        m, inp = build(spec)
        kw = {"dynamic": True}
        if mode:
            kw["mode"] = mode
        err = ""
        try:
            torch.compile(m, **kw)(*inp); ok = True
        except Exception as e:
            ok = False; err = f"{type(e).__name__}: {str(e)[:150]}"
        s = snapshot()
        cg = {k: v for k, v in counters["inductor"].items() if "cudagraph" in k or "partition" in k}
        print(f"mode={str(mode):<16} {spec:<18} ok={ok} true={s['n_partition_true']:<3} "
              f"n_partitions={s['n_partitions']} skips={s['cudagraph_skips']} {err}")
        print(f"   inductor counters: {cg}")
