import torch, torch.nn as nn, traceback
import torch._inductor.config as ic, torch._dynamo.config as dc
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instrument import install, reset, snapshot

ic.force_disable_caches = True
dc.capture_dynamic_output_shape_ops = True
dc.capture_scalar_outputs = True
install()

class Plain(nn.Module):
    def __init__(s):
        super().__init__(); s.a = nn.Linear(256, 256)
    def forward(s, x): return s.a(x).relu()

class NZ(nn.Module):
    def __init__(s):
        super().__init__(); s.a = nn.Linear(256, 256)
    def forward(s, x):
        y = s.a(x)
        return y[torch.nonzero(y[:, 0] > 0).squeeze(-1)]

for Model in (Plain, NZ):
    for mode in ("reduce-overhead", None):
        torch._dynamo.reset(); reset()
        label = f"{Model.__name__:<6} mode={str(mode):<16}"
        try:
            fm = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
            with fm, torch.device("cuda"):
                m = Model(); x = torch.randn(64, 256)
                kw = {"dynamic": True}
                if mode:
                    kw["mode"] = mode
                else:
                    ic.triton.cudagraphs = True   # turn cudagraph on by hand, without mode
                torch.compile(m, **kw)(x)
            s = snapshot()
            print(f"{label} OK   nodes={s['n_nodes_not_cudagraphable']} "
                  f"partitions={s['n_partitions_observed']} reasons={s['partition_reason_counts']}")
            if s["partition_reason_samples"]:
                print(f"{'':<24} samples: {s['partition_reason_samples'][:3]}")
        except Exception as e:
            s2 = snapshot()
            got = bool(s2["n_partitions_observed"])
            print(f"{label} run failed ({type(e).__name__}), compile data {'collected' if got else 'not collected'}: "
                  f"nodes={s2['n_nodes_not_cudagraphable']} partitions={s2['n_partitions_observed']} "
                  f"reasons={s2['partition_reason_counts']}")
            if s2["partition_reason_samples"]:
                print(f"{'':<24} samples: {s2['partition_reason_samples'][:3]}")
print("memory_allocated:", torch.cuda.memory_allocated() // 1024, "KiB")
