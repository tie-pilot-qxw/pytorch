"""Does input size change the partition conclusion? Check with a small probe so it does not compete with the full run for resources."""
import sys, os
import torch, torch.nn as nn
import torch._inductor.config as ic, torch._dynamo.config as dc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instrument import install, reset, snapshot

ic.force_disable_caches = True
dc.capture_dynamic_output_shape_ops = True
dc.capture_scalar_outputs = True
install()


class Clean(nn.Module):
    def __init__(s):
        super().__init__(); s.a = nn.Linear(256, 256); s.b = nn.Linear(256, 256)
    def forward(s, x): return s.b(s.a(x).relu())


class HasCpuOps(nn.Module):
    """nn.ReLU6 -> scalar computation on the CPU"""
    def __init__(s):
        super().__init__(); s.a = nn.Linear(256, 256); s.r = nn.ReLU6()
    def forward(s, x): return s.r(s.a(x))


class HasUnbacked(nn.Module):
    def __init__(s):
        super().__init__(); s.a = nn.Linear(256, 256)
    def forward(s, x):
        y = s.a(x)
        return y[torch.nonzero(y[:, 0] > 0).squeeze(-1)]


for Model in (Clean, HasCpuOps, HasUnbacked):
    res = []
    for bs in (2, 32, 256):
        torch._dynamo.reset(); reset()
        try:
            with torch.device("cuda"):
                m = Model()
            x = torch.randn(bs, 256, device="cuda")
            torch.compile(m, dynamic=True, mode="reduce-overhead")(x)
            tag = "ok"
        except Exception as e:
            tag = type(e).__name__
        s = snapshot()
        res.append((bs, tag, s["n_nodes_not_cudagraphable"],
                    tuple(s["n_partitions_observed"]),
                    tuple(sorted(s["partition_reason_counts"].items()))))
    same = len({(r[2], r[3], r[4]) for r in res}) == 1
    print(f"\n{Model.__name__}  conclusion across batches: {'consistent' if same else '**INCONSISTENT**'}")
    for bs, tag, n, parts, reasons in res:
        print(f"  batch={bs:<4} {tag:<24} nodes={n:<3} partitions={list(parts)} {dict(reasons)}")
