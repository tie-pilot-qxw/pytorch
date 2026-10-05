"""Confirm causality: is it dynamic=True symbolizing the module's float attributes?"""
import torch, torch.nn as nn
from collections import Counter
import torch._inductor.config as ic
import torch._inductor.scheduler as S

ic.force_disable_caches = True
seen = {}
orig = S.Scheduler.should_partition


def patched(self, node, *a, **kw):
    out = orig(self, node, *a, **kw)
    if isinstance(out, str) and "unbacked" not in out:
        try:
            n = node.get_name()
        except Exception:
            n = str(id(node))
        seen.setdefault(n, out)
    return out


S.Scheduler.should_partition = patched
C = 32
x = torch.randn(2, C, 56, 56, device="cuda")

for dyn in (True, False):
    for tag, m in (("nn.ReLU6()", nn.ReLU6()),
                   ("nn.Hardtanh(0, 6) int", nn.Hardtanh(0, 6)),
                   ("nn.Hardtanh(0., 6.) float", nn.Hardtanh(0.0, 6.0))):
        seen.clear(); torch._dynamo.reset()
        try:
            torch.compile(m.cuda().eval(), dynamic=dyn, mode="reduce-overhead")(x)
        except Exception:
            pass
        c = Counter(seen.values())
        print(f"dynamic={str(dyn):<6} {tag:<28} nonGPU_nodes={len(seen):<3} {dict(c)}")
