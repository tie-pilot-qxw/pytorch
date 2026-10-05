"""
Pin down: which structure produces size computations on the CPU (cat/unsqueeze) under dynamic shapes?

MobileNet-V2 has 71 of them, ResNet-18 has 0. Try one feature at a time.
Standalone script; does not import instrument.py.
"""
import sys, os
from collections import Counter
import torch, torch.nn as nn
import torch._inductor.config as ic, torch._dynamo.config as dc
import torch._inductor.scheduler as S

ic.force_disable_caches = True
dc.capture_dynamic_output_shape_ops = True
dc.capture_scalar_outputs = True

seen = {}
orig = S.Scheduler.should_partition


def patched(self, node, *a, **kw):
    out = orig(self, node, *a, **kw)
    if isinstance(out, str) and "unbacked" not in out:
        try:
            name = node.get_name()
        except Exception:
            name = str(id(node))
        if name not in seen:
            ir = getattr(node, "node", None)
            org = getattr(ir, "origins", None)
            try:
                o = ",".join(sorted({str(getattr(x, "target", x)) for x in org})[:2]) if org else "-"
            except Exception:
                o = "-"
            seen[name] = (out, o)
    return out


S.Scheduler.should_partition = patched


def probe(tag, mod, inp):
    seen.clear()
    torch._dynamo.reset()
    try:
        torch.compile(mod, dynamic=True, mode="reduce-overhead")(inp)
    except Exception:
        pass
    c = Counter(v[0] for v in seen.values())
    org = Counter(v[1] for v in seen.values())
    top = org.most_common(1)[0][0][:58] if org else "-"
    print(f"{tag:<34} non_gpu_nodes={len(seen):<4} {dict(c)}  origin: {top}")


C = 32
x = torch.randn(2, C, 56, 56, device="cuda")

probe("conv+bn+relu", nn.Sequential(
    nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU()).cuda().eval(), x)

probe("conv+bn+relu6", nn.Sequential(
    nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU6()).cuda().eval(), x)

probe("depthwise conv+bn+relu", nn.Sequential(
    nn.Conv2d(C, C, 3, padding=1, groups=C), nn.BatchNorm2d(C), nn.ReLU()).cuda().eval(), x)

probe("depthwise+pointwise (MBConv-like)", nn.Sequential(
    nn.Conv2d(C, C * 6, 1), nn.BatchNorm2d(C * 6), nn.ReLU6(),
    nn.Conv2d(C * 6, C * 6, 3, padding=1, groups=C * 6), nn.BatchNorm2d(C * 6), nn.ReLU6(),
    nn.Conv2d(C * 6, C, 1), nn.BatchNorm2d(C)).cuda().eval(), x)

probe("conv+bn+adaptive_avg_pool+flatten", nn.Sequential(
    nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C),
    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(C, 10)).cuda().eval(), x)

probe("conv+bn+dropout+linear", nn.Sequential(
    nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C),
    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
    nn.Dropout(0.2), nn.Linear(C, 10)).cuda().eval(), x)
