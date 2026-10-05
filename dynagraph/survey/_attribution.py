"""
Attribution split: replace nn.ReLU6 with an equivalent form that does not trigger the bug, and see how many cpu_ops remain.

This is the key step of the argument. If all cpu_ops come from an upstream bug that a three-line change fixes,
then "DynaGraph is needed" cannot be argued from them; if a large pile remains after subtracting it, that is the real target.
"""
import sys, os
from collections import Counter
import torch, torch.nn as nn
import torch.nn.functional as F
import torch._inductor.config as ic, torch._dynamo.config as dc
import torch._inductor.scheduler as S

ic.force_disable_caches = True
dc.capture_dynamic_output_shape_ops = True
dc.capture_scalar_outputs = True

seen = {}
orig = S.Scheduler.should_partition


def patched(self, node, *a, **kw):
    out = orig(self, node, *a, **kw)
    if isinstance(out, str):
        try:
            n = node.get_name()
        except Exception:
            n = str(id(node))
        if n not in seen:
            ir = getattr(node, "node", None)
            org = getattr(ir, "origins", None)
            try:
                o = ",".join(sorted({str(getattr(x, "target", x)) for x in org})[:2]) if org else "-"
            except Exception:
                o = "-"
            seen[n] = (out, o)
    return out


S.Scheduler.should_partition = patched


class FixedReLU6(nn.Module):
    """Equivalent to nn.ReLU6, but the bounds are literals, not module attributes, so they do not get symbolized."""

    def forward(self, x):
        return F.hardtanh(x, 0.0, 6.0)


def swap_relu6(m):
    n = 0
    for name, child in list(m.named_children()):
        if isinstance(child, nn.ReLU6) or (
            type(child) is nn.Hardtanh
            and float(child.min_val) == 0.0 and float(child.max_val) == 6.0
        ):
            setattr(m, name, FixedReLU6())
            n += 1
        else:
            n += swap_relu6(child)
    return n


def probe(tag, build):
    seen.clear(); torch._dynamo.reset()
    m, x = build()
    try:
        c = torch.compile(m, dynamic=True, mode="reduce-overhead")
        c(x) if not isinstance(x, list) else c(x)
    except Exception as e:
        pass
    c = Counter(v[0] for v in seen.values())
    org = Counter(v[1] for v in seen.values())
    print(f"{tag:<32} non_gpu_nodes={len(seen):<4} {dict(c)}")
    for o, n in org.most_common(3):
        print(f"{'':<32}   {n:>4}  {o[:70]}")


import torchvision.models as tvm
import torchvision.models.detection as tvd

TARGETS = sys.argv[1:] or ["mobilenet_v2"]

for name in TARGETS:
    det = name.startswith("det:")
    raw = name[4:] if det else name

    def mk(raw=raw, det=det):
        if det:
            m = getattr(tvd, raw)(weights=None, weights_backbone=None).cuda().eval()
            return m, [torch.randn(3, 224, 224, device="cuda")]
        return getattr(tvm, raw)(weights=None).cuda().eval(), \
               torch.randn(2, 3, 224, 224, device="cuda")

    def orig_build():
        return mk()

    def fixed_build():
        m, x = mk()
        swap_relu6(m)
        return m, x

    print(f"\n===== {name}")
    probe("original", orig_build)
    m, _ = mk()
    print(f"  (replaceable ReLU6/Hardtanh(0,6): {swap_relu6(m)})")
    probe("literal bounds", fixed_build)
