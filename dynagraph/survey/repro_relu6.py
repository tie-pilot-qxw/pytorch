#!/usr/bin/env python3
"""
Minimal repro: under dynamic=True, nn.ReLU6 produces scalar computation on the CPU that splits the CUDA graph.

Self-contained; does not depend on any other file in this project. Just run:
    python repro_relu6.py

Expected output: the CPU node count on the `nn.ReLU6` and `nn.Hardtanh(0.0, 6.0)` lines should be 0;
it is actually 4 (3 cpu ops + 1 DeviceCopy).

Root cause: under dynamic=True, **float** attributes of an nn.Module get symbolized and materialized as CPU tensors
(aten.unsqueeze + aten.cat), then device_put back to the GPU. All three conditions are required:
  1. dynamic=True
  2. the bound values come in through nn.Module attributes (not literals, not a functional call)
  3. the attribute is a float (ints do not trigger it)

nn.ReLU6 hits all three, because it is exactly Hardtanh(0.0, 6.0) and its forward reads self.min_val/self.max_val.
Consequence: any model using nn.ReLU6 gets split under mode="reduce-overhead" and gives up cudagraph.
MobileNet-V2 ends up with 71 CPU nodes as a result, 2 segments each in forward and backward, and neither segment goes into cudagraph.
"""
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._inductor.config as inductor_config
import torch._inductor.scheduler as scheduler

inductor_config.force_disable_caches = True

_seen: dict = {}
_orig = scheduler.Scheduler.should_partition


def _patched(self, node, *args, **kwargs):
    out = _orig(self, node, *args, **kwargs)
    # main's should_partition returns a reason string; older versions return bool
    if isinstance(out, str):
        try:
            name = node.get_name()
        except Exception:
            name = str(id(node))
        _seen.setdefault(name, out)
    return out


scheduler.Scheduler.should_partition = _patched

DEV = "cuda"
X = torch.randn(2, 32, 56, 56, device=DEV)


def probe(label, mod_or_fn, dynamic=True):
    _seen.clear()
    torch._dynamo.reset()
    m = mod_or_fn
    if isinstance(m, nn.Module):
        m = m.to(DEV).eval()
    try:
        torch.compile(m, dynamic=dynamic, mode="reduce-overhead")(X)
    except Exception:
        pass  # only the compile-time partition decision matters; the run phase doesn't
    counts = Counter(_seen.values())
    print(f"  dynamic={str(dynamic):<5} {label:<34} non-GPU nodes = {len(_seen):<3} {dict(counts)}")


print(f"torch {torch.__version__}\n")
print("Triggers:")
probe("nn.ReLU6()", nn.ReLU6())
probe("nn.Hardtanh(0.0, 6.0)  float", nn.Hardtanh(0.0, 6.0))

print("\nDoes not trigger (same math, written differently):")
probe("nn.Hardtanh(0, 6)  int", nn.Hardtanh(0, 6))
probe("F.hardtanh(x, 0.0, 6.0)", lambda t: F.hardtanh(t, 0.0, 6.0))
probe("F.relu6(x)", lambda t: F.relu6(t))
probe("x.clamp(0.0, 6.0)", lambda t: t.clamp(0.0, 6.0))
probe("nn.ReLU()", nn.ReLU())

print("\nWith dynamic off, none trigger:")
probe("nn.ReLU6()", nn.ReLU6(), dynamic=False)
probe("nn.Hardtanh(0.0, 6.0)", nn.Hardtanh(0.0, 6.0), dynamic=False)

print("\nConsequence on real models:")
try:
    import torchvision.models as tvm
    for name in ("mobilenet_v2", "resnet18"):
        _seen.clear()
        torch._dynamo.reset()
        m = getattr(tvm, name)(weights=None).to(DEV).eval()
        try:
            torch.compile(m, dynamic=True, mode="reduce-overhead")(
                torch.randn(2, 3, 224, 224, device=DEV))
        except Exception:
            pass
        counts = Counter(_seen.values())
        print(f"  {name:<44} non-GPU nodes = {len(_seen):<3} {dict(counts)}")
except ImportError:
    print("  (torchvision not installed, skipping)")
