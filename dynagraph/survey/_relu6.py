"""ReLU6 triggers CPU size computation. Narrow down which step does it."""
import torch, torch.nn as nn
from collections import Counter
import torch._inductor.config as ic, torch._dynamo.config as dc
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
        if n not in seen:
            ir = getattr(node, "node", None)
            org = getattr(ir, "origins", None)
            try:
                o = ",".join(sorted({str(getattr(x, "target", x)) for x in org})[:3]) if org else "-"
            except Exception:
                o = "-"
            seen[n] = (out, o)
    return out


S.Scheduler.should_partition = patched
C = 32
x = torch.randn(2, C, 56, 56, device="cuda")


def probe(tag, fn, inp=x, train=False):
    seen.clear(); torch._dynamo.reset()
    m = fn
    if isinstance(m, nn.Module):
        m = m.cuda()
        m.train() if train else m.eval()
    try:
        torch.compile(m, dynamic=True, mode="reduce-overhead")(inp)
    except Exception:
        pass
    c = Counter(v[0] for v in seen.values())
    org = Counter(v[1] for v in seen.values()).most_common(1)
    print(f"{tag:<38} nonGPU={len(seen):<3} {dict(c)}  {org[0][0][:52] if org else ''}")


probe("ReLU6 only", nn.ReLU6())
probe("clamp(0,6) only", lambda t: t.clamp(0, 6))
probe("hardtanh only", lambda t: torch.nn.functional.hardtanh(t, 0.0, 6.0))
probe("BatchNorm (eval) only", nn.BatchNorm2d(C))
probe("BatchNorm (train) only", nn.BatchNorm2d(C), train=True)
probe("BN(eval) + ReLU6", nn.Sequential(nn.BatchNorm2d(C), nn.ReLU6()))
probe("BN(train) + ReLU6", nn.Sequential(nn.BatchNorm2d(C), nn.ReLU6()), train=True)
probe("BN(eval) + clamp", nn.Sequential(nn.BatchNorm2d(C), nn.Hardtanh(0, 6)))
probe("Conv + ReLU6 (no BN)", nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.ReLU6()))

print("--- int vs float boundary ---")
probe("nn.Hardtanh(0, 6)  int", nn.Hardtanh(0, 6))
probe("nn.Hardtanh(0.0, 6.0) float", nn.Hardtanh(0.0, 6.0))
probe("clamp(0.0, 6.0) float", lambda t: t.clamp(0.0, 6.0))
probe("F.hardtanh float", lambda t: torch.nn.functional.hardtanh(t, 0.0, 6.0))
probe("F.relu6", lambda t: torch.nn.functional.relu6(t))
probe("nn.ReLU6(inplace=True)", nn.ReLU6(inplace=True))
import inspect
print("\nnn.ReLU6.__init__:", inspect.getsource(nn.ReLU6.__init__))
print("nn.ReLU6 forward comes from Hardtanh:", nn.ReLU6.forward is nn.Hardtanh.forward)
print(inspect.getsource(nn.Hardtanh.forward))
