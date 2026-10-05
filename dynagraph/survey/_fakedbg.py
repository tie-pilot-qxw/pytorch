import torch, torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

class M(nn.Module):
    def __init__(s):
        super().__init__(); s.a = nn.Linear(256, 256)
    def forward(s, x): return s.a(x).relu()

print("--- variant 1: with fm, torch.device('cuda') ---")
fm = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
with fm, torch.device("cuda"):
    m = M(); x = torch.randn(64, 256)
    print("  param device:", {n: str(p.device) for n, p in m.named_parameters()})
    print("  input device:", x.device, type(x).__name__)

print("--- variant 2: build the model on real cuda, then fakify ---")
m2 = M().cuda()
fm2 = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
with fm2:
    fx = fm2.from_tensor(torch.randn(64, 256, device="cuda"))
    print("  param device:", {n: str(p.device) for n, p in m2.named_parameters()})
    print("  input device:", fx.device, type(fx).__name__)
    try:
        torch.compile(m2, dynamic=True, mode="reduce-overhead")(fx)
        print("  compile OK")
    except Exception as e:
        print("  compile failed:", type(e).__name__, str(e)[:200])
print("  memory_allocated:", torch.cuda.memory_allocated() // 1024, "KiB")
