import os, sys, traceback
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

print("torch", torch.__version__)
shape_env = ShapeEnv()
fm = FakeTensorMode(shape_env=shape_env)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 64)
    def forward(self, x):
        return torch.relu(self.lin(x)) * 2

with fm:
    with torch.device("cuda"):
        m = M()
    x = torch.randn(8, 64, device="cuda")
    print("param is fake:", type(m.lin.weight))
    try:
        out = torch.compile(m, backend="inductor")(x)
        print("OK", type(out))
    except Exception as e:
        print("FAILED:", type(e).__name__)
        traceback.print_exc()
