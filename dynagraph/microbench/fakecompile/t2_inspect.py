import torch, traceback
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._guards import TracingContext, tracing, detect_fake_mode

shape_env = ShapeEnv()
fm = FakeTensorMode(shape_env=shape_env)
print("USER fm", hex(id(fm)))

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 64)
    def forward(self, x):
        return torch.relu(self.lin(x)) * 2

def backend(gm, example_inputs):
    tc = TracingContext.try_get()
    print("  backend: TracingContext.fake_mode =", hex(id(tc.fake_mode)) if tc and tc.fake_mode else None)
    for i, a in enumerate(example_inputs):
        print("   input", i, type(a).__name__, getattr(a, "shape", None),
              hex(id(a.fake_mode)) if isinstance(a, torch._subclasses.fake_tensor.FakeTensor) else "REAL/other")
    print("  graph:"); gm.graph.print_tabular()
    return gm.forward

with fm:
    with torch.device("cuda"):
        m = M()
    x = torch.randn(8, 64, device="cuda")
    print("== plain ==")
    try:
        torch.compile(m, backend=backend, fullgraph=True)(x)
    except Exception as e:
        print("  ERR", type(e).__name__, str(e)[:300])
    torch._dynamo.reset()
    print("== wrapped in tracing(TracingContext(fm)) ==")
    try:
        with tracing(TracingContext(fm)):
            torch.compile(m, backend=backend, fullgraph=True)(x)
    except Exception as e:
        print("  ERR", type(e).__name__, str(e)[:500])
