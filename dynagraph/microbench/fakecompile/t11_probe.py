import sys, torch, traceback
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.experimental.proxy_tensor import make_fx
from torch._guards import TracingContext, tracing
from torch._inductor.virtualized import V
import torch._inductor.config as icfg
icfg.force_disable_caches = True

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 64)
    def forward(self, x): return torch.relu(self.lin(x)).sum()

fm = FakeTensorMode(shape_env=ShapeEnv())
print("user fm", hex(id(fm)))
with fm:
    with torch.device("cuda"):
        m = M()
    x = torch.randn(8, 64, device="cuda")
    gm = make_fx(m, tracing_mode="real")(x)

print("gm params:", {k: hex(id(v.fake_mode)) for k,v in gm.named_parameters()} or "none")
print("gm buffers:", {k: hex(id(v.fake_mode)) for k,v in gm.named_buffers()} or "none")
attrs = {k: hex(id(v.fake_mode)) for k,v in gm.__dict__.items() if isinstance(v, FakeTensor)}
print("gm tensor attrs:", attrs)
import copy
gm2 = copy.deepcopy(gm)
attrs2 = {k: hex(id(v.fake_mode)) for k,v in gm2.__dict__.items() if isinstance(v, FakeTensor)}
print("after deepcopy:", attrs2)
p2 = {k: hex(id(v.fake_mode)) for k,v in gm2.named_parameters()}
print("after deepcopy params:", p2)
