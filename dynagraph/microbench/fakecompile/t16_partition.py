import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._guards import TracingContext
import torch._inductor.config as icfg
icfg.triton.cudagraphs = True; icfg.graph_partition = True; icfg.force_disable_caches = True

import torch._inductor.scheduler as S
seen=[]; _o=S.Scheduler.should_partition
S.Scheduler.should_partition = lambda self,n,*a,**k:(seen.append((n.get_name(), _o(self,n,*a,**k))), seen[-1][1])[1]

def fake_inductor(gm, ei, **kw):
    from torch._inductor.compile_fx import compile_fx
    tfm = TracingContext.get().fake_mode
    with tfm:
        ins=[tfm.from_tensor(a) if isinstance(a,FakeTensor) and a.fake_mode is not tfm else a for a in ei]
    compile_fx(gm, ins, **kw)
    return gm.forward

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(256,256)
    def forward(self, x):
        y = torch.relu(self.l1(x))
        z = y.cpu() + 1          # forces a DeviceCopy -> should_partition True
        return (y.sum(), z.sum())

fm = FakeTensorMode(shape_env=ShapeEnv())
with fm:
    with torch.device("cuda"):
        m = Net()
    x = torch.randn(64,256, device="cuda")
    out = torch.compile(m, backend=fake_inductor, fullgraph=True)(x)
print("out:", [type(o).__name__ for o in out])
print("should_partition:", seen)
print("alloc", torch.cuda.memory_allocated())
