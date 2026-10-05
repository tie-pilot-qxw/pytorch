import os, sys, torch, time
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._guards import TracingContext
import torch._inductor.config as icfg
icfg.triton.cudagraphs = True
icfg.graph_partition = True
icfg.force_disable_caches = True     # force real scheduling every time

import torch._inductor.scheduler as S
log = {"sp": [], "init": 0, "nodes": None}
_sp = S.Scheduler.should_partition
def sp(self, n, *a, **k):
    r = _sp(self, n, *a, **k); log["sp"].append((n.get_name(), r)); return r
S.Scheduler.should_partition = sp
_si = S.Scheduler.__init__
def si(self, *a, **k):
    log["init"] += 1; r = _si(self, *a, **k); log["nodes"] = [n.get_name() for n in self.nodes]; return r
S.Scheduler.__init__ = si

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(256, 512); self.lin2 = torch.nn.Linear(512, 256)
    def forward(self, x):
        return self.lin2(torch.relu(self.lin(x)) * 2).sum()

fm = FakeTensorMode(shape_env=ShapeEnv())
with fm:
    with torch.device("cuda"):
        m = M()
    x = torch.randn(64, 256, device="cuda")

def backend(gm, example_inputs, **kw):
    from torch._inductor.compile_fx import compile_fx
    tfm = TracingContext.get().fake_mode
    with tfm:
        ni = [tfm.from_tensor(a) if isinstance(a, FakeTensor) and a.fake_mode is not tfm else a for a in example_inputs]
    compile_fx(gm, ni, **kw)
    def never(*a, **k): raise RuntimeError("STOP")
    return never

print("PHASE=pre", flush=True); time.sleep(6)
t=time.time()
try:
    torch.compile(m, backend=backend, fullgraph=True)(x)
except RuntimeError:
    pass
print("compile %.2fs sched_init=%d nodes=%s" % (time.time()-t, log["init"], log["nodes"]), flush=True)
print("should_partition:", log["sp"], flush=True)
print("torch alloc=%d reserved=%d" % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()), flush=True)
print("PHASE=post", flush=True); time.sleep(8)
