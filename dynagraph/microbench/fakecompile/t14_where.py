import os, sys, torch, time
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import compile_fx
import torch._inductor.config as icfg
icfg.triton.cudagraphs = True; icfg.graph_partition = True; icfg.force_disable_caches = True
MODE = sys.argv[1]   # full | codegen_only

class Stop(Exception):
    pass
if MODE == "codegen_only":
    import torch._inductor.graph as G
    def ctm(self):
        self.codegen()
        raise Stop()
    G.GraphLowering.compile_to_module = ctm

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(256,512); self.lin2 = torch.nn.Linear(512,256)
    def forward(self,x): return self.lin2(torch.relu(self.lin(x))*2).sum()

fm = FakeTensorMode(shape_env=ShapeEnv())
with fm:
    with torch.device("cuda"):
        m = M()
    x = torch.randn(64,256,device="cuda")
    gm = make_fx(m, tracing_mode="real")(x)
print("PHASE=pre", os.getpid(), flush=True); time.sleep(6)
try:
    compile_fx(gm, [x])
    print("compiled", flush=True)
except Exception as e:
    c=e
    while c is not None and not isinstance(c, Stop):
        c = c.__cause__ or c.__context__
    print("stopped after codegen" if c is not None else "ERR %r" % e, flush=True)
print("PHASE=post alloc=%d reserved=%d" % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()), flush=True)
time.sleep(10)
