import sys, torch, time, traceback
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.experimental.proxy_tensor import make_fx
from torch._guards import TracingContext, tracing
from torch._inductor.virtualized import V
import torch._inductor.config as icfg
icfg.triton.cudagraphs = True; icfg.graph_partition = True; icfg.force_disable_caches = True

import torch._inductor.scheduler as S
log = {"sp":0,"init":0,"nodes":None}
_sp=S.Scheduler.should_partition
S.Scheduler.should_partition=lambda self,n,*a,**k:(log.__setitem__("sp",log["sp"]+1),_sp(self,n,*a,**k))[1]
_si=S.Scheduler.__init__
def si(self,*a,**k):
    log["init"]+=1; r=_si(self,*a,**k); log["nodes"]=[n.get_name() for n in self.nodes]; return r
S.Scheduler.__init__=si

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(256, 512); self.lin2 = torch.nn.Linear(512,256)
    def forward(self, x): return self.lin2(torch.relu(self.lin(x))*2).sum()

R = sys.argv[1]
fm = FakeTensorMode(shape_env=ShapeEnv())
with fm:
    with torch.device("cuda"):
        m = M()
    x = torch.randn(64, 256, device="cuda")

t=time.time()
try:
    if R == "compile_fx_no_V":
        from torch._inductor.compile_fx import compile_fx
        with fm: gm = make_fx(m, tracing_mode="real")(x)
        with tracing(TracingContext(fm)):
            compile_fx(gm, [x])
        print("OK (no V.set_fake_mode)")
    elif R == "compile_fx_no_tracing":
        from torch._inductor.compile_fx import compile_fx
        with fm: gm = make_fx(m, tracing_mode="real")(x)
        compile_fx(gm, [x])
        print("OK (no tracing ctx at all)")
    elif R == "export_own_fakes":
        from torch._inductor.compile_fx import compile_fx
        with fm: ep = torch.export.export(m, (x,))
        gm = ep.module()
        vals = [n.meta["val"] for n in gm.graph.nodes if n.op=="placeholder"]
        efm = vals[0].fake_mode
        print("export fm", hex(id(efm)), "n placeholders", len(vals))
        with tracing(TracingContext(efm)), V.set_fake_mode(efm):
            compile_fx(gm, vals)
        print("OK export->compile_fx")
    elif R == "export_user_inputs":
        from torch._inductor.compile_fx import compile_fx
        with fm: ep = torch.export.export(m, (x,))
        gm = ep.module()
        print("params fm:", {k:hex(id(v.fake_mode)) for k,v in list(gm.named_parameters())[:2]}, "user fm", hex(id(fm)))
        compile_fx(gm, [x])
        print("OK export.module() + user fake inputs")
    elif R == "export_aot_own_fakes":
        with fm: ep = torch.export.export(m, (x,))
        gm = ep.module()
        vals = [n.meta["val"] for n in gm.graph.nodes if n.op=="placeholder"]
        efm = vals[0].fake_mode
        with tracing(TracingContext(efm)), V.set_fake_mode(efm):
            so = torch._inductor.aot_compile(gm, tuple(vals))
        print("aot_compile ->", so)
except Exception:
    traceback.print_exc()
print(f"R={R} {time.time()-t:.2f}s sched_init={log['init']} sp={log['sp']} nodes={log['nodes']} alloc={torch.cuda.memory_allocated()}")
