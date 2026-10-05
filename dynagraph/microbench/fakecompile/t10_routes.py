import os, sys, torch, time, traceback
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.experimental.proxy_tensor import make_fx
from torch._guards import TracingContext, tracing
import torch._inductor.config as icfg
icfg.triton.cudagraphs = True
icfg.graph_partition = True
icfg.force_disable_caches = True

import torch._inductor.scheduler as S
log = {"sp": 0, "init": 0, "nodes": None}
_sp = S.Scheduler.should_partition
def sp(self, n, *a, **k):
    log["sp"] += 1; return _sp(self, n, *a, **k)
S.Scheduler.should_partition = sp
_si = S.Scheduler.__init__
def si(self, *a, **k):
    log["init"] += 1; r=_si(self,*a,**k); log["nodes"]=[n.get_name() for n in self.nodes]; return r
S.Scheduler.__init__ = si
def reset(): log.update(sp=0, init=0, nodes=None); torch._dynamo.reset()

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(256, 512); self.lin2 = torch.nn.Linear(512, 256)
    def forward(self, x):
        return self.lin2(torch.relu(self.lin(x)) * 2).sum()

ROUTE = sys.argv[1]
fm = FakeTensorMode(shape_env=ShapeEnv())
with fm:
    with torch.device("cuda"):
        m = M()
    x = torch.randn(64, 256, device="cuda")

t = time.time()
try:
    if ROUTE == "standalone_makefx":
        with fm:
            gm = make_fx(m, tracing_mode="real")(x)
        art = torch._inductor.standalone_compile(gm, [x])
        print("artifact:", type(art).__name__)

    elif ROUTE == "standalone_makefx_tracing":
        with fm:
            gm = make_fx(m, tracing_mode="real")(x)
        with tracing(TracingContext(fm)):
            art = torch._inductor.standalone_compile(gm, [x], dynamic_shapes="from_tracing_context")
        print("artifact:", type(art).__name__)

    elif ROUTE == "compile_fx_inner":
        from torch._inductor.compile_fx import compile_fx
        from torch._inductor.virtualized import V
        with fm:
            gm = make_fx(m, tracing_mode="real")(x)
        with tracing(TracingContext(fm)), V.set_fake_mode(fm):
            cfn = compile_fx(gm, [x])
        print("compiled:", type(cfn).__name__)

    elif ROUTE == "export":
        with fm:
            ep = torch.export.export(m, (x,))
        print("exported:", type(ep).__name__)
        gm = ep.module()
        print("gm fake mode of placeholder val:",
              [hex(id(n.meta["val"].fake_mode)) for n in gm.graph.nodes if n.op=="placeholder" and isinstance(n.meta.get("val"), FakeTensor)][:3],
              "user fm:", hex(id(fm)))
        art = torch._inductor.standalone_compile(gm, [x])
        print("artifact:", type(art).__name__)

    elif ROUTE == "export_aot":
        with fm:
            ep = torch.export.export(m, (x,))
        so = torch._inductor.aot_compile(ep.module(), (x,))
        print("aot_compile ->", so)
except Exception:
    traceback.print_exc()
print(f"ROUTE={ROUTE} time={time.time()-t:.2f}s sched_init={log['init']} should_partition={log['sp']} nodes={log['nodes']}")
print("torch alloc=%d reserved=%d" % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
