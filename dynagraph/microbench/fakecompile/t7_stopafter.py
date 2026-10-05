import os, sys, torch, time, traceback
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._guards import TracingContext
import torch._inductor.config as icfg
icfg.triton.cudagraphs = True
icfg.graph_partition = True

MODE = sys.argv[1]  # "full" | "codegen_only"

class StopAfterScheduling(Exception):
    def __init__(self, src): self.src = src

if MODE == "codegen_only":
    import torch._inductor.graph as G
    def compile_to_module(self):
        wrapper_code, kernel_code = self.codegen()   # runs Scheduler + codegen, no cubin
        raise StopAfterScheduling(wrapper_code.value if hasattr(wrapper_code,'value') else wrapper_code)
    G.GraphLowering.compile_to_module = compile_to_module

import torch._inductor.scheduler as S
log = {"sp": 0, "init": 0, "nodes": None}
_sp = S.Scheduler.should_partition
def sp(self, n, *a, **k):
    log["sp"] += 1; return _sp(self, n, *a, **k)
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

torch.cuda.init()
f0 = torch.cuda.mem_get_info()[0]
src_box = {}
def backend(gm, example_inputs, **kw):
    from torch._inductor.compile_fx import compile_fx
    tfm = TracingContext.get().fake_mode
    with tfm:
        ni = [tfm.from_tensor(a) if isinstance(a, FakeTensor) and a.fake_mode is not tfm else a for a in example_inputs]
    try:
        compile_fx(gm, ni, **kw)
    except Exception as e:
        # unwrap InductorError / BackendCompilerFailed
        cur = e
        while cur is not None:
            if isinstance(cur, StopAfterScheduling):
                src_box["src"] = cur.src; break
            cur = cur.__cause__ or cur.__context__
        else:
            raise
    def never(*a, **k): raise RuntimeError("STOP")
    return never

t=time.time()
try:
    torch.compile(m, backend=backend, fullgraph=True)(x)
except RuntimeError as e:
    print("stopped:", str(e)[:120])
except Exception:
    traceback.print_exc()
f1 = torch.cuda.mem_get_info()[0]
print(f"MODE={MODE} time={time.time()-t:.2f}s sched_init={log['init']} should_partition={log['sp']} nodes={log['nodes']}")
print(f"device mem delta = {(f0-f1)/2**20:.1f} MiB ; torch alloc={torch.cuda.memory_allocated()} reserved={torch.cuda.memory_reserved()}")
if src_box.get("src"):
    print("---- generated wrapper (first 25 lines) ----")
    print("\n".join(str(src_box["src"]).splitlines()[:25]))
