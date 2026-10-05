import os, sys, torch, traceback, subprocess, glob, time
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._guards import TracingContext
import torch._inductor.config as icfg

icfg.triton.cudagraphs = True     # make should_partition do real work
icfg.graph_partition = True
icfg.fallback_random = True

def gpu_mem(tag):
    pid = os.getpid()
    out = subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    mine = [l for l in out.splitlines() if l.strip().startswith(str(pid))]
    print(f"[mem {tag}] nvidia-smi_for_pid={mine or '<none>'} "
          f"torch_alloc={torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 'n/a'} "
          f"torch_reserved={torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 'n/a'} "
          f"cuda_init={torch.cuda.is_initialized()}")

import torch._inductor.scheduler as S
log = {"should_partition": [], "scheduler_init": 0}
_sp = S.Scheduler.should_partition
def sp(self, node, *a, **k):
    r = _sp(self, node, *a, **k)
    log["should_partition"].append((node.get_name() if hasattr(node,'get_name') else str(node), r))
    return r
S.Scheduler.should_partition = sp
_si = S.Scheduler.__init__
def si(self, *a, **k):
    log["scheduler_init"] += 1
    return _si(self, *a, **k)
S.Scheduler.__init__ = si

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 128)
        self.lin2 = torch.nn.Linear(128, 64)
    def forward(self, x):
        y = torch.relu(self.lin(x)) * 2
        return self.lin2(y).sum()

compiled_box = {}
def compile_only_inductor(gm, example_inputs, **kw):
    from torch._inductor.compile_fx import compile_fx
    fm = TracingContext.get().fake_mode
    with fm:
        new_inputs = [fm.from_tensor(x) if isinstance(x, FakeTensor) and x.fake_mode is not fm else x
                      for x in example_inputs]
    compiled_box["fn"] = compile_fx(gm, new_inputs, **kw)
    # NEVER return the real compiled fn -> nothing ever executes on the GPU.
    def never_run(*args, **kwargs):
        raise RuntimeError("compile-only: refusing to execute")
    return never_run

user_fm = FakeTensorMode(shape_env=ShapeEnv())
with user_fm:
    with torch.device("cuda"):
        m = M()
    x = torch.randn(8, 64, device="cuda")

gpu_mem("before")
t0=time.time()
try:
    torch.compile(m, backend=compile_only_inductor, fullgraph=True)(x)
except RuntimeError as e:
    print("expected stop:", e)
except Exception:
    traceback.print_exc()
print("compile wall time %.2fs" % (time.time()-t0))
print("scheduler_init:", log["scheduler_init"])
print("should_partition (%d calls):" % len(log["should_partition"]))
for n, r in log["should_partition"]:
    print("   ", n, "->", r)
print("compiled artifact:", type(compiled_box.get("fn")))
gpu_mem("after")
