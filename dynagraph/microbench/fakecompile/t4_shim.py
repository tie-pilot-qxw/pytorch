import os, sys, torch, traceback, subprocess
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._guards import TracingContext

def gpu_mem(tag):
    pid = os.getpid()
    out = subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    mine = [l for l in out.splitlines() if l.strip().startswith(str(pid))]
    print(f"[mem {tag}] nvidia-smi={mine or '<none>'} "
          f"torch_alloc={torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 'n/a'} "
          f"cuda_init={torch.cuda.is_initialized()}")

# --- instrumentation: did the Scheduler run? did any kernel launch? ---
import torch._inductor.scheduler as S
calls = {"should_partition":0, "scheduler_init":0, "codegen":0}
_sp = S.Scheduler.should_partition
def sp(self, node, *a, **k):
    calls["should_partition"] += 1
    return _sp(self, node, *a, **k)
S.Scheduler.should_partition = sp
_si = S.Scheduler.__init__
def si(self, *a, **k):
    calls["scheduler_init"] += 1
    return _si(self, *a, **k)
S.Scheduler.__init__ = si

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 128)
        self.lin2 = torch.nn.Linear(128, 64)
    def forward(self, x):
        return self.lin2(torch.relu(self.lin(x)) * 2).sum()

def refakify_inductor(gm, example_inputs, **kw):
    """Emulates torch/_functorch/_aot_autograd/frontend_utils.py (main HEAD):
    clone example inputs from a foreign fake mode into the tracing fake mode."""
    from torch._inductor.compile_fx import compile_fx
    fm = TracingContext.get().fake_mode
    with fm:
        new_inputs = [
            fm.from_tensor(x) if isinstance(x, FakeTensor) and x.fake_mode is not fm else x
            for x in example_inputs
        ]
    return compile_fx(gm, new_inputs, **kw)

user_fm = FakeTensorMode(shape_env=ShapeEnv())
with user_fm:
    with torch.device("cuda"):
        m = M()
    x = torch.randn(8, 64, device="cuda")

gpu_mem("before compile")
try:
    out = torch.compile(m, backend=refakify_inductor, fullgraph=True)(x)
    print("RESULT:", type(out).__name__, getattr(out, "shape", None),
          "fake_mode=", hex(id(out.fake_mode)) if isinstance(out, FakeTensor) else None,
          "user_fm=", hex(id(user_fm)))
except Exception:
    traceback.print_exc()
print("instrumentation:", calls)
gpu_mem("after compile")
