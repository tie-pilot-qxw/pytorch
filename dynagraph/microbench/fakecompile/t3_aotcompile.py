import os, sys, torch, traceback
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

def gpu_mem():
    import subprocess
    pid = os.getpid()
    out = subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    mine = [l for l in out.splitlines() if l.strip().startswith(str(pid))]
    return mine or ["<none>"]

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 64)
    def forward(self, x):
        return torch.relu(self.lin(x)) * 2

# Track whether the Scheduler ran and whether any kernel launched
sched_calls = []
import torch._inductor.scheduler as S
orig_sp = S.Scheduler.should_partition
def patched_sp(self, node, *a, **k):
    r = orig_sp(self, node, *a, **k)
    sched_calls.append((type(node).__name__, r))
    return r
S.Scheduler.should_partition = patched_sp

fm = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
with fm:
    with torch.device("cuda"):
        m = M()
    x = torch.randn(8, 64, device="cuda")
print("built fake model. mem:", gpu_mem())

try:
    torch._dynamo.config.enable_aot_compile = True
    compiled = torch.compile(m.forward, fullgraph=True).aot_compile(((x,), {}))
    print("aot_compile OK ->", type(compiled))
except Exception:
    traceback.print_exc()
print("should_partition calls:", len(sched_calls), sched_calls[:10])
print("mem after:", gpu_mem())
print("torch.cuda.memory_allocated:", torch.cuda.memory_allocated() if torch.cuda.is_initialized() else "cuda not init")
print("cuda initialized:", torch.cuda.is_initialized())
