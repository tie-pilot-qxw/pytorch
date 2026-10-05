import os, torch, time, subprocess
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._guards import TracingContext
import torch._inductor.config as icfg
icfg.triton.cudagraphs = True
icfg.graph_partition = True

def dev_free():
    free, total = torch.cuda.mem_get_info()
    return free

print("cuda_init at import:", torch.cuda.is_initialized())
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(256, 512)
        self.lin2 = torch.nn.Linear(512, 256)
    def forward(self, x):
        return self.lin2(torch.relu(self.lin(x)) * 2).sum()

user_fm = FakeTensorMode(shape_env=ShapeEnv())
with user_fm:
    with torch.device("cuda"):
        m = M()
    x = torch.randn(64, 256, device="cuda")
print("after fake model build: cuda_init =", torch.cuda.is_initialized(),
      "alloc=", torch.cuda.memory_allocated(), "reserved=", torch.cuda.memory_reserved())
f0 = dev_free(); print("device free before compile: %.1f MiB" % (f0/2**20))

def compile_only(gm, example_inputs, **kw):
    from torch._inductor.compile_fx import compile_fx
    fm = TracingContext.get().fake_mode
    with fm:
        ni = [fm.from_tensor(a) if isinstance(a, FakeTensor) and a.fake_mode is not fm else a
              for a in example_inputs]
    compile_fx(gm, ni, **kw)
    def never(*a, **k): raise RuntimeError("STOP")
    return never

t=time.time()
try:
    torch.compile(m, backend=compile_only, fullgraph=True)(x)
except RuntimeError as e:
    pass
f1 = dev_free()
print("compile %.2fs" % (time.time()-t))
print("device free after compile: %.1f MiB   delta = %.1f MiB" % (f1/2**20, (f0-f1)/2**20))
print("torch alloc=%d reserved=%d" % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
