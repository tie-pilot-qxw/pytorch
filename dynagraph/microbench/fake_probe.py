# Can fake tensors carry us all the way to the Inductor scheduler's should_partition?
import os, sys, traceback
import torch, torch.nn as nn

REASONS = []
def hook():
    import torch._inductor.scheduler as S
    orig = S.Scheduler.should_partition
    def patched(self, node, should_log=False):
        r = orig(self, node, should_log=should_log)
        REASONS.append((type(node).__name__, bool(r)))
        return r
    S.Scheduler.should_partition = patched
hook()

class M(nn.Module):
    def __init__(s):
        super().__init__(); s.l1=nn.Linear(256,256); s.l2=nn.Linear(256,256)
    def forward(s,x):
        y = s.l1(x).relu()
        n = (y>0).sum()                 # data-dependent -> unbacked symint
        return s.l2(y) * n

mode = sys.argv[1] if len(sys.argv)>1 else "fake"
print(f"=== mode={mode}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

if mode == "fake":
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    fm = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
    with fm, torch.device("cuda"):
        m = M()
        x = torch.randn(64,256)
        try:
            torch.compile(m, dynamic=True)(x)
            print("compile OK under FakeTensorMode")
        except Exception as e:
            print("FAILED:", type(e).__name__, str(e)[:400])
            traceback.print_exc(limit=6)
else:
    m = M().cuda(); x = torch.randn(64,256, device="cuda")
    torch.compile(m, dynamic=True)(x)
    print("compile OK (real)")

print(f"should_partition called {len(REASONS)} times, True {sum(r for _,r in REASONS)} times")
print("mem alloc:", torch.cuda.memory_allocated()//1024, "KiB" if torch.cuda.is_initialized() else "(cuda not init)")
