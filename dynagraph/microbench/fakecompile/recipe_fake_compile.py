"""
VERIFIED RECIPE: run torch.compile(model)(x) end-to-end on FakeTensors
(Dynamo -> AOTAutograd -> Inductor GraphLowering -> Inductor Scheduler),
with zero torch GPU allocations and no kernel launches.

torch 2.11.0a0+eb65b36914.nv26.02
"""
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._guards import TracingContext
import torch._inductor.config as icfg

# keep every compile-time benchmark/autotune path off (these are the only
# things in inductor that would really launch kernels at compile time)
icfg.max_autotune = False
icfg.max_autotune_gemm = False
icfg.max_autotune_pointwise = False
icfg.benchmark_kernel = False
icfg.benchmark_fusion = False
icfg.coordinate_descent_tuning = False
icfg.triton.autotune_at_compile_time = False
# turn ON the things we actually want to exercise
icfg.triton.cudagraphs = True
icfg.graph_partition = True
icfg.force_disable_caches = True   # so the Scheduler really runs every time


def fake_inductor(gm, example_inputs, **kwargs):
    """Dynamo backend = inductor, with the cross-fake-mode fix that upstream
    main does inside AOTAutograd's process_inputs()."""
    from torch._inductor.compile_fx import compile_fx
    tfm = TracingContext.get().fake_mode          # Dynamo's backend FakeTensorMode
    with tfm:
        inputs = [
            tfm.from_tensor(a)
            if isinstance(a, FakeTensor) and a.fake_mode is not tfm else a
            for a in example_inputs
        ]
    compile_fx(gm, inputs, **kwargs)              # full lowering + Scheduler
    # return a fake-executing callable so nothing ever runs on the GPU
    return gm.forward


# ---------------------------------------------------------------- instrument
import torch._inductor.scheduler as S
seen = []
_orig = S.Scheduler.should_partition
S.Scheduler.should_partition = lambda self, n, *a, **k: (
    seen.append((n.get_name(), _orig(self, n, *a, **k))), seen[-1][1])[1]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(256, 512)
        self.l2 = torch.nn.Linear(512, 256)
    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)) * 2).sum()


fm = FakeTensorMode(shape_env=ShapeEnv())
with fm:                                    # build params + inputs as FakeTensors
    with torch.device("cuda"):
        model = Net()
    x = torch.randn(64, 256, device="cuda", requires_grad=True)

    out = torch.compile(model, backend=fake_inductor, fullgraph=True)(x)

print("out:", type(out).__name__, out.shape, out.device,
      "same fake mode as caller:", isinstance(out, FakeTensor) and out.fake_mode is fm)
print("Scheduler.should_partition calls:", len(seen), "->", seen[:6])
print("torch.cuda.memory_allocated:", torch.cuda.memory_allocated())
print("torch.cuda.memory_reserved :", torch.cuda.memory_reserved())
