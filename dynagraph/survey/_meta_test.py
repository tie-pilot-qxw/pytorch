"""Build on meta (initialization does no real compute), then convert to fake cuda."""
import torch, sys, os
import torch._inductor.config as ic, torch._dynamo.config as dc
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models
from instrument import install, reset, snapshot

ic.force_disable_caches = True
dc.capture_dynamic_output_shape_ops = True
dc.capture_scalar_outputs = True
install()

for spec in ("tv:vit_b_16", "tv:convnext_tiny", "tv:resnet18"):
    torch._dynamo.reset(); reset()
    try:
        # 1) build on meta: weight init is a no-op and produces no unbacked symbols
        with torch.device("meta"):
            m, a, k = models.build(spec)
        # 2) enter fake mode and swap the meta params for fake cuda
        fm = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
        with fm:
            m = m.to_empty(device="cuda")
            a = tuple(x.to("cuda") if isinstance(x, torch.Tensor) else
                      [t.to("cuda") for t in x] if isinstance(x, list) else x for x in a)
            k = {kk: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for kk, v in k.items()}
            torch.compile(m, dynamic=True, mode="reduce-overhead")(*a, **k)
        tag = "ok(ran to end)"
    except Exception as e:
        tag = type(e).__name__
    s = snapshot()
    print(f"{spec:<18} {tag:<32} compile_data={'yes' if s['n_partitions_observed'] else 'no'} "
          f"partitions={s['n_partitions_observed']} reasons={s['partition_reason_counts']}")
print("memory_allocated:", torch.cuda.memory_allocated() // 1024 // 1024, "MiB")
