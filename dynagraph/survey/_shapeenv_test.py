"""Difference between constructing with and without a ShapeEnv. Initializing models like ViT produces unbacked symbols."""
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

for name, mk in (("with ShapeEnv", lambda: FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)),
                 ("without ShapeEnv", lambda: FakeTensorMode(allow_non_fake_inputs=True))):
    for spec in ("tv:vit_b_16", "tv:resnet18"):
        torch._dynamo.reset(); reset()
        try:
            fm = mk()
            with fm:
                with models.device_ctx():
                    m, a, k = models.build(spec)
                torch.compile(m, dynamic=True, mode="reduce-overhead")(*a, **k)
            s = snapshot(); tag = "ok(ran to end)"
        except Exception as e:
            s = snapshot()
            tag = f"{type(e).__name__}"
        got = bool(s["n_partitions_observed"])
        print(f"{name:<14} {spec:<16} {tag:<32} compile_data={'yes' if got else 'no'} "
              f"partitions={s['n_partitions_observed']} reasons={s['partition_reason_counts']}")
print("memory_allocated:", torch.cuda.memory_allocated() // 1024 // 1024, "MiB")
