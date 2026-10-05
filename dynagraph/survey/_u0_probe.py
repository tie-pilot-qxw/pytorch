"""Only construct ViT, do not compile, to see whether the unbacked symbols are produced at construction time."""
import torch, sys, os
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models

for spec in ("tv:resnet18", "tv:vit_b_16"):
    env = ShapeEnv()
    fm = FakeTensorMode(shape_env=env, allow_non_fake_inputs=True)
    with fm, models.device_ctx():
        m, a, k = models.build(spec)
    n_unbacked = len(getattr(env, "var_to_range", {}))
    unbacked = [str(s) for s in getattr(env, "unbacked_symbol_counter", [])] \
        if not hasattr(env, "var_to_val") else \
        [str(s) for s in env.var_to_range if str(s).startswith("u")]
    print(f"{spec:<16} symbols in ShapeEnv after construction: total {n_unbacked}, unbacked={unbacked[:6]}")
