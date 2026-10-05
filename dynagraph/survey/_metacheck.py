"""Is the meta -> fake cuda conversion clean? Tensors left on meta turn into DeviceCopy in the graph."""
import torch, sys, os
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models

def _to(o, dev):
    if isinstance(o, torch.Tensor): return o.to(dev)
    if isinstance(o, (list, tuple)): return type(o)(_to(x, dev) for x in o)
    if isinstance(o, dict): return {k: _to(v, dev) for k, v in o.items()}
    return o

for spec in ("tv:mobilenet_v2", "tv:resnet18"):
    with torch.device("meta"):
        m, a, k = models.build(spec)
    fm = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
    with fm:
        m = m.to_empty(device="cuda")
        a = _to(a, "cuda"); k = _to(k, "cuda")
        bad_p = [n for n, p in m.named_parameters() if p.device.type != "cuda"]
        bad_b = [n for n, b in m.named_buffers() if b.device.type != "cuda"]
        bad_a = [i for i, x in enumerate(a)
                 if isinstance(x, torch.Tensor) and x.device.type != "cuda"]
        print(f"{spec:<20} params not on cuda: {len(bad_p)} {bad_p[:3]}  "
              f"buffers not on cuda: {len(bad_b)} {bad_b[:3]}  inputs: {bad_a}")
        # After to_empty the weights are uninitialized garbage; that does not affect graph-structure stats, but check there is no NaN crash risk
        print(f"{'':<20} n_params={sum(1 for _ in m.parameters())} "
              f"n_buffers={sum(1 for _ in m.buffers())}")
