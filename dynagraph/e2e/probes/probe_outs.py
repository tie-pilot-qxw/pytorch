import time, torch
from torch._inductor.dynagraph import _over_storage
a = torch.empty(1 << 26, dtype=torch.uint8, device="cuda")
specs = [(i * 4096, torch.bfloat16, (64, 64), (64, 1)) for i in range(916)]
for _ in range(3):
    t = time.perf_counter()
    outs = [_over_storage(a, o, d, s, st) for o, d, s, st in specs]
    dt = time.perf_counter() - t
print(f"_over_storage: {dt / len(specs) * 1e6:.2f} us each, {dt * 1e3:.2f} ms for {len(specs)}")
from torch._inductor.dynagraph import _outs_ext
ext = _outs_ext()
protos = [torch.empty(0, dtype=torch.bfloat16, device="cuda")]
args = ([s[0] for s in specs], [0] * len(specs), [2] * len(specs), [x for s in specs for x in s[2]], [x for s in specs for x in s[3]])
for _ in range(3):
    t = time.perf_counter()
    outs2 = ext.make_outputs(a, protos, *args)
    dt = time.perf_counter() - t
print(f"make_outputs: {dt / len(specs) * 1e6:.2f} us each, {dt * 1e3:.2f} ms")
assert all(x.data_ptr() == y.data_ptr() and x.shape == y.shape and x.stride() == y.stride() and x.dtype == y.dtype for x, y in zip(outs, outs2))
print("same")
