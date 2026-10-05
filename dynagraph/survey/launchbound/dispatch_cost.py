# CPU-ONLY. Measures the host-side per-op cost of eager PyTorch, with CUDA fully disabled.
# Tiny CPU tensors => arithmetic is ~free; what remains is Python + dispatcher + autograd + allocator.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time, torch
torch.set_num_threads(1)
assert not torch.cuda.is_available(), "CUDA must be off"
print("torch", torch.__version__)

def bench(fn, n=200000, warm=20000):
    for _ in range(warm): fn()
    t0 = time.perf_counter_ns()
    for _ in range(n): fn()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / n / 1000.0   # us

a = torch.randn(8, 8)
b = torch.randn(8, 8)
out = torch.empty(8, 8)
ag = torch.randn(8, 8, requires_grad=True)
bg = torch.randn(8, 8, requires_grad=True)

rows = []
rows.append(("python loop overhead (pass)",      bench(lambda: None)))
rows.append(("torch.add  no-grad, alloc out",    bench(lambda: torch.add(a, b))))
rows.append(("torch.add  no-grad, out= (no alloc)", bench(lambda: torch.add(a, b, out=out))))
rows.append(("torch.relu no-grad",               bench(lambda: torch.relu(a))))
rows.append(("a.add_(b)  in-place no-grad",      bench(lambda: a.add_(b))))
rows.append(("torch.add  requires_grad (autograd on)", bench(lambda: torch.add(ag, bg), n=100000, warm=10000)))
rows.append(("torch.relu requires_grad",         bench(lambda: torch.relu(ag), n=100000, warm=10000)))
rows.append(("torch.mm 8x8 no-grad",             bench(lambda: torch.mm(a, b))))
rows.append(("torch.empty(8,8) alloc only",      bench(lambda: torch.empty(8, 8))))

with torch.no_grad():
    rows.append(("torch.add under no_grad",      bench(lambda: torch.add(ag, bg), n=100000, warm=10000)))

for name, us in rows:
    print(f"{name:45s} {us:8.3f} us")

# split: python-arg-parsing vs C++ dispatch, via the raw aten op
op = torch.ops.aten.add.Tensor
print()
print(f"{'torch.ops.aten.add.Tensor (OpOverload)':45s} {bench(lambda: op(a,b)):8.3f} us")
