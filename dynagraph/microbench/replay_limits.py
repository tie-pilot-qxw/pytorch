"""Why not just capture once and replay? -- see when that breaks down."""
import torch
DEV = "cuda"
s = torch.cuda.Stream()

def build(x, y, out):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        torch.add(x, y, out=out)          # warm-up
    torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=s):
        torch.add(x, y, out=out)
    return g

print("=== Failure mode 1: same shape, but the tensor addresses changed ===")
x = torch.ones(8, device=DEV); y = torch.ones(8, device=DEV) * 10; out = torch.zeros(8, device=DEV)
g = build(x, y, out)
g.replay(); torch.cuda.synchronize()
print(f"  inputs from capture time -> {out[0].item():.0f}  (expected 11)")
x2 = torch.ones(8, device=DEV) * 100          # new tensor, new address
y2 = torch.ones(8, device=DEV) * 1000
print(f"  new input has the same address as the old one: {x2.data_ptr() == x.data_ptr()}")
g.replay(); torch.cuda.synchronize()
print(f"  replay after new inputs -> {out[0].item():.0f}  (would be 1100 if replay worked)")
print(f"  => the graph has the two pointers from capture time baked in; it cannot see the new tensors at all\n")

print("=== Failure mode 2: the shape changed ===")
x3 = torch.ones(64, device=DEV); y3 = torch.ones(64, device=DEV) * 10
out3 = torch.zeros(64, device=DEV)
g3 = build(x3, y3, out3)
g3.replay(); torch.cuda.synchronize()
print(f"  elements computed correctly at n=64: {(out3 == 11).sum().item()} / 64")
# now try to use the n=64 graph to compute n=256
x4 = torch.ones(256, device=DEV); y4 = torch.ones(256, device=DEV)*10; out4 = torch.zeros(256, device=DEV)
try:
    x3.copy_(x4[:64])                       # can only move data into the old buffer; the size cannot change
    g3.replay(); torch.cuda.synchronize()
    print(f"  trying n=256: numel and grid in the graph are baked in for 64, so only the first 64 get computed")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
print(f"  => this is why cudagraph trees must re-record for every new shape\n")

print("=== So how does PyTorch handle it today ===")
print("  cudagraph trees uses two tricks:")
print("   1) copy inputs into the fixed addresses from capture time (fixes mode 1, at the cost of one copy per step)")
print("   2) re-record a graph for every new shape (fixes mode 2, cost shown below)")
import time
sizes = list(range(1000, 1000+40))            # 40 distinct shapes, simulating a variable-length workload
@torch.compile(mode="reduce-overhead", dynamic=False)
def f(a, b): return (a + b).sum()
torch.cuda.synchronize(); t0 = time.perf_counter()
for n in sizes:
    a = torch.ones(n, device=DEV); b = torch.ones(n, device=DEV)
    f(a, b)
torch.cuda.synchronize(); t1 = time.perf_counter()
# second round: the same shapes all hit
t2 = time.perf_counter()
for n in sizes:
    a = torch.ones(n, device=DEV); b = torch.ones(n, device=DEV)
    f(a, b)
torch.cuda.synchronize(); t3 = time.perf_counter()
print(f"  40 distinct shapes, first run (compile + record): {(t1-t0):.2f} s")
print(f"  same 40 shapes, second run (all hits)         : {(t3-t2)*1e3:.1f} ms")
print(f"  => Hits are fast. The problem is that in a variable-length workload the next shape is usually new, so you always pay the first-line cost")
