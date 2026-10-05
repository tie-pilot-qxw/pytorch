"""What the "host code" inside forward really is.
Use torch.cuda.set_sync_debug_mode("error") to show directly which patterns force a GPU->CPU sync.
Then use dynamo.explain + inductor counters to see how torch.compile reacts."""
import torch, torch._dynamo as dyn
from torch._dynamo.utils import counters
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_scalar_outputs = True
DEV = "cuda"

def mask_select(x):    return x[x > 0].sum()                      # filter padding / select valid boxes
def nonzero_gather(x):
    idx = (x > 0.5).nonzero(); return x.flatten()[idx[:, 0]].sum() # detection / MoE routing skeleton
def unique_count(x):   return torch.unique((x * 10).long()).sum()
def bincount_op(x):    return torch.bincount((x.abs()*5).long().flatten()).sum()
def item_branch(x):
    n = int((x > 0).sum()); return x.flatten()[:n].sum()           # Python control flow reads a GPU scalar
def static_baseline(x): return (x * 2 + 1).sum()                   # control

CASES = [("x[x>0] mask select", mask_select), ("nonzero + gather", nonzero_gather),
         ("torch.unique", unique_count), ("torch.bincount", bincount_op),
         ("int(tensor) control flow", item_branch), ("[control] fully static", static_baseline)]

print("=== 1. Does eager force a GPU->CPU sync? (using PyTorch's built-in sync detector) ===")
for name, fn in CASES:
    x = torch.randn(1024, device=DEV); fn(x); torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn(x); torch.cuda.synchronize(); verdict = "no sync"
    except Exception as e:
        verdict = "**forced sync**"
    finally:
        torch.cuda.set_sync_debug_mode("default")
    print(f"  {name:<22} {verdict}")

print("\n=== 2. How torch.compile reacts ===")
print(f"  {'pattern':<22} {'FX graphs':>9} {'graph break':>12} {'cudagraph skips':>18}  first skip reason")
print("  " + "-" * 104)
for name, fn in CASES:
    dyn.reset()
    try:
        ex = dyn.explain(fn)(torch.randn(1024, device=DEV))
        ng, nb = ex.graph_count, ex.graph_break_count
    except Exception:
        ng = nb = -1
    dyn.reset(); counters.clear()
    reason = ""
    try:
        cf = torch.compile(fn, mode="reduce-overhead", dynamic=True)
        for n in [1024, 999, 777]: cf(torch.randn(n, device=DEV))
        torch.cuda.synchronize()
    except Exception as e:
        reason = f"{type(e).__name__}: {str(e)[:40]}"
    skips = counters["inductor"].get("cudagraph_skips", 0)
    if not reason:
        rs = [k for k in counters["inductor"] if "cudagraph" in k and k != "cudagraph_skips"]
        reason = "; ".join(f"{k}={counters['inductor'][k]}" for k in rs)[:60]
    print(f"  {name:<22} {ng:>9} {nb:>12} {skips:>18}  {reason}")

print("\n=== 3. Why the host must know this number: all three happen on the CPU ===")
x = torch.randn(4096, device=DEV)
idx = (x > 0).nonzero()
n = idx.shape[0]
print(f"  nonzero returned {n} rows -- this number was computed by a kernel on the GPU")
print(f"  1) allocate torch.empty({n}) : the CUDA caching allocator runs on the CPU and needs this number to allocate")
print(f"  2) compute launch config grid=ceil({n}/128)={-(-n//128)} : the CPU computes the grid before launch")
print(f"  3) any Python branch if {n} > 0 : needs the real value")
print("  These three are the so-called \"host code inside forward\" -- the author wrote pure torch; the host work is injected by PyTorch itself.")
