"""Wrap-up check for two things:
1) Under dynamic=True, does Inductor regenerate Triton kernels when the shape changes? (If it does, the main premise of tier-1 collapses)
2) How much does the "ask" path cost per call?"""
import os, glob, time, torch
CACHE=os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "closure_triton"); os.environ["TRITON_CACHE_DIR"]=CACHE
import shutil; shutil.rmtree(CACHE, ignore_errors=True)
DEV="cuda"

def count_kernels():
    return len({os.path.basename(f) for f in glob.glob(CACHE+"/**/*.cubin", recursive=True)})

class Blk(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.n=torch.nn.LayerNorm(256); s.f=torch.nn.Linear(256,1024); s.g=torch.nn.Linear(1024,256)
    def forward(s,x): return s.g(torch.nn.functional.gelu(s.f(s.n(x))))*0.5 + x

m = Blk().cuda().half()
cm = torch.compile(m, dynamic=True)
print("=== 1. Inductor dynamic=True: does changing the shape regenerate kernels? ===")
prev=0
for n in [128, 129, 333, 1024, 4097, 16000, 65537]:
    x = torch.randn(n,256,device=DEV,dtype=torch.float16)
    cm(x); torch.cuda.synchronize()
    cur = count_kernels()
    print(f"  n={n:>6}  cubins generated so far {cur:>3}   new this call {cur-prev}")
    prev = cur
print(f"  => If only the first call adds any, one set of kernels covers all shapes and the tier-1 premise holds")

print("\n=== 2. torch.compile's own host overhead (as a reference for the cost of \"asking\") ===")
x = torch.randn(4096,256,device=DEV,dtype=torch.float16)
for _ in range(5): cm(x)
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(200): cm(x)
torch.cuda.synchronize(); t1=time.perf_counter()
print(f"  Repeated calls, same shape: {(t1-t0)/200*1e6:.1f} us/call (incl. python + guard + launch)")
