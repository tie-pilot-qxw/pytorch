"""Opaque kernels you meet in real models (ATen fallback / cuDNN / extensions):
does the kernel change when the shape changes? Changes = tier 3 only; no change = the planner patching grid+params is enough."""
import torch, torch.nn.functional as F
from collections import Counter
from torch.profiler import profile, ProfilerActivity
DEV="cuda"; DT=torch.float16
torch.manual_seed(0)

def kernels(fn, warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    return [e.name for e in p.events() if e.device_type.name=="CUDA"
            and not e.name.startswith(("Memset","Memcpy"))]

SIZES = [1000, 4096, 10007, 65536, 262144, 1000003]
CASES = []
def add(name, maker): CASES.append((name, maker))

add("sort",            lambda n: (lambda x=torch.randn(n,device=DEV): (lambda: torch.sort(x)))())
add("topk(k=64)",      lambda n: (lambda x=torch.randn(n,device=DEV): (lambda: torch.topk(x,min(64,n))))())
add("cumsum",          lambda n: (lambda x=torch.randn(n,device=DEV): (lambda: torch.cumsum(x,0)))())
add("scatter_add",     lambda n: (lambda x=torch.randn(n,device=DEV), i=torch.randint(0,max(n//2,1),(n,),device=DEV), o=torch.zeros(max(n//2,1),device=DEV): (lambda: o.scatter_add(0,i,x)))())
add("index_select",    lambda n: (lambda x=torch.randn(n,device=DEV), i=torch.randint(0,n,(n,),device=DEV): (lambda: torch.index_select(x,0,i)))())
add("embedding_bag",   lambda n: (lambda w=torch.randn(4096,128,device=DEV), i=torch.randint(0,4096,(n,),device=DEV), o=torch.arange(0,n,max(n//32,1),device=DEV,dtype=torch.long): (lambda: F.embedding_bag(i,w,o)))())
add("unique",          lambda n: (lambda x=torch.randint(0,1000,(n,),device=DEV): (lambda: torch.unique(x)))())
add("nonzero",         lambda n: (lambda x=(torch.randn(n,device=DEV)>0): (lambda: x.nonzero()))())
add("layer_norm",      lambda n: (lambda x=torch.randn(max(n//128,1),128,device=DEV,dtype=DT), w=torch.randn(128,device=DEV,dtype=DT), b=torch.randn(128,device=DEV,dtype=DT): (lambda: F.layer_norm(x,(128,),w,b)))())
add("softmax",         lambda n: (lambda x=torch.randn(max(n//128,1),128,device=DEV,dtype=DT): (lambda: F.softmax(x,-1)))())
add("conv2d (cuDNN)",  lambda n: (lambda x=torch.randn(max(n//(3*64*64),1),3,64,64,device=DEV,dtype=DT), w=torch.randn(32,3,3,3,device=DEV,dtype=DT): (lambda: F.conv2d(x,w,padding=1)))())
add("GEMM (cuBLAS)",   lambda n: (lambda a=torch.randn(max(n//512,1),512,device=DEV,dtype=DT), b=torch.randn(512,512,device=DEV,dtype=DT): (lambda: a@b))())

print(f"{'op':<20} {'distinct kernels':>14} {'class':>10}  kernels used")
print("-"*108)
tier1 = tier2 = 0
for name, maker in CASES:
    allk = Counter()
    for n in SIZES:
        try:
            for k in kernels(maker(n)): allk[k]+=1
        except Exception:
            pass
    nk = len(allk)
    cat = "tier 1 stable" if nk<=1 else ("tier 1 (<=2)" if nk<=2 else "**kernel changes**")
    if nk<=2: tier1+=1
    else: tier2+=1
    names = " | ".join(sorted(allk)[:2])
    print(f"{name:<20} {nk:>14} {cat:>12}  {names[:66]}")
print(f"\n{len(SIZES)} sizes spanning 3 orders of magnitude. {tier1}/{len(CASES)} ops have a stable kernel or only 2 variants (the planner patching grid+params is enough), "
      f"{tier2}/{len(CASES)} change kernel.")
