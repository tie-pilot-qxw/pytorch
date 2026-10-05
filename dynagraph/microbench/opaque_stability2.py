"""Corrected version: for each size, record the sequence of "kernels launched by this one call",
then compare that sequence across sizes. Same sequence = stable topology = the planner patching grid+params is enough."""
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
DEV="cuda"; DT=torch.float16
torch.manual_seed(0)

def seq(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    out=[]
    for e in p.events():
        if e.device_type.name!="CUDA": continue
        n=e.name
        if n.startswith(("Memcpy","Memset")): continue
        out.append(n.split("<")[0].split("(")[0][:48])
    return tuple(out)

SIZES=[1000, 4096, 10007, 65536, 262144, 1000003]
CASES=[]
def add(n,f): CASES.append((n,f))
add("sort",          lambda n: (lambda x=torch.randn(n,device=DEV): (lambda: torch.sort(x)))())
add("topk(k=64)",    lambda n: (lambda x=torch.randn(n,device=DEV): (lambda: torch.topk(x,min(64,n))))())
add("cumsum",        lambda n: (lambda x=torch.randn(n,device=DEV): (lambda: torch.cumsum(x,0)))())
add("scatter_add",   lambda n: (lambda x=torch.randn(n,device=DEV),i=torch.randint(0,max(n//2,1),(n,),device=DEV),o=torch.zeros(max(n//2,1),device=DEV): (lambda: o.scatter_add(0,i,x)))())
add("index_select",  lambda n: (lambda x=torch.randn(n,device=DEV),i=torch.randint(0,n,(n,),device=DEV): (lambda: torch.index_select(x,0,i)))())
add("embedding_bag", lambda n: (lambda w=torch.randn(4096,128,device=DEV),i=torch.randint(0,4096,(n,),device=DEV),o=torch.arange(0,n,max(n//32,1),device=DEV,dtype=torch.long): (lambda: F.embedding_bag(i,w,o)))())
add("unique",        lambda n: (lambda x=torch.randint(0,1000,(n,),device=DEV): (lambda: torch.unique(x)))())
add("nonzero",       lambda n: (lambda x=(torch.randn(n,device=DEV)>0): (lambda: x.nonzero()))())
add("layer_norm",    lambda n: (lambda x=torch.randn(max(n//128,1),128,device=DEV,dtype=DT),w=torch.randn(128,device=DEV,dtype=DT),b=torch.randn(128,device=DEV,dtype=DT): (lambda: F.layer_norm(x,(128,),w,b)))())
add("softmax",       lambda n: (lambda x=torch.randn(max(n//128,1),128,device=DEV,dtype=DT): (lambda: F.softmax(x,-1)))())
add("conv2d vary bs",   lambda n: (lambda x=torch.randn(max(n//(3*64*64),1),3,64,64,device=DEV,dtype=DT),w=torch.randn(32,3,3,3,device=DEV,dtype=DT): (lambda: F.conv2d(x,w,padding=1)))())
add("conv2d vary HW",   lambda n: (lambda s=max(int((n/3)**0.5),8),: (lambda x=torch.randn(4,3,s,s,device=DEV,dtype=DT),w=torch.randn(32,3,3,3,device=DEV,dtype=DT): (lambda: F.conv2d(x,w,padding=1)))())())
add("GEMM (cuBLAS)", lambda n: (lambda a=torch.randn(max(n//512,1),512,device=DEV,dtype=DT),b=torch.randn(512,512,device=DEV,dtype=DT): (lambda: a@b))())

print(f"{'op':<18} {'kernels per call':>18} {'distinct seqs, 6 sizes':>24} {'verdict':>16}")
print("-"*92)
stable=switch=0
detail=[]
for name,maker in CASES:
    seqs=[]
    for n in SIZES:
        try: seqs.append(seq(maker(n)))
        except Exception: pass
    if not seqs: continue
    uniq=set(seqs); lens=sorted(set(len(s) for s in seqs))
    topo_same = len(set(len(s) for s in seqs))==1
    verdict = "tier 1 fully stable" if len(uniq)==1 else ("**topology changes**" if not topo_same else "**kernel changes**")
    if len(uniq)==1: stable+=1
    else: switch+=1; detail.append((name,uniq))
    print(f"{name:<18} {str(lens):>18} {len(uniq):>24} {verdict:>18}")
print(f"\n{stable}/{stable+switch} ops have an \"identical kernel sequence\" across 6 sizes spanning 3 orders of magnitude --"
      f" the planner patching grid+params is enough, no re-recording needed.")
print(f"{switch}/{stable+switch} change. A closer look at the ones that change:")
for name,uniq in detail:
    print(f"  {name}: {len(uniq)} sequences, lengths {sorted(set(len(u) for u in uniq))}")
    for u in list(uniq)[:2]:
        print(f"     {' -> '.join(x[:26] for x in u)[:100]}")
