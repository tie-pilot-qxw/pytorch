"""Claims 1, 3, 20, 4: extern_kernels namespace size, gemm backend defaults,
which ops really fall back, and extern-vs-triton launch counts."""
import io, logging, re, torch

print("=== claim 1: extern_kernels membership, measured at several import stages")
def count():
    from torch._inductor.select_algorithm import extern_kernels
    ms=[m for m in dir(extern_kernels) if not m.startswith("_")]
    return ms
import torch._inductor.select_algorithm as SA
print("  right after importing select_algorithm :", len(count()), sorted(count()))
import torch._inductor.kernel.mm, torch._inductor.kernel.conv
print("  after importing kernel.mm + kernel.conv:", len(count()))
try:
    import torch._inductor.kernel.bmm, torch._inductor.kernel.mm_plus_mm
except Exception as e: print("   (",e,")")
print("  after importing kernel.bmm            :", len(count()))
from torch._inductor.select_algorithm import ExternKernelChoice
print("  ExternKernelChoice._registry size      :", len(ExternKernelChoice._registry))

print("\n=== claim 3: gemm backend config defaults")
from torch._inductor import config
print("  max_autotune =",config.max_autotune," max_autotune_gemm =",config.max_autotune_gemm)
print("  max_autotune_gemm_backends =",config.max_autotune_gemm_backends)
from torch._inductor.utils import use_aten_gemm_kernels
print("  use_aten_gemm_kernels() =",use_aten_gemm_kernels())

import os, glob, time
CACHE=os.environ.get("TORCHINDUCTOR_CACHE_DIR","/tmp/torchinductor_root")
def _wrappers():
    return {f for f in glob.glob(CACHE+"/**/*.py",recursive=True)}
def compiled_code(fn,args,**kw):
    before=_wrappers()
    torch._dynamo.reset()
    c=torch.compile(fn,**kw); r=c(*args)
    if kw.get("bwd"): pass
    torch.cuda.synchronize()
    new=[f for f in _wrappers()-before]
    txt=""
    for f in sorted(new):
        t=open(f).read()
        if "def call(" in t: txt+="\n#####FILE "+f+"\n"+t
    return txt, r

def report(tag,pair):
    from collections import Counter
    code=pair[0] if isinstance(pair,tuple) else pair
    print(f"\n### {tag}")
    parts=[p for p in code.split("#####FILE ") if p.strip()]
    if not parts: print("   NO NEW WRAPPER (cache hit)"); return
    for gi,part in enumerate(parts):
        body=part.split("\n",1)[1]
        ext=re.findall(r"extern_kernels\.([\w\.]+)\(",body)
        aten=re.findall(r"torch\.ops\.((?:aten|_c10d_functional)\.[\w\.]+)\(",body)
        trit_objs=set(re.findall(r"(triton_\w+) = async_compile\.triton",body))
        runs=re.findall(r"(triton_\w+)\.run\(",body)
        print(f"  -- graph[{gi}]")
        print("     extern_kernels.*  :",dict(Counter(ext)),"total",len(ext))
        print("     aten/c10d calls   :",dict(Counter(aten)),"total",len(aten))
        print("     triton objs:",len(trit_objs)," triton .run sites:",len(runs))
        print("     OPAQUE",len(ext)+len(aten),"vs TRITON .run",len(runs))

dev="cuda"; bf=torch.bfloat16
class Blk(torch.nn.Module):
    def __init__(s,d=256,h=4):
        super().__init__(); s.a=torch.nn.MultiheadAttention(d,h,batch_first=True,dtype=bf)
        s.l1=torch.nn.Linear(d,4*d,dtype=bf); s.l2=torch.nn.Linear(4*d,d,dtype=bf)
        s.n1=torch.nn.LayerNorm(d,dtype=bf); s.n2=torch.nn.LayerNorm(d,dtype=bf)
    def forward(s,x):
        y,_=s.a(x,x,x,need_weights=False); x=s.n1(x+y)
        return s.n2(x+s.l2(torch.nn.functional.gelu(s.l1(x))))
class Net(torch.nn.Module):
    def __init__(s,n=4):
        super().__init__(); s.b=torch.nn.ModuleList([Blk() for _ in range(n)])
    def forward(s,x):
        for b in s.b: x=b(x)
        return x
m=Net().to(dev).eval()
x=torch.randn(2,128,256,device=dev,dtype=bf)
with torch.no_grad():
    report("transformer 4-layer FWD bf16", compiled_code(m,(x,)))

def misc(a,b,idx):
    s,_=torch.sort(a,dim=1)
    t,_=torch.topk(a,4,dim=-1)
    c=torch.cumsum(a,dim=1)
    d=torch.zeros_like(a); d.scatter_add_(1,idx,a)
    e=torch.zeros_like(a); e.index_put_((torch.arange(8,device=a.device),),b,accumulate=True)
    return s.sum()+t.sum()+c.sum()+d.sum()+e.sum()
a=torch.randn(8,64,device=dev); b2=torch.randn(8,64,device=dev)
idx=torch.randint(0,64,(8,64),device=dev)
report("misc sort/topk/cumsum/scatter/index_put", compiled_code(misc,(a,b2,idx)))
# claim 5: backward
m2=Net().to(dev)
x2=torch.randn(2,128,256,device=dev,dtype=bf,requires_grad=True)
def fwdbwd(xx):
    o=m2(xx); o.sum().backward(); return o
report("transformer 4-layer FWD+BWD bf16", compiled_code(fwdbwd,(x2,)))
# claim 6: resnet18
try:
    import torchvision
    r18=torchvision.models.resnet18().to(dev).eval()
    xi=torch.randn(2,3,224,224,device=dev)
    with torch.no_grad():
        report("resnet18 FWD fp32 (dynamic=True)", compiled_code(r18,(xi,),dynamic=True))
except Exception as e:
    print("resnet18 skipped:",type(e).__name__,e)
print("DONE")
