"""Claims 17/18/20: does sort really fall back 'in the default config', or only
under dynamic shapes / large dims?"""
import re, collections, torch
from torch._inductor.utils import run_and_get_code
from torch._inductor import config
print("decompose_sort_ops =", config.triton.decompose_sort_ops)
CALL=re.compile(r"(extern_kernels\.[A-Za-z_][\w.]*|torch\.ops\.aten\.[\w.]+)\s*\(")
dev="cuda"
def f(a,b,idx):
    s,_=torch.sort(a,dim=1)
    t,_=torch.topk(a,4,dim=-1)
    c=torch.cumsum(a,dim=1)
    d=torch.zeros_like(a); d.scatter_add_(1,idx,a)
    e=torch.zeros_like(a); e.index_put_((torch.arange(a.shape[0],device=a.device),),b,accumulate=True)
    return s.sum()+t.sum()+c.sum()+d.sum()+e.sum()
for N in (64, 4096, 40000):
    for dyn in (False, True):
        torch._dynamo.reset()
        a=torch.randn(8,N,device=dev); b=torch.randn(8,N,device=dev)
        idx=torch.randint(0,N,(8,N),device=dev)
        cf=torch.compile(f,dynamic=dyn)
        _,codes=run_and_get_code(cf,a,b,idx)
        body="\n".join(l for c in codes for l in c.splitlines() if not l.lstrip().startswith("#"))
        cnt=collections.Counter(CALL.findall(body))
        runs=len(re.findall(r"triton_[a-z_]+_\d+\.run\(",body))
        print(f"N={N:6d} dynamic={dyn!s:5s}  fallbacks={dict(cnt)}  triton .run={runs}")
print("DONE")
