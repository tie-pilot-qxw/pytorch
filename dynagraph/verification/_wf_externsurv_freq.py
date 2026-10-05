import re, collections, torch, torch.nn as nn, torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
dev="cuda"
class Blk(nn.Module):
    def __init__(s,d=256,h=4):
        super().__init__(); s.h=h
        s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d)
        s.l1=nn.LayerNorm(d); s.l2=nn.LayerNorm(d)
        s.f1=nn.Linear(d,4*d); s.f2=nn.Linear(4*d,d)
    def forward(s,x):
        B,S,D=x.shape
        q=s.qkv(s.l1(x)).view(B,S,3,s.h,D//s.h).permute(2,0,3,1,4)
        a=F.scaled_dot_product_attention(q[0],q[1],q[2]).transpose(1,2).reshape(B,S,D)
        x=x+s.o(a); return x+s.f2(F.gelu(s.f1(s.l2(x))))
class Net(nn.Module):
    def __init__(s,n=4):
        super().__init__(); s.emb=nn.Embedding(1000,256); s.b=nn.ModuleList([Blk() for _ in range(n)]); s.head=nn.Linear(256,1000)
    def forward(s,i):
        x=s.emb(i)
        for b in s.b: x=b(x)
        return s.head(x)
CALL=re.compile(r"(extern_kernels\.[A-Za-z_][\w.]*|torch\.ops\.(?:aten|_c10d_functional|inductor)\.[\w.]+)\s*\(")
def rep(tag,codes):
    print(f"\n#### {tag}")
    for i,c in enumerate(codes):
        body=[l for l in c.splitlines() if not l.lstrip().startswith("#")]
        body="\n".join(body)
        cnt=collections.Counter(CALL.findall(body))
        cnt.pop("torch.ops.inductor._alloc_from_pool",None)
        tri=len(set(re.findall(r"(triton_[a-z_]+_\d+)\.run\(",body)))
        runs=len(re.findall(r"triton_[a-z_]+_\d+\.run\(",body))
        print(f"  graph[{i}]: triton kernel objs={tri}, triton .run sites={runs}")
        for k,v in sorted(cnt.items(),key=lambda kv:-kv[1]): print(f"     {v:3d} x {k}")

m=Net(4).to(dev).to(torch.bfloat16).eval(); cm=torch.compile(m,dynamic=True)
ids=torch.randint(0,1000,(2,128),device=dev)
with torch.no_grad(): _,codes=run_and_get_code(cm,ids)
rep("transformer 4-layer FWD bf16 (d=256,h=4)",codes)

m2=Net(4).to(dev).to(torch.bfloat16).train(); cm2=torch.compile(m2,dynamic=True)
def fb(i):
    o=cm2(i).float().sum(); o.backward(); return o
_,codes=run_and_get_code(fb,torch.randint(0,1000,(2,128),device=dev))
rep("transformer 4-layer FWD+BWD bf16",codes)

import torchvision
try:
    rn=torchvision.models.resnet18().to(dev).eval(); crn=torch.compile(rn,dynamic=True)
    with torch.no_grad(): _,codes=run_and_get_code(crn,torch.randn(2,3,128,128,device=dev))
    rep("resnet18 FWD fp32",codes)
except Exception as e:
    print("torchvision:",e)
print("DONE")
