"""Does cuBLAS mm/addmm really always produce exactly 1 kernel node?  fp32 split-K."""
import sys, os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_vfy_util import capture
dev="cuda"
Ms=[8,16,32,64,96,128,192,256,384,512,1024,2048,4096]
for dt,tag in ((torch.float32,"fp32"),(torch.bfloat16,"bf16")):
    print(f"\n=== mm {tag}  K=N=256")
    res={}
    for p,order in enumerate([Ms,list(reversed(Ms))]):
        for M in order:
            a=torch.randn(M,256,device=dev,dtype=dt); b=torch.randn(256,256,device=dev,dtype=dt)
            o=torch.empty(M,256,device=dev,dtype=dt)
            n,g=capture(lambda a=a,o=o,b=b: torch.mm(a,b,out=o))
            res.setdefault(M,[]).append((len([x for x in n if x["type"]=="KERNEL"]),
                                         tuple(x["name"][:55] for x in n if x["type"]=="KERNEL")))
            del g,n
        torch.cuda.empty_cache()
    for M in Ms:
        a,b=res[M]
        flag="  <<< MULTI-NODE" if a[0]>1 else ""
        print(f"  M={M:5d} kernel_nodes={a[0]} stable_across_passes={a==b}{flag}")
        if a[0]>1:
            for nm in a[1]: print(f"          {nm}")
print("\n=== addmm fp32 (bias epilogue) node count")
for M in [8,64,128,256,1024]:
    a=torch.randn(M,256,device=dev); b=torch.randn(256,256,device=dev)
    bias=torch.randn(256,device=dev); o=torch.empty(M,256,device=dev)
    n,g=capture(lambda a=a,o=o,b=b,bias=bias: torch.addmm(bias,a,b,out=o))
    print(f"  M={M:5d} nodes={len(n)} {[x['type'] for x in n]}")
    for x in n:
        if x["type"]=="KERNEL": print(f"        {x['name'][:70]}")
    del g,n
print("DONE")
