import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_vfy_util import capture
dev="cuda"; bf=torch.bfloat16
print("=== claim 13: fp32 mm kernel identity")
for M in (128,1024,8192):
    a=torch.randn(M,256,device=dev); b=torch.randn(256,256,device=dev); o=torch.empty(M,256,device=dev)
    n,g=capture(lambda a=a,o=o: torch.mm(a,b,out=o))
    for x in n:
        if x["type"]=="KERNEL": print(f"  M={M:5d} {x['name'][:95]} grid={x['grid']} smem={x['smem']} nparams={len(x['paraminfo'])}")
    del g,n
print("\n=== claim 16: pytorch_flash node count vs S")
for S in (64,128,256,512,1024,2048):
    q=torch.randn(2,4,S,64,device=dev,dtype=bf);k=torch.randn(2,4,S,64,device=dev,dtype=bf);v=torch.randn(2,4,S,64,device=dev,dtype=bf)
    box={}
    n,g=capture(lambda: box.__setitem__('o',torch.ops.aten._scaled_dot_product_flash_attention.default(q,k,v)))
    print(f"  S={S:5d} nodes={len(n)} {[x['type'] for x in n]}")
    for x in n:
        if x["type"]=="KERNEL": print(f"        {x['name'][:88]} grid={x['grid']}")
    del g,n
print("\n=== claim 14: bf16 channels_last conv engine")
x=torch.randn(4,32,32,32,device=dev,dtype=bf).to(memory_format=torch.channels_last)
w=torch.randn(64,32,3,3,device=dev,dtype=bf).to(memory_format=torch.channels_last)
n,g=capture(lambda: torch.convolution(x,w,None,(1,1),(1,1),(1,1),False,(0,0),1))
print(f"  nodes={len(n)}")
for y in n:
    if y["type"]=="KERNEL": print("   ",y["name"][:95])
print("DONE")
