"""How often does the node topology / kernel change across a realistic shape sweep?
Plus: verify that the shape-only slots in the cuDNN conv param struct are libdivide
magic multipliers derived from the shape (i.e. not raw shape values)."""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from _wf_cudnn_lib import graph_nodes, node_type, kernel_params, func_name

print("### magic-number check on the cuDNN conv param struct")
for d, observed in ((28, 2454267027), (23, 2987803337)):
    for s in range(0, 8):
        if (1 << (32 + s)) // d + 1 == observed:
            print(f"  divisor {d}: floor(2^(32+{s})/{d})+1 = {observed}  MATCH "
                  f"(libdivide/unsigned-magic form)")
            break
    else:
        print(f"  divisor {d}: no magic-number match for {observed}")

HOLD=[]
def cap(fn):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): out=fn()
    torch.cuda.synchronize(); HOLD.append((g,out))
    return graph_nodes(g.raw_cuda_graph())

def sig(nodes):
    out=[]
    for nd in nodes:
        t=node_type(nd)
        if t!="KERNEL": out.append((t,None,None)); continue
        p=kernel_params(nd)
        out.append((t, func_name(p.func), (p.gridDimX,p.gridDimY,p.gridDimZ)))
    return out

DEV="cuda"
QB=torch.randn(2,8,4096,64,device=DEV,dtype=torch.bfloat16)
LS=[128,256,333,512,777,1024,1536,2048,3000,4096]
for label,bk in (("FLASH",SDPBackend.FLASH_ATTENTION),
                 ("EFFICIENT",SDPBackend.EFFICIENT_ATTENTION),
                 ("CUDNN",SDPBackend.CUDNN_ATTENTION)):
    print(f"\n{'='*92}\n### SDPA {label} over seqlen {LS}")
    topo={}
    for L in LS:
        def f(L=L):
            with sdpa_kernel(bk):
                return F.scaled_dot_product_attention(QB[:,:,:L],QB[:,:,:L],QB[:,:,:L],is_causal=True)
        try:
            sg=sig(cap(f))
        except Exception as e:
            print(f"  L={L}: FAIL {type(e).__name__}: {str(e)[:90]}"); continue
        key=tuple((t,n) for t,n,_ in sg)
        topo.setdefault(key,[]).append((L,[g for _,_,g in sg]))
        print(f"  L={L:5d}: {len(sg)} nodes  grids={[g for _,_,g in sg]}")
    print(f"  -> {len(topo)} distinct (node-type, kernel) topolog{'y' if len(topo)==1 else 'ies'}:")
    for key,ls in topo.items():
        print(f"     L={[l for l,_ in ls]}")
        for t,n in key: print(f"        {t} {(n or '')[:95]}")

print(f"\n{'='*92}\n### conv2d channels_last over batch (recap, with grids)")
W=torch.randn(64,64,3,3,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)
XB=torch.randn(48,64,56,56,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)
topo={}
for b in [1,2,3,4,5,6,7,8,12,16,20,24,32,48]:
    sg=sig(cap(lambda b=b: F.conv2d(XB[:b],W,padding=1)))
    key=tuple((t,n) for t,n,_ in sg)
    topo.setdefault(key,[]).append(b)
    print(f"  b={b:3d}: {len(sg)} nodes grids={[g for _,_,g in sg]}")
print(f"  -> {len(topo)} distinct topologies over 14 batch sizes:")
for key,bs in topo.items():
    print(f"     b={bs}")
    for t,n in key: print(f"        {t} {(n or '')[:95]}")
