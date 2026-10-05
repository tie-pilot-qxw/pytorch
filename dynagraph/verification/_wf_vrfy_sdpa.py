"""INDEPENDENT re-test of the tier-1 verdict claim:
   'cuDNN SDPA: 10 seqlens -> 1 topology, same kernel, only grid.x changes, clamped at 132.'
Probes harder than the original sweep: also varies batch, heads, head_dim, dtype,
is_causal, and non-multiple-of-64 lengths, and checks grid.y/grid.z too.
Then re-does the cross-L host patch with a FRESH output buffer as the target.
"""
import ctypes, gc
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

libcuda = ctypes.CDLL("libcuda.so.1")
class KNP(ctypes.Structure):
    _fields_ = [("func", ctypes.c_void_p),("gx",ctypes.c_uint),("gy",ctypes.c_uint),("gz",ctypes.c_uint),
                ("bx",ctypes.c_uint),("by",ctypes.c_uint),("bz",ctypes.c_uint),("smem",ctypes.c_uint),
                ("kernelParams",ctypes.POINTER(ctypes.c_void_p)),("extra",ctypes.POINTER(ctypes.c_void_p)),
                ("kern",ctypes.c_void_p),("ctx",ctypes.c_void_p)]
TY={0:"KERNEL",1:"MEMCPY",2:"MEMSET",3:"HOST",4:"GRAPH",5:"EMPTY"}
def errname(rc):
    s=ctypes.c_char_p(); libcuda.cuGetErrorName(ctypes.c_int(rc), ctypes.byref(s))
    return f"{rc}({s.value.decode() if s.value else '?'})"
def nodes_of(g):
    n=ctypes.c_size_t(0); libcuda.cuGraphGetNodes(ctypes.c_void_p(g),None,ctypes.byref(n))
    a=(ctypes.c_void_p*n.value)(); libcuda.cuGraphGetNodes(ctypes.c_void_p(g),a,ctypes.byref(n))
    return [a[i] for i in range(n.value)]
def ntype(nd):
    t=ctypes.c_int(0); libcuda.cuGraphNodeGetType(ctypes.c_void_p(nd),ctypes.byref(t)); return TY.get(t.value,str(t.value))
def kp(nd):
    p=KNP(); return p if libcuda.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(nd),ctypes.byref(p))==0 else None
def fname(f):
    s=ctypes.c_char_p()
    return s.value.decode(errors="replace") if libcuda.cuFuncGetName(ctypes.byref(s),ctypes.c_void_p(f))==0 else "?"
def pinfo(f):
    o=[];off=ctypes.c_size_t();sz=ctypes.c_size_t()
    for i in range(64):
        if libcuda.cuFuncGetParamInfo(ctypes.c_void_p(f),ctypes.c_size_t(i),ctypes.byref(off),ctypes.byref(sz))!=0: break
        o.append((i,off.value,sz.value))
    return o

def snapshot(fn):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): out=fn()
    torch.cuda.synchronize()
    sig=[]
    for nd in nodes_of(g.raw_cuda_graph()):
        t=ntype(nd)
        if t!="KERNEL": sig.append((t,"-",None,None,None)); continue
        p=kp(nd); sig.append((t,fname(p.func),(p.gx,p.gy,p.gz),(p.bx,p.by,p.bz),p.smem))
    del out; g.reset(); del g; gc.collect(); torch.cuda.empty_cache()
    return sig

DEV="cuda"
def sdpa_fn(B,H,L,D,dt,causal,backend):
    q=torch.randn(B,H,L,D,device=DEV,dtype=dt)
    def f():
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q,q,q,is_causal=causal)
    return f,q

def sweep(label, cases):
    print("="*100); print("###", label)
    topo={}
    for desc,(B,H,L,D,dt,causal) in cases:
        try:
            fn,q = sdpa_fn(B,H,L,D,dt,causal,SDPBackend.CUDNN_ATTENTION)
            sig = snapshot(fn)
            del q; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {desc:<34} FAIL {type(e).__name__}: {str(e)[:70]}"); continue
        key=tuple((t,n) for t,n,_,_,_ in sig)
        topo.setdefault(key,[]).append(desc)
        print(f"  {desc:<34} {len(sig)} nodes  grids={[g for _,_,g,_,_ in sig]} "
              f"blocks={[b for _,_,b,_,_ in sig]} smem={[s for _,_,_,_,s in sig]}")
    print(f"  -> {len(topo)} distinct topologies")
    for k,v in topo.items():
        print(f"     {v}")
        for t,n in k: print(f"        {t} {n[:92]}")
    print()
    return topo

# A. reproduce their exact seqlen sweep (B=2,H=8,D=64,bf16,causal)
sweep("cuDNN SDPA: their seqlen sweep (B=2,H=8,D=64,bf16,causal)",
      [(str(L),(2,8,L,64,torch.bfloat16,True)) for L in [128,256,333,512,777,1024,1536,2048,3000,4096]])

# B. push on the OTHER axes they never varied
cases=[]
for B in [1,2,4,8,16]: cases.append((f"B={B}",(B,8,1024,64,torch.bfloat16,True)))
for H in [1,4,8,16,32]: cases.append((f"H={H}",(2,H,1024,64,torch.bfloat16,True)))
sweep("cuDNN SDPA: vary batch and heads at L=1024", cases)

cases=[(f"D={D}",(2,8,1024,D,torch.bfloat16,True)) for D in [32,64,96,128,256]]
cases += [("fp16",(2,8,1024,64,torch.float16,True)), ("bf16",(2,8,1024,64,torch.bfloat16,True))]
cases += [("causal=False",(2,8,1024,64,torch.bfloat16,False)), ("causal=True",(2,8,1024,64,torch.bfloat16,True))]
sweep("cuDNN SDPA: vary head_dim / dtype / causal at L=1024", cases)

# C. tiny + odd lengths (does the kernel change at the small end?)
sweep("cuDNN SDPA: small / odd seqlens",
      [(str(L),(2,8,L,64,torch.bfloat16,True)) for L in [1,7,16,31,64,65,100,127,128,129,192]])
