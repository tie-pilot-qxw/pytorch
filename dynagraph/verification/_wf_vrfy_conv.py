"""Independent re-run of claim 10/11/12: conv topology instability.
Uses cuFuncGetName (stable identity), fills the gaps in their batch sweep (9..11, 17..19),
and repeats the whole sweep twice to check the b->kernel map is deterministic."""
import ctypes, gc
import torch, torch.nn.functional as F
libcuda=ctypes.CDLL("libcuda.so.1")
class KNP(ctypes.Structure):
    _fields_=[("func",ctypes.c_void_p),("gx",ctypes.c_uint),("gy",ctypes.c_uint),("gz",ctypes.c_uint),
              ("bx",ctypes.c_uint),("by",ctypes.c_uint),("bz",ctypes.c_uint),("smem",ctypes.c_uint),
              ("kernelParams",ctypes.POINTER(ctypes.c_void_p)),("extra",ctypes.POINTER(ctypes.c_void_p)),
              ("kern",ctypes.c_void_p),("ctx",ctypes.c_void_p)]
TY={0:"KERNEL",1:"MEMCPY",2:"MEMSET"}
def nodes_of(g):
    n=ctypes.c_size_t(0); libcuda.cuGraphGetNodes(ctypes.c_void_p(g),None,ctypes.byref(n))
    a=(ctypes.c_void_p*n.value)(); libcuda.cuGraphGetNodes(ctypes.c_void_p(g),a,ctypes.byref(n))
    return [a[i] for i in range(n.value)]
def ntype(nd):
    t=ctypes.c_int(0); libcuda.cuGraphNodeGetType(ctypes.c_void_p(nd),ctypes.byref(t)); return TY.get(t.value,str(t.value))
def fname(f):
    s=ctypes.c_char_p()
    return s.value.decode(errors="replace") if libcuda.cuFuncGetName(ctypes.byref(s),ctypes.c_void_p(f))==0 else "?"
def psize(f):
    tot=0; off=ctypes.c_size_t(); sz=ctypes.c_size_t()
    for i in range(64):
        if libcuda.cuFuncGetParamInfo(ctypes.c_void_p(f),ctypes.c_size_t(i),ctypes.byref(off),ctypes.byref(sz))!=0: break
        tot+=sz.value
    return tot
def sig(fn):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): out=fn()
    torch.cuda.synchronize()
    r=[]
    for nd in nodes_of(g.raw_cuda_graph()):
        t=ntype(nd)
        if t!="KERNEL": r.append((t,"-",None,None,0)); continue
        p=KNP(); libcuda.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(nd),ctypes.byref(p))
        r.append((t,fname(p.func),(p.gx,p.gy,p.gz),(p.bx,p.by,p.bz),psize(p.func)))
    del out; g.reset(); del g; gc.collect(); torch.cuda.empty_cache()
    return r
DEV="cuda"
W=torch.randn(64,64,3,3,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)
X=torch.randn(48,64,56,56,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)
BS=list(range(1,21))+[24,32,48]
runs=[]
for r in range(2):
    m={}
    for b in BS:
        s=sig(lambda: F.conv2d(X[:b],W,padding=1))
        m[b]=tuple((t,n) for t,n,_,_,_ in s)
        if r==0:
            print(f"  b={b:3d}  {[t for t,_,_,_,_ in s]}  paramB={[z for _,_,_,_,z in s]}  "
                  f"grid={[g for _,_,g,_,_ in s]}")
            print(f"        {' | '.join(n[:78] for _,n,_,_,_ in s if n!='-')}")
    runs.append(m)
topo={}
for b,k in runs[0].items(): topo.setdefault(k,[]).append(b)
print(f"\n  distinct topologies over {len(BS)} batch sizes: {len(topo)}")
for k,v in topo.items():
    print(f"    {v}")
    for t,n in k: print(f"       {t} {n[:88]}")
print(f"\n  run1 == run2 (b->topology map deterministic within a process): {runs[0]==runs[1]}")
