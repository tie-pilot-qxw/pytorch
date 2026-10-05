"""1) Profiler-verify that host SetParams really swaps the cuDNN kernel FUNCTION.
   2) Separate 'pointer slots' from 'shape slots' inside the opaque cuDNN param struct,
      by diffing (same shape, different addresses) vs (same addresses, different shape)."""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from _wf_cudnn_lib import libcuda, graph_nodes, node_type, kernel_params, func_name, param_info, KNP

KEEP=[]
def patch_kernel(ex, dst, src):
    ps = kernel_params(src); pi = param_info(ps.func); n=len(pi)
    bufs=[]; ptrs=(ctypes.c_void_p*n)()
    for i,(_,off,sz) in enumerate(pi):
        b=(ctypes.c_ubyte*sz).from_buffer_copy(bytes((ctypes.c_ubyte*sz).from_address(ps.kernelParams[i])))
        bufs.append(b); ptrs[i]=ctypes.cast(b,ctypes.c_void_p)
    KEEP.append((bufs,ptrs))
    new=KNP(); new.func=ps.func
    new.gridDimX,new.gridDimY,new.gridDimZ=ps.gridDimX,ps.gridDimY,ps.gridDimZ
    new.blockDimX,new.blockDimY,new.blockDimZ=ps.blockDimX,ps.blockDimY,ps.blockDimZ
    new.sharedMemBytes=ps.sharedMemBytes
    new.kernelParams=ctypes.cast(ptrs,ctypes.POINTER(ctypes.c_void_p))
    new.extra=None; new.kern=None; new.ctx=None
    return libcuda.cuGraphExecKernelNodeSetParams_v2(ctypes.c_void_p(ex),ctypes.c_void_p(dst),ctypes.byref(new))

HOLD=[]
def cap(fn):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(4): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): out=fn()
    torch.cuda.synchronize(); HOLD.append((g,out))
    return g,out,graph_nodes(g.raw_cuda_graph())

def kernels_in_replay(g):
    g.replay(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay(); torch.cuda.synchronize()
    return [e.name for e in p.events() if e.device_type.name=="CUDA"]

DEV="cuda"; DT=torch.float16
W  = torch.randn(64,64,3,3,device=DEV,dtype=DT).to(memory_format=torch.channels_last)
XB = torch.randn(32,64,56,56,device=DEV,dtype=DT).to(memory_format=torch.channels_last)

print("="*92); print("### 1. does SetParams really change which kernel executes?")
gA,oA,nA = cap(lambda: F.conv2d(XB[:4], W, padding=1))
gB,oB,nB = cap(lambda: F.conv2d(XB[:8], W, padding=1))
print("  graph A (b=4)  kernel:", func_name(kernel_params(nA[0]).func)[:90])
print("  graph B (b=8)  kernel:", func_name(kernel_params(nB[0]).func)[:90])
print("  A before patch, profiler sees:", [k[:80] for k in kernels_in_replay(gA)])
gA.instantiate(); ex=gA.raw_cuda_graph_exec()
rc = patch_kernel(ex, nA[0], nB[0]); print(f"  patch rc={rc}")
ref = F.conv2d(XB[:8], W, padding=1); oB.fill_(float('nan')); torch.cuda.synchronize()
seen = kernels_in_replay(gA)
print("  A after  patch, profiler sees:", [k[:80] for k in seen])
print(f"  numerics: allclose={torch.allclose(oB.float(),ref.float(),rtol=2e-2,atol=2e-2)} "
      f"nan={bool(torch.isnan(oB).any())}")
print(f"  -> FUNCTION SWAP CONFIRMED: {any('sm80_xmma' in k for k in seen)}")

print(); print("="*92); print("### 2. pointer slots vs shape slots inside the opaque param struct")
XA = torch.randn(32,64,56,56,device=DEV,dtype=DT).to(memory_format=torch.channels_last)
XC = torch.randn(32,64,56,56,device=DEV,dtype=DT).to(memory_format=torch.channels_last)
WA = W.clone(); WC = W.clone()
def pbytes(nodes):
    out=[]
    for nd in nodes:
        if node_type(nd)!="KERNEL": out.append(None); continue
        p=kernel_params(nd); pi=param_info(p.func)
        out.append([(i,off,sz,bytes((ctypes.c_ubyte*sz).from_address(p.kernelParams[i]))) for i,off,sz in pi])
    return out
def runs(pa,pb):
    res=[]
    for (i,off,sz,ba),(_,_,_,bb) in zip(pa,pb):
        j=0
        while j<sz:
            if ba[j]!=bb[j]:
                k=j
                while k<sz and ba[k]!=bb[k]: k+=1
                lo=(j//4)*4; hi=((k+3)//4)*4
                res.append((i,lo,hi,int.from_bytes(ba[lo:hi],'little'),int.from_bytes(bb[lo:hi],'little')))
                j=k
            else: j+=1
    merged=[]
    for r in res:
        if merged and merged[-1][0]==r[0] and r[1]<=merged[-1][2]:
            merged[-1]=(r[0],merged[-1][1],max(r[2],merged[-1][2]),merged[-1][3],r[4])
        else: merged.append(r)
    return merged

g1,o1,n1 = cap(lambda: F.conv2d(XA[:16], WA, padding=1))   # shape 16, addr set 1
g2,o2,n2 = cap(lambda: F.conv2d(XC[:16], WC, padding=1))   # shape 16, addr set 2
g3,o3,n3 = cap(lambda: F.conv2d(XA[:13], WA, padding=1))   # shape 13, addr set 1 (same base ptrs)
p1,p2,p3 = pbytes(n1)[0], pbytes(n2)[0], pbytes(n3)[0]
print(f"  kernel: {func_name(kernel_params(n1[0]).func)[:90]}")
print(f"  param struct size: {sum(sz for _,_,sz,_ in p1)} bytes in {len(p1)} param(s)")
ptr_runs   = runs(p1,p2)
shape_runs = runs(p1,p3)
print(f"\n  A) same shape, different tensor addresses -> {len(ptr_runs)} differing run(s) = POINTER/alloc slots")
for i,lo,hi,a,b in ptr_runs: print(f"     param{i}[{lo}:{hi}] {a} -> {b}")
print(f"\n  B) same addresses, shape 16 -> 13         -> {len(shape_runs)} differing run(s)")
pset={(i,lo) for i,lo,_,_,_ in ptr_runs}
for i,lo,hi,a,b in shape_runs:
    tag = "also-moves-with-address" if (i,lo) in pset else "SHAPE-ONLY"
    print(f"     param{i}[{lo}:{hi}] {a} -> {b}   [{tag}]")
shape_only=[r for r in shape_runs if (r[0],r[1]) not in pset]
print(f"\n  -> {len(shape_only)} slot(s) move with shape but not with address; "
      f"{len(ptr_runs)} slot(s) move with address.")
