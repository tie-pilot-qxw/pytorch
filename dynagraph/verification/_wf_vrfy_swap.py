"""INDEPENDENT re-test of the load-bearing claim:
   "cuGraphExecKernelNodeSetParams in CUDA 13.1 can swap the kernel FUNCTION of a node."

Discriminator is chosen so that a *silent no-op* is impossible to mistake for success:
graph A computes relu(x), graph B computes sigmoid(x) over the SAME input buffer.
If the func really swaps, replaying A must produce sigmoid values, bit-exact vs eager.
Also directly tests the reviewer's warning: is the CUfunction pointer stable across
two captures of the same op?  (If not, 'pointer changed' != 'kernel changed'.)
"""
import ctypes, sys
import torch

libcuda = ctypes.CDLL("libcuda.so.1")

class KNP(ctypes.Structure):
    _fields_ = [("func", ctypes.c_void_p),
                ("gx", ctypes.c_uint), ("gy", ctypes.c_uint), ("gz", ctypes.c_uint),
                ("bx", ctypes.c_uint), ("by", ctypes.c_uint), ("bz", ctypes.c_uint),
                ("smem", ctypes.c_uint),
                ("kernelParams", ctypes.POINTER(ctypes.c_void_p)),
                ("extra", ctypes.POINTER(ctypes.c_void_p)),
                ("kern", ctypes.c_void_p), ("ctx", ctypes.c_void_p)]

def errname(rc):
    s = ctypes.c_char_p(); libcuda.cuGetErrorName(ctypes.c_int(rc), ctypes.byref(s))
    return f"{rc}({s.value.decode() if s.value else '?'})"

def nodes_of(graph):
    n = ctypes.c_size_t(0)
    rc = libcuda.cuGraphGetNodes(ctypes.c_void_p(graph), None, ctypes.byref(n)); assert rc==0, errname(rc)
    arr = (ctypes.c_void_p*n.value)()
    rc = libcuda.cuGraphGetNodes(ctypes.c_void_p(graph), arr, ctypes.byref(n)); assert rc==0, errname(rc)
    return [arr[i] for i in range(n.value)]

def ntype(nd):
    t = ctypes.c_int(0); libcuda.cuGraphNodeGetType(ctypes.c_void_p(nd), ctypes.byref(t)); return t.value

def kparams(nd):
    p = KNP(); rc = libcuda.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(nd), ctypes.byref(p))
    return p if rc==0 else None

def fname(f):
    s = ctypes.c_char_p(); rc = libcuda.cuFuncGetName(ctypes.byref(s), ctypes.c_void_p(f))
    return s.value.decode(errors="replace") if rc==0 else f"<rc={rc}>"

def pinfo(f):
    out=[]; off=ctypes.c_size_t(); sz=ctypes.c_size_t()
    for i in range(64):
        if libcuda.cuFuncGetParamInfo(ctypes.c_void_p(f), ctypes.c_size_t(i),
                                      ctypes.byref(off), ctypes.byref(sz)) != 0: break
        out.append((i, off.value, sz.value))
    return out

HOLD=[]; KEEP=[]
def cap(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): out = fn()
    torch.cuda.synchronize(); HOLD.append((g,out))
    return g, out, nodes_of(g.raw_cuda_graph())

def copy_params_onto(ex, dst_node, src_node, override_func=None):
    ps = kparams(src_node); pi = pinfo(ps.func)
    n = len(pi); ptrs = (ctypes.c_void_p*n)(); bufs=[]
    for i,(_,_,sz) in enumerate(pi):
        b = (ctypes.c_ubyte*sz).from_buffer_copy(bytes((ctypes.c_ubyte*sz).from_address(ps.kernelParams[i])))
        bufs.append(b); ptrs[i] = ctypes.cast(b, ctypes.c_void_p)
    KEEP.append((bufs,ptrs))
    new = KNP(); new.func = override_func if override_func is not None else ps.func
    new.gx,new.gy,new.gz = ps.gx,ps.gy,ps.gz
    new.bx,new.by,new.bz = ps.bx,ps.by,ps.bz
    new.smem = ps.smem
    new.kernelParams = ctypes.cast(ptrs, ctypes.POINTER(ctypes.c_void_p))
    new.extra=None; new.kern=None; new.ctx=None
    return libcuda.cuGraphExecKernelNodeSetParams_v2(ctypes.c_void_p(ex), ctypes.c_void_p(dst_node), ctypes.byref(new))

torch.manual_seed(0)
DEV="cuda"
N = 1<<20
x  = torch.randn(N, device=DEV)          # has negatives -> relu != sigmoid everywhere

print("="*90)
print("### 0. control: is the CUfunction pointer stable across two captures of the SAME op?")
gR1,oR1,nR1 = cap(lambda: torch.relu(x))
gR2,oR2,nR2 = cap(lambda: torch.relu(x))
f1 = kparams(nR1[0]).func; f2 = kparams(nR2[0]).func
print(f"  capture1 func=0x{f1:x}  name={fname(f1)[:70]}")
print(f"  capture2 func=0x{f2:x}  name={fname(f2)[:70]}")
print(f"  pointer identical across captures: {f1==f2}   name identical: {fname(f1)==fname(f2)}")

print()
print("="*90)
print("### 1. FUNC SWAP with an unmistakable discriminator: relu -> sigmoid")
gA,oA,nA = cap(lambda: torch.relu(x))
gB,oB,nB = cap(lambda: torch.sigmoid(x))
print(f"  graph A nodes={len(nA)} types={[ntype(z) for z in nA]}")
print(f"  graph B nodes={len(nB)} types={[ntype(z) for z in nB]}")
pA, pB = kparams(nA[0]), kparams(nB[0])
print(f"  A kernel: {fname(pA.func)[:100]}")
print(f"  B kernel: {fname(pB.func)[:100]}")
print(f"  distinct kernel NAMES (not just pointers): {fname(pA.func)!=fname(pB.func)}")
print(f"  A grid={(pA.gx,pA.gy,pA.gz)} block={(pA.bx,pA.by,pA.bz)}   B grid={(pB.gx,pB.gy,pB.gz)}")

ref_sig = torch.sigmoid(x)
ref_rel = torch.relu(x)
torch.cuda.synchronize()

# control: replay A unpatched -> B's output buffer must stay NaN
gA.instantiate(); ex = gA.raw_cuda_graph_exec()
oB.fill_(float("nan")); torch.cuda.synchronize()
gA.replay(); torch.cuda.synchronize()
print(f"  [control] after UNPATCHED replay of A, B's out buffer all-NaN: {bool(torch.isnan(oB).all())}")

rc = copy_params_onto(ex, nA[0], nB[0])
print(f"  cuGraphExecKernelNodeSetParams_v2 rc={errname(rc)}")
oB.fill_(float("nan")); torch.cuda.synchronize()
gA.replay(); torch.cuda.synchronize()
eq_sig = bool(torch.equal(oB, ref_sig))
eq_rel = bool(torch.equal(oB, ref_rel))
print(f"  after PATCHED replay of A:  oB bit-equal to sigmoid(x) = {eq_sig}")
print(f"                              oB bit-equal to relu(x)    = {eq_rel}")
print(f"                              any NaN left               = {bool(torch.isnan(oB).any())}")
print(f"  -> FUNC SWAP REALLY HAPPENED: {eq_sig and not eq_rel}")

# and swap back
rc2 = copy_params_onto(ex, nA[0], nA[0])
oA.fill_(float("nan")); oB.fill_(float("nan")); torch.cuda.synchronize()
gA.replay(); torch.cuda.synchronize()
print(f"  swap BACK rc={errname(rc2)}  oA==relu(x): {bool(torch.equal(oA, ref_rel))}  oB still NaN: {bool(torch.isnan(oB).all())}")

print()
print("="*90)
print("### 2. is the swap restricted to same-library / same-module funcs?")
# aten kernel -> cuDNN/cublas kernel from a totally different module
import torch.nn.functional as F
Wc = torch.randn(64,64,3,3,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)
Xc = torch.randn(4,64,56,56,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)
gC,oC,nC = cap(lambda: F.conv2d(Xc, Wc, padding=1))
kn = [z for z in nC if ntype(z)==0]
print(f"  conv graph nodes={len(nC)} kernel-nodes={len(kn)}")
pC = kparams(kn[0])
print(f"  conv kernel: {fname(pC.func)[:100]}")
m1 = ctypes.c_void_p(); m2 = ctypes.c_void_p()
libcuda.cuFuncGetModule(ctypes.byref(m1), ctypes.c_void_p(pA.func))
libcuda.cuFuncGetModule(ctypes.byref(m2), ctypes.c_void_p(pC.func))
print(f"  module(aten relu)=0x{(m1.value or 0):x}   module(cudnn conv)=0x{(m2.value or 0):x}  same={m1.value==m2.value}")
gA.instantiate(); ex2 = gA.raw_cuda_graph_exec()
rc3 = copy_params_onto(ex2, nA[0], kn[0])
print(f"  patch aten-relu node <- cuDNN conv kernel: rc={errname(rc3)}")
if rc3 == 0:
    refc = F.conv2d(Xc, Wc, padding=1); oC.fill_(float("nan")); torch.cuda.synchronize()
    gA.replay(); torch.cuda.synchronize()
    print(f"    replay -> conv out matches eager: {bool(torch.allclose(oC.float(), refc.float(), rtol=2e-2, atol=2e-2))}"
          f"  any_nan={bool(torch.isnan(oC).any())}")
