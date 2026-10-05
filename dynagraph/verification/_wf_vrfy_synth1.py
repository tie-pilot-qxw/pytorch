"""Attack claim 6: 'cuDNN shape slots are libdivide magic numbers, so you can NEVER
synthesize param bytes for an unseen shape -- you must really capture.'
Step 1: dump the cuDNN-SDPA node's param image across several L (and across two
different address sets at the same L) and classify every changed 4-byte word.
"""
import ctypes, gc
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

libcuda = ctypes.CDLL("libcuda.so.1")
class KNP(ctypes.Structure):
    _fields_ = [("func",ctypes.c_void_p),("gx",ctypes.c_uint),("gy",ctypes.c_uint),("gz",ctypes.c_uint),
                ("bx",ctypes.c_uint),("by",ctypes.c_uint),("bz",ctypes.c_uint),("smem",ctypes.c_uint),
                ("kernelParams",ctypes.POINTER(ctypes.c_void_p)),("extra",ctypes.POINTER(ctypes.c_void_p)),
                ("kern",ctypes.c_void_p),("ctx",ctypes.c_void_p)]
class MSP(ctypes.Structure):
    _fields_=[("dst",ctypes.c_ulonglong),("pitch",ctypes.c_size_t),("value",ctypes.c_uint),
              ("elementSize",ctypes.c_uint),("width",ctypes.c_size_t),("height",ctypes.c_size_t)]
TY={0:"KERNEL",1:"MEMCPY",2:"MEMSET"}
def nodes_of(g):
    n=ctypes.c_size_t(0); libcuda.cuGraphGetNodes(ctypes.c_void_p(g),None,ctypes.byref(n))
    a=(ctypes.c_void_p*n.value)(); libcuda.cuGraphGetNodes(ctypes.c_void_p(g),a,ctypes.byref(n))
    return [a[i] for i in range(n.value)]
def ntype(nd):
    t=ctypes.c_int(0); libcuda.cuGraphNodeGetType(ctypes.c_void_p(nd),ctypes.byref(t)); return TY.get(t.value,str(t.value))
def kp(nd):
    p=KNP(); return p if libcuda.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(nd),ctypes.byref(p))==0 else None
def msp(nd):
    p=MSP(); return p if libcuda.cuGraphMemsetNodeGetParams(ctypes.c_void_p(nd),ctypes.byref(p))==0 else None
def fname(f):
    s=ctypes.c_char_p()
    return s.value.decode(errors="replace") if libcuda.cuFuncGetName(ctypes.byref(s),ctypes.c_void_p(f))==0 else "?"
def pinfo(f):
    o=[];off=ctypes.c_size_t();sz=ctypes.c_size_t()
    for i in range(64):
        if libcuda.cuFuncGetParamInfo(ctypes.c_void_p(f),ctypes.c_size_t(i),ctypes.byref(off),ctypes.byref(sz))!=0: break
        o.append((i,off.value,sz.value))
    return o

HOLD=[]
DEV="cuda"; DT=torch.bfloat16
B,H,D = 2,8,64
def cap(L, tag):
    q=torch.randn(B,H,L,D,device=DEV,dtype=DT)
    def f():
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(q,q,q,is_causal=True)
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): out=f()
    torch.cuda.synchronize()
    nds=nodes_of(g.raw_cuda_graph())
    kn=[z for z in nds if ntype(z)=="KERNEL"][0]
    mn=[z for z in nds if ntype(z)=="MEMSET"]
    p=kp(kn); pi=pinfo(p.func)
    img=[bytes((ctypes.c_ubyte*sz).from_address(p.kernelParams[i])) for i,_,sz in pi]
    m = msp(mn[0]) if mn else None
    HOLD.append((g,out,q))
    return dict(L=L,tag=tag,g=g,out=out,q=q,node=kn,mnode=(mn[0] if mn else None),p=p,pi=pi,img=img,
                grid=(p.gx,p.gy,p.gz),block=(p.bx,p.by,p.bz),smem=p.smem,
                memset=(None if m is None else (m.dst,m.pitch,m.value,m.elementSize,m.width,m.height)),
                name=fname(p.func))

caps = {}
for L in [256,512,1024,1500,2048]:
    caps[(L,'a')] = cap(L,'a')
caps[(512,'b')] = cap(512,'b')      # same L, different addresses

c0 = caps[(512,'a')]
print(f"kernel: {c0['name'][:95]}")
print(f"param_info: {c0['pi']}  total={sum(sz for _,_,sz in c0['pi'])} bytes")
for k,v in caps.items():
    print(f"  L={v['L']}{v['tag']}: grid={v['grid']} block={v['block']} smem={v['smem']} memset={v['memset']}")
    print(f"      q=0x{v['q'].data_ptr():x} out=0x{v['out'].data_ptr():x}")

def words(img):
    """flatten to list of (param_idx, offset, uint32)"""
    out=[]
    for k,b in enumerate(img):
        for o in range(0,len(b)-3,4):
            out.append((k,o,int.from_bytes(b[o:o+4],'little')))
    return out

def diff(A,B):
    return [(k,o,va,vb) for (k,o,va),(_,_,vb) in zip(words(A['img']),words(B['img'])) if va!=vb]

print("\n--- same L=512, different tensor addresses -> POINTER/alloc words")
ptr = diff(caps[(512,'a')], caps[(512,'b')])
for k,o,va,vb in ptr: print(f"   p{k}[{o}:{o+4}] {va:#x} -> {vb:#x}")
pset = {(k,o) for k,o,_,_ in ptr}

print("\n--- same address set, L=512 -> L=1024")
for k,o,va,vb in diff(caps[(512,'a')], caps[(1024,'a')]):
    tag = "ptr-word" if (k,o) in pset else "SHAPE"
    print(f"   p{k}[{o}:{o+4}] {va} -> {vb}   ({va:#x} -> {vb:#x})  [{tag}]")

print("\n--- shape words as a function of L (address set 'a')")
Ls = [256,512,1024,1500,2048]
base = caps[(256,'a')]
allshape = set()
for L in Ls[1:]:
    for k,o,va,vb in diff(base, caps[(L,'a')]):
        if (k,o) not in pset: allshape.add((k,o))
for (k,o) in sorted(allshape):
    row = []
    for L in Ls:
        b = caps[(L,'a')]['img'][k]
        row.append(int.from_bytes(b[o:o+4],'little'))
    print(f"   p{k}[{o}:{o+4}]  " + "  ".join(f"L={L}:{v}" for L,v in zip(Ls,row)))
