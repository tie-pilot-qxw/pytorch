"""DECISIVE test against claim 6 ('cuDNN shape slots are libdivide magic -> you can
NEVER synthesize params for an unseen shape, you must really capture it').

Fit closed-form formulas for every shape word from captures at L=256,512,1024 only.
1) validate byte-exactly against HELD-OUT captures at L=1500 and L=2048
2) then patch a graph captured at L=512 with a FULLY SYNTHESIZED image for an L that
   is NEVER captured anywhere in this process, and compare the replay to eager.
"""
import ctypes, math, gc
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
TY={0:"KERNEL",2:"MEMSET"}
def en(rc):
    s=ctypes.c_char_p(); libcuda.cuGetErrorName(ctypes.c_int(rc),ctypes.byref(s))
    return f"{rc}({s.value.decode() if s.value else '?'})"
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
def pinfo(f):
    o=[];off=ctypes.c_size_t();sz=ctypes.c_size_t()
    for i in range(64):
        if libcuda.cuFuncGetParamInfo(ctypes.c_void_p(f),ctypes.c_size_t(i),ctypes.byref(off),ctypes.byref(sz))!=0: break
        o.append((i,off.value,sz.value))
    return o

HOLD=[]; DEV="cuda"; DT=torch.bfloat16; B,H,D=2,8,64
def cap(L):
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
    kn=[z for z in nds if ntype(z)=="KERNEL"][0]; mn=[z for z in nds if ntype(z)=="MEMSET"][0]
    p=kp(kn); pi=pinfo(p.func)
    img=[bytearray((ctypes.c_ubyte*sz).from_address(p.kernelParams[i])) for i,_,sz in pi]
    m=msp(mn)
    HOLD.append((g,out,q))
    return dict(L=L,g=g,out=out,q=q,kn=kn,mn=mn,p=p,pi=pi,img=img,grid=(p.gx,p.gy,p.gz),
                block=(p.bx,p.by,p.bz),smem=p.smem,
                memset=(m.dst,m.pitch,m.value,m.elementSize,m.width,m.height))

FIT=[256,512,1024]; HELD=[1500,2048]
C={L:cap(L) for L in FIT+HELD}
c=C[512]
print("param_info (idx,offset,size):", c['pi'], " total =", sum(s for _,_,s in c['pi']))
for L in FIT+HELD:
    v=C[L]
    print(f"  L={L:5d} grid={v['grid']} block={v['block']} smem={v['smem']} "
          f"memset dst=0x{v['memset'][0]:x} w={v['memset'][4]} es={v['memset'][3]} "
          f"q=0x{v['q'].data_ptr():x} out=0x{v['out'].data_ptr():x}")

# ---- identify 8-byte pointer slots: every 8-aligned qword equal to a known buffer base
def find_ptr_slots(v):
    # NOTE: order matters -- the stats/workspace buffer sits exactly out+sizeof(out)
    # in the graph pool, so it must be matched BEFORE "out" or it is misattributed.
    known={"aux":v['memset'][0],"q":v['q'].data_ptr(),"out":v['out'].data_ptr()}
    slots=[]
    for k,b in enumerate(v['img']):
        for o in range(0,len(b)-7,8):
            val=int.from_bytes(b[o:o+8],'little')
            for nm,base in known.items():
                if 0 <= val-base < (1<<16):
                    slots.append((k,o,nm,val-base)); break
    return slots
slots=find_ptr_slots(C[512])
print("\npointer slots (param, offset, role, delta):")
for s in slots: print("   ", s)
pwords={(k,o+d) for k,o,_,_ in slots for d in (0,4)}

# ---- closed-form model for the shape words -------------------------------------
def magic(d):
    if d & (d-1) == 0: return 0x80000000
    L = math.ceil(math.log2(d))
    return (1 << (31+L)) // d + 1
def shiftv(d): return math.ceil(math.log2(d))

def model(L):
    t64 = -(-L//64); t128 = -(-L//128)
    return {
      (0,16): L, (0,20): L,
      (1,0): 8*t64,
      (2,0): 4*t64, (2,4): shiftv(4*t64), (2,8): magic(4*t64),
      (3,0): t128,  (3,4): shiftv(t128),  (3,8): magic(t128),
      **{(p,16): H*L for p in (5,6,9,10)},
      **{(p,20): D*L for p in (5,6,9,10)},
      **{(p,36): L-1 for p in (5,6,9,10)},
    }
def grid_of(L): return (min(132, B*H*(-(-L//128))), 1, 1)

# sanity: does the model reproduce the FIT captures?
def check(L, verbose=False):
    v=C[L]; m=model(L); bad=[]
    for (k,o),want in m.items():
        got=int.from_bytes(v['img'][k][o:o+4],'little')
        if got!=want: bad.append(((k,o),got,want))
    # and: are there any OTHER words that differ from the L=512 capture and are not modelled/not ptr?
    unexplained=[]
    for k,b in enumerate(v['img']):
        for o in range(0,len(b)-3,4):
            if (k,o) in m or (k,o) in pwords: continue
            if bytes(b[o:o+4]) != bytes(C[512]['img'][k][o:o+4]):
                unexplained.append(((k,o), int.from_bytes(C[512]['img'][k][o:o+4],'little'),
                                    int.from_bytes(b[o:o+4],'little')))
    ok_grid = (v['grid']==grid_of(L))
    print(f"  L={L:5d}: modelled words wrong={len(bad)}  unexplained-changed words={len(unexplained)}  "
          f"grid predicted {grid_of(L)} actual {v['grid']} -> {ok_grid}  block={v['block']} smem={v['smem']}")
    for x in bad[:8]: print("      WRONG", x)
    for x in unexplained[:8]: print("      UNEXPLAINED", x)
    return not bad and not unexplained and ok_grid

print("\n--- model reproduces the 3 FIT captures:")
for L in FIT: check(L)
print("--- model predicts the HELD-OUT captures (never used to fit):")
held_ok = all(check(L) for L in HELD)
print(f"  held-out byte-exact prediction: {held_ok}")

# ---- end-to-end: synthesize for an L that is NEVER captured ---------------------
KEEP=[]
def run_synth(Ltgt):
    print(f"\n{'='*94}\n### synthesize params for L={Ltgt} (NEVER captured) and patch the L=512 graph")
    qN   = torch.randn(B,H,Ltgt,D,device=DEV,dtype=DT)
    outN = torch.empty(B,H,Ltgt,D,device=DEV,dtype=DT)
    aux  = torch.zeros(4*1024*1024//4, device=DEV, dtype=torch.float32)   # stats/workspace
    roles={"q":qN.data_ptr(), "out":outN.data_ptr(), "aux":aux.data_ptr()}
    img=[bytearray(b) for b in C[512]['img']]
    for (k,o),val in model(Ltgt).items():
        img[k][o:o+4] = int(val).to_bytes(4,'little')
    for k,o,nm,d in slots:
        img[k][o:o+8] = (roles[nm]+d).to_bytes(8,'little')
    src=C[512]
    n=len(img); ptrs=(ctypes.c_void_p*n)(); bufs=[]
    for i,b in enumerate(img):
        cb=(ctypes.c_ubyte*len(b)).from_buffer_copy(bytes(b)); bufs.append(cb); ptrs[i]=ctypes.cast(cb,ctypes.c_void_p)
    KEEP.append((bufs,ptrs,qN,outN,aux))
    new=KNP(); new.func=src['p'].func
    new.gx,new.gy,new.gz = grid_of(Ltgt)
    new.bx,new.by,new.bz = src['block']
    new.smem = src['smem']
    new.kernelParams=ctypes.cast(ptrs,ctypes.POINTER(ctypes.c_void_p)); new.extra=None; new.kern=None; new.ctx=None
    src['g'].instantiate(); ex=src['g'].raw_cuda_graph_exec()
    ms=MSP(); ms.dst=roles["aux"]; ms.pitch=0; ms.value=0; ms.elementSize=src['memset'][3]
    ms.width=src['memset'][4]; ms.height=src['memset'][5]
    rcm=libcuda.cuGraphExecMemsetNodeSetParams(ctypes.c_void_p(ex),ctypes.c_void_p(src['mn']),ctypes.byref(ms),ctypes.c_void_p(0))
    rck=libcuda.cuGraphExecKernelNodeSetParams_v2(ctypes.c_void_p(ex),ctypes.c_void_p(src['kn']),ctypes.byref(new))
    print(f"  memset patch rc={en(rcm)}   kernel patch rc={en(rck)}   grid={grid_of(Ltgt)}")
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        ref = F.scaled_dot_product_attention(qN,qN,qN,is_causal=True)
    outN.fill_(float("nan")); torch.cuda.synchronize()
    src['g'].replay(); torch.cuda.synchronize()
    nan=bool(torch.isnan(outN).any())
    md=(outN.float()-ref.float()).abs().max().item()
    print(f"  replay: any_nan={nan}  max_abs_diff_vs_eager={md:.6g}  "
          f"allclose={bool(torch.allclose(outN.float(),ref.float(),rtol=2e-2,atol=2e-2))}")
    print(f"  (bit-exact vs eager: {bool(torch.equal(outN, ref))})")

for Ltgt in [900, 1792, 320]:
    run_synth(Ltgt)
