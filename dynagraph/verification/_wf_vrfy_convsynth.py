"""(a) verify claim 6's libdivide arithmetic, (b) see whether the conv fprop param image
is closed-form across batch INSIDE one kernel bucket (b=11..17, all sm90 256x64x32)."""
import ctypes, math
import torch, torch.nn.functional as F
libcuda=ctypes.CDLL("libcuda.so.1")
class KNP(ctypes.Structure):
    _fields_=[("func",ctypes.c_void_p),("gx",ctypes.c_uint),("gy",ctypes.c_uint),("gz",ctypes.c_uint),
              ("bx",ctypes.c_uint),("by",ctypes.c_uint),("bz",ctypes.c_uint),("smem",ctypes.c_uint),
              ("kernelParams",ctypes.POINTER(ctypes.c_void_p)),("extra",ctypes.POINTER(ctypes.c_void_p)),
              ("kern",ctypes.c_void_p),("ctx",ctypes.c_void_p)]
def nodes_of(g):
    n=ctypes.c_size_t(0); libcuda.cuGraphGetNodes(ctypes.c_void_p(g),None,ctypes.byref(n))
    a=(ctypes.c_void_p*n.value)(); libcuda.cuGraphGetNodes(ctypes.c_void_p(g),a,ctypes.byref(n))
    return [a[i] for i in range(n.value)]
def ntype(nd):
    t=ctypes.c_int(0); libcuda.cuGraphNodeGetType(ctypes.c_void_p(nd),ctypes.byref(t)); return t.value
def pinfo(f):
    o=[];off=ctypes.c_size_t();sz=ctypes.c_size_t()
    for i in range(64):
        if libcuda.cuFuncGetParamInfo(ctypes.c_void_p(f),ctypes.c_size_t(i),ctypes.byref(off),ctypes.byref(sz))!=0: break
        o.append((i,off.value,sz.value))
    return o
def magic(d):
    if d&(d-1)==0: return 0x80000000
    return (1<<(31+math.ceil(math.log2(d))))//d + 1

print("--- claim 6 arithmetic check (their two measured pairs)")
for d,m in [(28,2454267027),(23,2987803337)]:
    print(f"   divisor {d}: floor(2^(31+ceil(log2 d))/d)+1 = {magic(d)}  measured {m}  MATCH={magic(d)==m}")
print("   -> the 'magic number' is a closed form of the divisor, not an opaque token.\n")

HOLD=[]; DEV="cuda"
W=torch.randn(64,64,3,3,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)
X=torch.randn(24,64,56,56,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)
def cap(b):
    f=lambda: F.conv2d(X[:b],W,padding=1)
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): out=f()
    torch.cuda.synchronize()
    nds=nodes_of(g.raw_cuda_graph()); kn=[z for z in nds if ntype(z)==0][0]
    p=KNP(); libcuda.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(kn),ctypes.byref(p))
    pi=pinfo(p.func)
    img=[bytes((ctypes.c_ubyte*sz).from_address(p.kernelParams[i])) for i,_,sz in pi]
    HOLD.append((g,out))
    return dict(b=b,img=img,grid=(p.gx,p.gy,p.gz),out=out,x=X,nnodes=len(nds))
C={b:cap(b) for b in range(11,18)}
print("--- conv channels_last 64->64 3x3 56x56 fp16, bucket b=11..17 (one kernel, one topology)")
for b,v in C.items(): print(f"   b={b}: nodes={v['nnodes']} grid={v['grid']} out=0x{v['out'].data_ptr():x}")
ptrwords=set()
for k,bb in enumerate(C[11]['img']):
    for o in range(0,len(bb)-7,8):
        v=int.from_bytes(bb[o:o+8],'little')
        if 0<=v-X.data_ptr()<(1<<26) or 0<=v-W.data_ptr()<(1<<20) or 0<=v-C[11]['out'].data_ptr()<(1<<26):
            ptrwords |= {(k,o),(k,o+4)}
changed={}
for k,bb in enumerate(C[11]['img']):
    for o in range(0,len(bb)-3,4):
        vals=[int.from_bytes(C[b]['img'][k][o:o+4],'little') for b in range(11,18)]
        if len(set(vals))>1 and (k,o) not in ptrwords: changed[(k,o)]=vals
print(f"\n   {len(changed)} non-pointer 4-byte words vary with batch inside the bucket:")
def classify(vals):
    bs=list(range(11,18))
    if all(v==b for v,b in zip(vals,bs)): return "= N"
    if all(v==b-1 for v,b in zip(vals,bs)): return "= N-1"
    for c in (3136,56,64,1792,3136*64):
        if all(v==b*c for v,b in zip(vals,bs)): return f"= N*{c}"
    if all(v==-(-(b*3136)//1792) for v,b in zip(vals,bs)): return "= ceil(N*H*W/1792)"
    ds=[-(-(b*3136)//1792) for b in bs]
    if all(v==magic(d) for v,d in zip(vals,ds)): return "= libdivide_magic(ceil(N*H*W/1792))  [CLOSED FORM]"
    if all(v==math.ceil(math.log2(d)) for v,d in zip(vals,ds)): return "= ceil(log2(ceil(N*H*W/1792)))"
    if all(v==-(-(b*3136)//256) for v,b in zip(vals,bs)): return "= ceil(N*H*W/256)"
    return "?"
nun=0
for (k,o),vals in sorted(changed.items()):
    c=classify(vals)
    if c=="?": nun+=1
    print(f"     p{k}[{o}:{o+4}] {vals}   {c}")
print(f"\n   closed form found for {len(changed)-nun}/{len(changed)} varying words; {nun} still unexplained")
