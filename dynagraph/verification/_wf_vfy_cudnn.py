"""VERIFY claim 10 (conv node count changes with shape) and claim 9 (SDPA 'same name,
different CUfunction' -- is that really a different kernel, or the same cubin twice?)."""
import sys, os, ctypes, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_vfy_util import capture, KNP, _cu, func_attrs, func_params

dev="cuda"; bf=torch.bfloat16
print("cudnn:",torch.backends.cudnn.version(),"allow_tf32:",torch.backends.cudnn.allow_tf32,
      "benchmark:",torch.backends.cudnn.benchmark)

print("\n=== EXP-CONV  fp32 NCHW conv, w=64x32x3x3, N=4, H=W sweep (ascending then descending)")
Hs=[16,24,32,40,48,56,64,80,96,128]
W=torch.randn(64,32,3,3,device=dev)
out={}
for p,order in enumerate([Hs, list(reversed(Hs))]):
    for H in order:
        x=torch.randn(4,32,H,H,device=dev)
        nodes,g = capture(lambda x=x: torch.convolution(x,W,None,(1,1),(1,1),(1,1),False,(0,0),1))
        sig=(len(nodes), tuple(n["type"] for n in nodes),
             tuple(n["name"][:70] for n in nodes if n["type"]=="KERNEL"))
        out.setdefault(H,[]).append(sig)
        del g,nodes
    torch.cuda.empty_cache()
for H in Hs:
    a,b=out[H]
    print(f"H={H:4d} pass0 nodes={a[0]} {a[1]}  | pass1 nodes={b[0]}  stable={a==b}")
    for nm in a[2]: print(f"        {nm}")

print("\n=== EXP-SDPA  cuDNN attention, S sweep")
info={}
keep=[]
for S in (128,256,512,1024):
    q=torch.randn(2,4,S,64,device=dev,dtype=bf);k=torch.randn(2,4,S,64,device=dev,dtype=bf);v=torch.randn(2,4,S,64,device=dev,dtype=bf)
    box={}
    def f(q=q,k=k,v=v,box=box):
        box['o']=torch.ops.aten._scaled_dot_product_cudnn_attention.default(q,k,v,None,False)
    nodes,g = capture(f)
    kn=[n for n in nodes if n["type"]=="KERNEL"]
    print(f"S={S:5d} nodes={len(nodes)} {[n['type'] for n in nodes]}")
    for n in kn:
        print(f"   name={n['name'][:95]}")
        print(f"   grid={n['grid']} blk={n['block']} smem={n['smem']} func=0x{n['func']:X} mod=0x{n['module']:X} "
              f"paraminfo={n['paraminfo']} kparams={n['kernelParams']} extra={n['extra']} "
              f"bloblen={len(n['extrablob']) if n['extrablob'] else None}")
        print(f"   attrs={func_attrs(n['func'])}")
    info[S]=(nodes,g,q,k,v,box)
    keep.append((g,box))

print("\n--- same-name-different-handle check")
n128=[n for n in info[128][0] if n["type"]=="KERNEL"][0]
n512=[n for n in info[512][0] if n["type"]=="KERNEL"][0]
print("name equal   :", n128["name"]==n512["name"])
print("func equal   :", n128["func"]==n512["func"], f"0x{n128['func']:X} vs 0x{n512['func']:X}")
print("module equal :", n128["module"]==n512["module"], f"0x{n128['module']:X} vs 0x{n512['module']:X}")
print("attrs equal  :", func_attrs(n128["func"])==func_attrs(n512["func"]))
print("paraminfo eq :", n128["paraminfo"]==n512["paraminfo"], n128["paraminfo"], n512["paraminfo"])
print("blob len     :", len(n128["extrablob"] or b""), len(n512["extrablob"] or b""))

print("\n--- DECISIVE: swap S=128 node's func to the S=512 func, keep S=128 grid/smem/blob.")
print("    If the result is still correct, the two handles are the SAME kernel and")
print("    'different handle' does NOT imply 'different kernel'.")
nodes,g,q,k,v,box = info[128]
g.instantiate()
outt = box['o'][0]
ref = torch.ops.aten._scaled_dot_product_cudnn_attention.default(q,k,v,None,False)[0].clone()
outt.zero_(); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
print("  baseline replay matches eager:", torch.equal(outt, ref))
hold=[]
def set_exec(g,node,func,grid,blk,smem,blob):
    buf=(ctypes.c_ubyte*len(blob)).from_buffer_copy(blob); sz=ctypes.c_size_t(len(blob))
    ex=(ctypes.c_void_p*5)()
    ex[0]=ctypes.c_void_p(1); ex[1]=ctypes.cast(buf,ctypes.c_void_p)
    ex[2]=ctypes.c_void_p(2); ex[3]=ctypes.cast(ctypes.pointer(sz),ctypes.c_void_p); ex[4]=ctypes.c_void_p(0)
    hold.extend([buf,sz,ex])
    p=KNP(); p.func=func
    p.gridDimX,p.gridDimY,p.gridDimZ=grid; p.blockDimX,p.blockDimY,p.blockDimZ=blk
    p.sharedMemBytes=smem; p.kernelParams=None
    p.extra=ctypes.cast(ex,ctypes.POINTER(ctypes.c_void_p)); p.kern=None; p.ctx=None
    r=_cu.cuGraphExecKernelNodeSetParams_v2(ctypes.c_void_p(g.raw_cuda_graph_exec()),
                                            ctypes.c_void_p(node),ctypes.byref(p))
    nm=ctypes.c_char_p(); _cu.cuGetErrorName(r,ctypes.byref(nm)); return r,nm.value
r,nm = set_exec(g,n128["node"],n512["func"],n128["grid"],n128["block"],n128["smem"],n128["extrablob"])
print("  setParams(func=S512func, rest=S128) ->",r,nm)
if r==0:
    outt.zero_(); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
    same=torch.equal(outt,ref)
    print("  result still bit-exact vs eager S=128:", same)
    print("  max abs diff:", (outt.float()-ref.float()).abs().max().item())
print("DONE")
