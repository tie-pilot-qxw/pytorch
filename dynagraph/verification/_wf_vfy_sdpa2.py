"""If S=128 and S=512 are the same cuDNN kernel, can one captured graph serve both?
DynaGraph-style: one padded buffer (S_max=512), vary the logical S."""
import os, sys, ctypes, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_vfy_util import capture, KNP, _cu
dev="cuda"; bf=torch.bfloat16; hold=[]
torch.manual_seed(0)
SM=512
Q=torch.randn(2,4,SM,64,device=dev,dtype=bf);Kt=torch.randn(2,4,SM,64,device=dev,dtype=bf);V=torch.randn(2,4,SM,64,device=dev,dtype=bf)

def cap(S):
    q=Q[:,:,:S,:];k=Kt[:,:,:S,:];v=V[:,:,:S,:]
    box={}
    nodes,g=capture(lambda: box.__setitem__('o',torch.ops.aten._scaled_dot_product_cudnn_attention.default(q,k,v,None,False)))
    return nodes,g,q,k,v,box

recs={}
for S in (128,256,512):
    nodes,g,q,k,v,box = cap(S)
    K=[n for n in nodes if n["type"]=="KERNEL"][0]
    print(f"S={S:4d} padded-view  nodes={len(nodes)} {[n['type'] for n in nodes]}")
    print(f"      name={K['name'][:80]}")
    print(f"      func=0x{K['func']:X} grid={K['grid']} smem={K['smem']} q.ptr=0x{q.data_ptr():X} out.ptr=0x{box['o'][0].data_ptr():X}")
    recs[S]=(nodes,g,q,k,v,box,K)

K1=recs[128][6]; K5=recs[512][6]
print("\nname same:",K1['name']==K5['name'],"  func same:",K1['func']==K5['func'],
      "  smem same:",K1['smem']==K5['smem'],"  paraminfo same:",K1['paraminfo']==K5['paraminfo'])
print("param slots that differ between S=128 and S=512 (same padded buffers):")
for i,((o,s),b_,c_) in enumerate(zip(K1["paraminfo"],K1["parambytes"],K5["parambytes"])):
    if b_!=c_:
        extra = ""
        if s<=16: extra = f"  S128={b_.hex(' ')}  S512={c_.hex(' ')}"
        print(f"   p{i} off={o} sz={s}{extra}")

def set_exec_kp(g,node,func,grid,blk,smem,blobs):
    ptrs=(ctypes.c_void_p*len(blobs))()
    for i,b in enumerate(blobs):
        buf=(ctypes.c_ubyte*len(b)).from_buffer_copy(b); hold.append(buf); ptrs[i]=ctypes.cast(buf,ctypes.c_void_p)
    hold.append(ptrs)
    p=KNP(); p.func=func
    p.gridDimX,p.gridDimY,p.gridDimZ=grid; p.blockDimX,p.blockDimY,p.blockDimZ=blk
    p.sharedMemBytes=smem; p.kernelParams=ctypes.cast(ptrs,ctypes.POINTER(ctypes.c_void_p)); p.extra=None
    p.kern=None; p.ctx=None
    r=_cu.cuGraphExecKernelNodeSetParams_v2(ctypes.c_void_p(g.raw_cuda_graph_exec()),ctypes.c_void_p(node),ctypes.byref(p))
    nm=ctypes.c_char_p(); _cu.cuGetErrorName(r,ctypes.byref(nm)); return r,nm.value

print("\n=== take the S=512 captured graph and patch it DOWN to S=128 ===")
nodes,g,q,k,v,box,K = recs[512]
g.instantiate(); out=box['o'][0]
ref512=torch.ops.aten._scaled_dot_product_cudnn_attention.default(Q,Kt,V,None,False)[0].clone()
out.zero_(); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
print("  baseline (S=512) replay == eager:", torch.equal(out,ref512))
Ksrc = recs[128][6]
r,nm = set_exec_kp(g,K["node"],Ksrc["func"],Ksrc["grid"],Ksrc["block"],Ksrc["smem"],Ksrc["parambytes"])
print("  patch node with S=128's func/grid/params ->",r,nm)
if r==0:
    out.fill_(float('nan')); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
    q1=Q[:,:,:128,:];k1=Kt[:,:,:128,:];v1=V[:,:,:128,:]
    ref128=torch.ops.aten._scaled_dot_product_cudnn_attention.default(q1,k1,v1,None,False)[0]
    got=out[:,:,:128,:]
    print("  out[:, :, :128] == eager S=128 :", torch.equal(got,ref128))
    print("  max abs diff                   :", (got.float()-ref128.float()).abs().max().item())
    tail=out[:,:,128:,:]
    print("  rows 128..511 left as NaN      :", bool(torch.isnan(tail).all()))
print("\n=== patch it back UP to S=512 ===")
r,nm = set_exec_kp(g,K["node"],K["func"],K["grid"],K["block"],K["smem"],K["parambytes"])
out.zero_(); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
print("  restore ->",r,nm," == eager S=512:", torch.equal(out,ref512))
print("DONE")
