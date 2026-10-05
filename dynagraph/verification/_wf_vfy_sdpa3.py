"""Redo: patch the S=512 cuDNN-SDPA graph down to S=128, remapping the donor's
output pointers onto the target graph's buffers."""
import os, sys, ctypes, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_vfy_util import capture, KNP, _cu
dev="cuda"; bf=torch.bfloat16; hold=[]
torch.manual_seed(0)
SM=512
Q=torch.randn(2,4,SM,64,device=dev,dtype=bf);Kt=torch.randn(2,4,SM,64,device=dev,dtype=bf);V=torch.randn(2,4,SM,64,device=dev,dtype=bf)
def cap(S):
    q=Q[:,:,:S,:];k=Kt[:,:,:S,:];v=V[:,:,:S,:]; box={}
    nodes,g=capture(lambda: box.__setitem__('o',torch.ops.aten._scaled_dot_product_cudnn_attention.default(q,k,v,None,False)))
    return nodes,g,box
n1,g1,b1 = cap(128)
n5,g5,b5 = cap(512)
K1=[n for n in n1 if n["type"]=="KERNEL"][0]; K5=[n for n in n5 if n["type"]=="KERNEL"][0]

def ptrs(box):
    d={}
    for i,t in enumerate(box['o']):
        if torch.is_tensor(t) and t.numel()>0: d[i]=t.data_ptr()
    return d
p1=ptrs(b1); p5=ptrs(b5)
print("S128 out tensors:",{i:hex(v) for i,v in p1.items()})
print("S512 out tensors:",{i:hex(v) for i,v in p5.items()})

# build donor->target pointer remap from the S=128 capture onto the S=512 capture
remap={p1[i]:p5[i] for i in p1 if i in p5}
print("remap:",{hex(a):hex(b) for a,b in remap.items()})

def remap_blob(b):
    ba=bytearray(b); hits=[]
    for off in range(0,len(ba)-7):
        val=int.from_bytes(ba[off:off+8],'little')
        if val in remap:
            ba[off:off+8]=remap[val].to_bytes(8,'little'); hits.append((off,hex(val)))
    return bytes(ba),hits

SCRATCH=torch.empty(1<<20,dtype=torch.uint8,device=dev)
print("p4 donor value:",hex(int.from_bytes(K1["parambytes"][4],"little")),
      " donor out ptr:",hex(p1[0]),
      " delta:",hex(int.from_bytes(K1["parambytes"][4],"little")-p1[0]))
remap[int.from_bytes(K1["parambytes"][4],"little")]=SCRATCH.data_ptr()
newblobs=[]; allhits=[]
for i,b in enumerate(K1["parambytes"]):
    nb,h=remap_blob(b); newblobs.append(nb)
    if h: allhits.append((i,h))
print("pointer slots rewritten:",allhits)

def set_exec_kp(g,node,func,grid,blk,smem,blobs):
    pp=(ctypes.c_void_p*len(blobs))()
    for i,b in enumerate(blobs):
        buf=(ctypes.c_ubyte*len(b)).from_buffer_copy(b); hold.append(buf); pp[i]=ctypes.cast(buf,ctypes.c_void_p)
    hold.append(pp)
    p=KNP(); p.func=func
    p.gridDimX,p.gridDimY,p.gridDimZ=grid; p.blockDimX,p.blockDimY,p.blockDimZ=blk
    p.sharedMemBytes=smem; p.kernelParams=ctypes.cast(pp,ctypes.POINTER(ctypes.c_void_p)); p.extra=None
    p.kern=None; p.ctx=None
    r=_cu.cuGraphExecKernelNodeSetParams_v2(ctypes.c_void_p(g.raw_cuda_graph_exec()),ctypes.c_void_p(node),ctypes.byref(p))
    nm=ctypes.c_char_p(); _cu.cuGetErrorName(r,ctypes.byref(nm)); return r,nm.value

g5.instantiate(); out=b5['o'][0]
ref512=torch.ops.aten._scaled_dot_product_cudnn_attention.default(Q,Kt,V,None,False)[0].clone()
q1=Q[:,:,:128,:];k1=Kt[:,:,:128,:];v1=V[:,:,:128,:]
ref128=torch.ops.aten._scaled_dot_product_cudnn_attention.default(q1,k1,v1,None,False)[0].clone()
out.zero_(); torch.cuda.synchronize(); g5.replay(); torch.cuda.synchronize()
print("baseline S=512 replay == eager:",torch.equal(out,ref512))
r,nm=set_exec_kp(g5,K5["node"],K1["func"],K1["grid"],K1["block"],K1["smem"],newblobs)
print("patch S512-graph -> S=128 :",r,nm)
out.fill_(float('nan')); torch.cuda.synchronize(); g5.replay(); torch.cuda.synchronize()
got=out[:,:,:128,:]
print("  out[:,:, :128] == eager S=128 :",torch.equal(got,ref128))
print("  max abs diff                  :",(got.float()-ref128.float()).abs().max().item())
print("  rows 128.. still NaN          :",bool(torch.isnan(out[:,:,128:,:]).all()))
r,nm=set_exec_kp(g5,K5["node"],K5["func"],K5["grid"],K5["block"],K5["smem"],K5["parambytes"])
out.zero_(); torch.cuda.synchronize(); g5.replay(); torch.cuda.synchronize()
print("restore S=512 :",torch.equal(out,ref512))
print("DONE")
