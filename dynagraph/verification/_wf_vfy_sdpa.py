"""Claim 9 decisive test: cuDNN SDPA S=128 and S=512 have identical mangled name,
identical smem, identical param layout, identical func attrs, but different CUfunction
AND different CUmodule.  Are they the same kernel code loaded twice?"""
import os, sys, ctypes, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_vfy_util import capture, KNP, _cu, func_attrs

dev="cuda"; bf=torch.bfloat16
hold=[]

def set_exec_kp(g,node,func,grid,blk,smem,pinfo,blobs):
    ptrs=(ctypes.c_void_p*len(blobs))()
    for i,b in enumerate(blobs):
        buf=(ctypes.c_ubyte*len(b)).from_buffer_copy(b); hold.append(buf)
        ptrs[i]=ctypes.cast(buf,ctypes.c_void_p)
    hold.append(ptrs)
    p=KNP(); p.func=func
    p.gridDimX,p.gridDimY,p.gridDimZ=grid; p.blockDimX,p.blockDimY,p.blockDimZ=blk
    p.sharedMemBytes=smem
    p.kernelParams=ctypes.cast(ptrs,ctypes.POINTER(ctypes.c_void_p)); p.extra=None
    p.kern=None; p.ctx=None
    r=_cu.cuGraphExecKernelNodeSetParams_v2(ctypes.c_void_p(g.raw_cuda_graph_exec()),
                                            ctypes.c_void_p(node),ctypes.byref(p))
    nm=ctypes.c_char_p(); _cu.cuGetErrorName(r,ctypes.byref(nm)); return r,nm.value

def cap(S):
    q=torch.randn(2,4,S,64,device=dev,dtype=bf);k=torch.randn(2,4,S,64,device=dev,dtype=bf);v=torch.randn(2,4,S,64,device=dev,dtype=bf)
    box={}
    nodes,g = capture(lambda: box.__setitem__('o', torch.ops.aten._scaled_dot_product_cudnn_attention.default(q,k,v,None,False)))
    return nodes,g,q,k,v,box

n1,g1,q1,k1,v1,b1 = cap(128)
n5,g5,q5,k5,v5,b5 = cap(512)
K1=[n for n in n1 if n["type"]=="KERNEL"][0]
K5=[n for n in n5 if n["type"]=="KERNEL"][0]
print("S128 func=0x%X mod=0x%X grid=%s" % (K1["func"],K1["module"],K1["grid"]))
print("S512 func=0x%X mod=0x%X grid=%s" % (K5["func"],K5["module"],K5["grid"]))
print("name identical:", K1["name"]==K5["name"])

print("\n--- per-parameter bytes (10 params).  q1=0x%X k1=0x%X v1=0x%X | q5=0x%X" %
      (q1.data_ptr(),k1.data_ptr(),v1.data_ptr(),q5.data_ptr()))
for i,((o,s),b_,c_) in enumerate(zip(K1["paraminfo"],K1["parambytes"],K5["parambytes"])):
    same = b_==c_
    print(f"  p{i} off={o:3d} sz={s:3d} same={same}")
    if not same and s<=16:
        print(f"      S128={b_.hex(' ')}  ->  {int.from_bytes(b_,'little')}")
        print(f"      S512={c_.hex(' ')}  ->  {int.from_bytes(c_,'little')}")

print("\n--- DECISIVE swap: S=128 node keeps its OWN grid/smem/params, only func -> S512's func")
g1.instantiate()
out1=b1['o'][0]
ref=torch.ops.aten._scaled_dot_product_cudnn_attention.default(q1,k1,v1,None,False)[0].clone()
out1.zero_(); torch.cuda.synchronize(); g1.replay(); torch.cuda.synchronize()
print("  baseline replay == eager :", torch.equal(out1,ref))
r,nm = set_exec_kp(g1,K1["node"],K5["func"],K1["grid"],K1["block"],K1["smem"],K1["paraminfo"],K1["parambytes"])
print("  setParams(func=S512's CUfunction) ->",r,nm)
if r==0:
    out1.zero_(); torch.cuda.synchronize(); g1.replay(); torch.cuda.synchronize()
    print("  result still bit-exact vs eager S=128 :", torch.equal(out1,ref))
    print("  max abs diff :", (out1.float()-ref.float()).abs().max().item())

print("\n--- sanity: re-set with the ORIGINAL func and its own params (round-trip)")
r,nm = set_exec_kp(g1,K1["node"],K1["func"],K1["grid"],K1["block"],K1["smem"],K1["paraminfo"],K1["parambytes"])
out1.zero_(); torch.cuda.synchronize(); g1.replay(); torch.cuda.synchronize()
print("  round-trip ok:", torch.equal(out1,ref), r, nm)
print("DONE")
