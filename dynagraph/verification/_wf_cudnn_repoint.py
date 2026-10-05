"""The strategy that would actually make cuDNN work inside DynaGraph:
   capture the op ONCE MORE at the new shape in a throwaway graph ("proxy capture"),
   then copy that node's opaque param image onto the real graph's node, rewriting only
   the slots that are known to be buffer addresses so they point at OUR buffers.
Slots are identified empirically: capture the same shape twice at different addresses,
and every 8-byte slot that moves is a buffer slot (delta kept, so ptr+offset survives)."""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from _wf_cudnn_lib import libcuda, graph_nodes, node_type, kernel_params, func_name, param_info, KNP

DEV="cuda"; DT=torch.float16
def mk(n): return torch.randn(n,64,56,56,device=DEV,dtype=DT).to(memory_format=torch.channels_last)
def mkw():  return torch.randn(64,64,3,3,device=DEV,dtype=DT).to(memory_format=torch.channels_last)

HOLD=[]
def cap(x,w,b):
    f=lambda: F.conv2d(x[:b],w,padding=1)
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): out=f()
    torch.cuda.synchronize(); HOLD.append((g,out))
    n=graph_nodes(g.raw_cuda_graph())
    assert len(n)==1 and node_type(n[0])=="KERNEL", [node_type(z) for z in n]
    p=kernel_params(n[0]); pi=param_info(p.func)
    img=[bytearray((ctypes.c_ubyte*sz).from_address(p.kernelParams[i])) for i,_,sz in pi]
    return g,out,n[0],p,pi,img

X1,W1 = mk(16), mkw()
X2,W2 = mk(16), mkw()
X3,W3 = mk(16), mkw()
X4,W4 = mk(16), mkw()                      # the target buffers; never captured
g1,o1,nd1,p1,pi1,im1 = cap(X1,W1,13)       # the graph we will keep and patch (shape 13!)
g2,o2,nd2,p2,pi2,im2 = cap(X2,W2,16)       # proxy capture at the new shape
g3,o3,nd3,p3,pi3,im3 = cap(X3,W3,16)       # second proxy, different addresses
print(f"kept graph kernel (b=13): {func_name(p1.func)[:80]}")
print(f"proxy      kernel (b=16): {func_name(p2.func)[:80]}")

roles2 = {"x":X2.data_ptr(), "w":W2.data_ptr(), "out":o2.data_ptr()}
roles3 = {"x":X3.data_ptr(), "w":W3.data_ptr(), "out":o3.data_ptr()}
roles4 = {"x":X4.data_ptr(), "w":W4.data_ptr(), "out":torch.empty_like(o2).data_ptr()}
o4 = None
out4 = torch.empty(16,64,56,56,device=DEV,dtype=DT).to(memory_format=torch.channels_last)
roles4["out"] = out4.data_ptr()

slots=[]; unknown=[]
for k,(pa,pb) in enumerate(zip(im2,im3)):
    for off in range(0,len(pa)-7,8):
        va=int.from_bytes(pa[off:off+8],"little"); vb=int.from_bytes(pb[off:off+8],"little")
        if va==vb: continue
        hit=None
        for r in roles2:
            if va-roles2[r] == vb-roles3[r] and 0 <= va-roles2[r] < (1<<28):
                hit=(r, va-roles2[r]); break
        if hit: slots.append((k,off,hit[0],hit[1]))
        else: unknown.append((k,off,va,vb))
print(f"\nbuffer slots found by the two proxy captures: {len(slots)}")
for k,off,r,d in slots: print(f"   param{k}[{off}:{off+8}] = {r}_ptr + {d}")
print(f"slots that moved but matched no known buffer: {len(unknown)}")
for k,off,va,vb in unknown: print(f"   param{k}[{off}:{off+8}] {hex(va)} -> {hex(vb)}  (left as proxy's value)")

img = [bytearray(b) for b in im2]
for k,off,r,d in slots:
    img[k][off:off+8] = (roles4[r]+d).to_bytes(8,"little")

g1.instantiate(); ex=g1.raw_cuda_graph_exec()
bufs=[]; ptrs=(ctypes.c_void_p*len(img))()
for i,b in enumerate(img):
    cb=(ctypes.c_ubyte*len(b)).from_buffer_copy(bytes(b)); bufs.append(cb)
    ptrs[i]=ctypes.cast(cb,ctypes.c_void_p)
new=KNP(); new.func=p2.func
new.gridDimX,new.gridDimY,new.gridDimZ = p2.gridDimX,p2.gridDimY,p2.gridDimZ
new.blockDimX,new.blockDimY,new.blockDimZ = p2.blockDimX,p2.blockDimY,p2.blockDimZ
new.sharedMemBytes = p2.sharedMemBytes
new.kernelParams = ctypes.cast(ptrs,ctypes.POINTER(ctypes.c_void_p))
new.extra=None; new.kern=None; new.ctx=None
rc = libcuda.cuGraphExecKernelNodeSetParams_v2(ctypes.c_void_p(ex), ctypes.c_void_p(nd1), ctypes.byref(new))
print(f"\ncuGraphExecKernelNodeSetParams rc={rc}")

ref = F.conv2d(X4[:16], W4, padding=1)
out4.fill_(float("nan")); torch.cuda.synchronize()
g1.replay(); torch.cuda.synchronize()
bad = torch.isnan(out4).any().item()
print(f"replay of the b=13 graph, now running the b=16 kernel on buffers it never saw:")
print(f"   any_nan={bad}  allclose_to_eager={torch.allclose(out4.float(),ref.float(),rtol=2e-2,atol=2e-2)}"
      f"  max_abs_diff={(out4.float()-ref.float()).abs().max().item() if not bad else float('nan')}")
print(f"   (the ORIGINAL b=13 output buffer was left untouched: "
      f"still all-equal to itself = {torch.isfinite(o1.float()).all().item()})")
