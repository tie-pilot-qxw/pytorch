"""Does the DEVICE-SIDE graph update actually work on an EXTERN (cuDNN) node,
after opting the captured node in with cuGraphKernelNodeSetAttribute?
Observable: change the cuDNN SDPA node's grid from the device, replay, and see the
output change; then restore it and see bit-exact agreement with eager come back."""
import ctypes, os
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
libcuda=ctypes.CDLL("libcuda.so.1")
# Build libsetgrid.so from setgrid.cu (next to this script) first:
#   mkdir -p $DG_OUT/_wf_vrfy_build && nvcc -shared -Xcompiler -fPIC -arch=sm_90 -rdc=true \
#     setgrid.cu -o $DG_OUT/_wf_vrfy_build/libsetgrid.so -lcudadevrt
dev=ctypes.CDLL(os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "_wf_vrfy_build", "libsetgrid.so"))
dev.launch_set_grid.restype=ctypes.c_int
dev.launch_set_grid.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_uint,ctypes.c_uint]
dev.launch_set_param.restype=ctypes.c_int
dev.launch_set_param.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint]

class KNP(ctypes.Structure):
    _fields_=[("func",ctypes.c_void_p),("gx",ctypes.c_uint),("gy",ctypes.c_uint),("gz",ctypes.c_uint),
              ("bx",ctypes.c_uint),("by",ctypes.c_uint),("bz",ctypes.c_uint),("smem",ctypes.c_uint),
              ("kernelParams",ctypes.POINTER(ctypes.c_void_p)),("extra",ctypes.POINTER(ctypes.c_void_p)),
              ("kern",ctypes.c_void_p),("ctx",ctypes.c_void_p)]
def en(rc):
    s=ctypes.c_char_p(); libcuda.cuGetErrorName(ctypes.c_int(rc),ctypes.byref(s))
    return f"{rc}({s.value.decode() if s.value else '?'})"
def nodes_of(g):
    n=ctypes.c_size_t(0); libcuda.cuGraphGetNodes(ctypes.c_void_p(g),None,ctypes.byref(n))
    a=(ctypes.c_void_p*n.value)(); libcuda.cuGraphGetNodes(ctypes.c_void_p(g),a,ctypes.byref(n))
    return [a[i] for i in range(n.value)]
def ntype(nd):
    t=ctypes.c_int(0); libcuda.cuGraphNodeGetType(ctypes.c_void_p(nd),ctypes.byref(t)); return t.value
def fname(f):
    s=ctypes.c_char_p()
    return s.value.decode(errors="replace") if libcuda.cuFuncGetName(ctypes.byref(s),ctypes.c_void_p(f))==0 else "?"

DEV="cuda"; DT=torch.bfloat16; B,H,L,D=2,8,512,64
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
kn=[z for z in nds if ntype(z)==0][0]
p=KNP(); libcuda.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(kn),ctypes.byref(p))
print(f"cuDNN kernel: {fname(p.func)[:85]}")
print(f"captured grid={(p.gx,p.gy,p.gz)} block={(p.bx,p.by,p.bz)}")

buf=(ctypes.c_ubyte*256)(); ctypes.memset(buf,0,256)
ctypes.cast(buf,ctypes.POINTER(ctypes.c_int))[0]=1
rc=libcuda.cuGraphKernelNodeSetAttribute(ctypes.c_void_p(kn), ctypes.c_int(13), ctypes.byref(buf))
devnode=ctypes.cast(ctypes.byref(buf,8),ctypes.POINTER(ctypes.c_void_p))[0]
print(f"opt-in rc={en(rc)}  devNode=0x{(devnode or 0):x}")
g.instantiate()
ex=g.raw_cuda_graph_exec()
print("uploaded:", en(libcuda.cuGraphUpload(ctypes.c_void_p(ex), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))))
torch.cuda.synchronize()

ref=f(); torch.cuda.synchronize()
out.fill_(float("nan")); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
print(f"baseline replay: bit_exact_vs_eager={bool(torch.equal(out,ref))} any_nan={bool(torch.isnan(out).any())}")

r=dev.launch_set_grid(ctypes.c_void_p(devnode), 1,1,1)
torch.cuda.synchronize()
print(f"\ndevice-side cudaGraphKernelNodeSetGridDim(node,(1,1,1)) -> cudaError={r}")
out.fill_(float("nan")); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
nn=int(torch.isnan(out).sum().item()); tot=out.numel()
print(f"  replay after device-side grid=1: NaN elements {nn}/{tot} ({100*nn/tot:.1f}% unwritten) "
      f"bit_exact_vs_eager={bool(torch.equal(out,ref))}")

r2=dev.launch_set_grid(ctypes.c_void_p(devnode), p.gx,p.gy,p.gz)
torch.cuda.synchronize()
print(f"\ndevice-side restore grid={(p.gx,p.gy,p.gz)} -> cudaError={r2}")
out.fill_(float("nan")); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
print(f"  replay after restore: bit_exact_vs_eager={bool(torch.equal(out,ref))} "
      f"any_nan={bool(torch.isnan(out).any())}")
print(f"\n-> DEVICE-SIDE UPDATE OF AN EXTERN/cuDNN NODE WORKS: "
      f"{r==0 and r2==0 and bool(torch.equal(out,ref))}")
