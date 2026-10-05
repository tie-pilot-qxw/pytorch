"""CONTROL for the previous test: does a device-side grid/param update take effect on a
HOST cudaGraphLaunch replay at all?  Use an aten elementwise kernel where truncating the
grid MUST leave part of the output unwritten."""
import ctypes, os
import torch
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
def pinfo(f):
    o=[];off=ctypes.c_size_t();sz=ctypes.c_size_t()
    for i in range(64):
        if libcuda.cuFuncGetParamInfo(ctypes.c_void_p(f),ctypes.c_size_t(i),ctypes.byref(off),ctypes.byref(sz))!=0: break
        o.append((i,off.value,sz.value))
    return o

DEV="cuda"; N=1<<20
x=torch.randn(N,device=DEV)
f=lambda: torch.relu(x)
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): f()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g=torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(g): out=f()
torch.cuda.synchronize()
kn=[z for z in nodes_of(g.raw_cuda_graph()) if ntype(z)==0][0]
p=KNP(); libcuda.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(kn),ctypes.byref(p))
print(f"aten kernel: {fname(p.func)[:80]}")
print(f"grid={(p.gx,p.gy,p.gz)} block={(p.bx,p.by,p.bz)} param_info={pinfo(p.func)}")
print(f"param0 (numel) = {int.from_bytes(bytes((ctypes.c_ubyte*4).from_address(p.kernelParams[0])),'little')}")

buf=(ctypes.c_ubyte*256)(); ctypes.memset(buf,0,256)
ctypes.cast(buf,ctypes.POINTER(ctypes.c_int))[0]=1
rc=libcuda.cuGraphKernelNodeSetAttribute(ctypes.c_void_p(kn),ctypes.c_int(13),ctypes.byref(buf))
devnode=ctypes.cast(ctypes.byref(buf,8),ctypes.POINTER(ctypes.c_void_p))[0]
print(f"opt-in rc={en(rc)} devNode=0x{(devnode or 0):x}")
g.instantiate(); ex=g.raw_cuda_graph_exec()
libcuda.cuGraphUpload(ctypes.c_void_p(ex), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
torch.cuda.synchronize()
ref=torch.relu(x); torch.cuda.synchronize()
out.fill_(float("nan")); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
print(f"baseline: exact={bool(torch.equal(out,ref))} nan={int(torch.isnan(out).sum())}")

r=dev.launch_set_grid(ctypes.c_void_p(devnode),1,1,1); torch.cuda.synchronize()
out.fill_(float("nan")); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
nn=int(torch.isnan(out).sum())
print(f"device-side SetGridDim(1,1,1) rc={r}: NaN left {nn}/{N}  -> grid update visible to host replay: {nn>0}")

dev.launch_set_grid(ctypes.c_void_p(devnode),p.gx,p.gy,p.gz); torch.cuda.synchronize()
poff=ctypes.c_size_t(); psz=ctypes.c_size_t()
libcuda.cuFuncGetParamInfo(ctypes.c_void_p(p.func),ctypes.c_size_t(0),ctypes.byref(poff),ctypes.byref(psz))
r2=dev.launch_set_param(ctypes.c_void_p(devnode), ctypes.c_size_t(poff.value), ctypes.c_uint(N//4))
torch.cuda.synchronize()
out.fill_(float("nan")); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
nn2=int(torch.isnan(out).sum())
print(f"device-side SetParam(offset={poff.value}, numel={N//4}) rc={r2}: NaN left {nn2}/{N} "
      f"-> param update visible to host replay: {nn2>0}")
