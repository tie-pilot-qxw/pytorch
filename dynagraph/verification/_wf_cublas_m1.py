import ctypes, math, struct, torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck
K=N=512; MMAX=1024
torch.manual_seed(0)
x=torch.randn(MMAX,K,device="cuda"); w=torch.randn(K,N,device="cuda")
b=torch.randn(N,device="cuda"); y=torch.zeros(MMAX,N,device="cuda")
def run(M): return torch.addmm(b,x[:M],w,out=y[:M])
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): run(947)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g=torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(g): run(947)
torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
raw=g.raw_cuda_graph(); ge=g.raw_cuda_graph_exec()
nn=ck(cr.cudaGraphGetNodes(raw))[1]; nd=ck(cr.cudaGraphGetNodes(raw,nn))[0][0]
dp=ck(cd.cuGraphKernelNodeGetParams(nd))
_,sz=ck(cd.cuFuncGetParamInfo(dp.func,0)); sz=int(sz)
orig=bytes((ctypes.c_ubyte*sz).from_address((ctypes.c_void_p*1).from_address(int(dp.kernelParams))[0]))
keep=[]
def patch(M):
    buf=bytearray(orig); struct.pack_into("<i",buf,0,M); struct.pack_into("<i",buf,12,math.ceil(M/64))
    cb=(ctypes.c_ubyte*sz).from_buffer(buf); arr=(ctypes.c_void_p*1)(ctypes.cast(cb,ctypes.c_void_p))
    keep.append((buf,cb,arr))
    p=cd.CUDA_KERNEL_NODE_PARAMS(); p.func=dp.func
    p.gridDimX=math.ceil(M/64)*(N//64); p.gridDimY=dp.gridDimY; p.gridDimZ=dp.gridDimZ
    p.blockDimX=dp.blockDimX; p.blockDimY=dp.blockDimY; p.blockDimZ=dp.blockDimZ
    p.sharedMemBytes=dp.sharedMemBytes; p.kernelParams=ctypes.addressof(arr); p.extra=0
    ck(cd.cuGraphExecKernelNodeSetParams(ge,nd,p))
print("M   patched_vs_fp64   eager_vs_fp64   patched_vs_eager   eager_kernel")
for M in (1,2,16,64):
    y.zero_(); patch(M); g.replay(); torch.cuda.synchronize()
    got=y[:M].double()
    ref64=(b.double()+x[:M].double()@w.double())
    eager=torch.addmm(b,x[:M],w).double()
    den=ref64.abs().max().item()
    print(f"{M:<4}{(got-ref64).abs().max().item()/den:<18.3g}"
          f"{(eager-ref64).abs().max().item()/den:<16.3g}"
          f"{(got-eager).abs().max().item()/den:<19.3g}")
print("\ntorch.backends.cuda.matmul.allow_tf32 =", torch.backends.cuda.matmul.allow_tf32,
      " fp32_precision =", torch.backends.cuda.matmul.fp32_precision)
