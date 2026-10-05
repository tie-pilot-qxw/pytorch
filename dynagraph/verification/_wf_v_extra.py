import ctypes, torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck
K=N=512; M=947
x=torch.randn(M,K,device="cuda",dtype=torch.bfloat16); w=torch.randn(K,N,device="cuda",dtype=torch.bfloat16)
b=torch.randn(N,device="cuda",dtype=torch.bfloat16); y=torch.empty(M,N,device="cuda",dtype=torch.bfloat16)
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): torch.addmm(b,x,w,out=y)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g=torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(g): torch.addmm(b,x,w,out=y)
torch.cuda.synchronize()
raw=g.raw_cuda_graph(); c=ck(cr.cudaGraphGetNodes(raw))[1]; nd=ck(cr.cudaGraphGetNodes(raw,c))[0][0]
dp=ck(cd.cuGraphKernelNodeGetParams(nd))
print("kernelParams=",int(dp.kernelParams)," extra=",hex(int(dp.extra)))
ents=(ctypes.c_void_p*8).from_address(int(dp.extra))
print("extra entries:",[hex(ents[i]) if ents[i] else "0x0" for i in range(6)])
print("CU_LAUNCH_PARAM_BUFFER_POINTER=",int(cd.CU_LAUNCH_PARAM_BUFFER_POINTER),
      " BUFFER_SIZE=",int(cd.CU_LAUNCH_PARAM_BUFFER_SIZE))
sz=ctypes.c_size_t.from_address(ents[1]).value
print("entry 0 tag=",hex(ents[0]),"-> if it is SIZE then size=",sz)
