import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck
K=N=512; M=3850
x=torch.randn(M,K,device="cuda"); w=torch.randn(K,N,device="cuda"); b=torch.randn(N,device="cuda"); y=torch.empty(M,N,device="cuda")
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): torch.addmm(b,x,w,out=y)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g=torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(g): torch.addmm(b,x,w,out=y)
torch.cuda.synchronize()
raw=g.raw_cuda_graph(); c=ck(cr.cudaGraphGetNodes(raw))[1]; nds=ck(cr.cudaGraphGetNodes(raw,c))[0]
print("nodes",c)
for nd in nds:
    t=int(getattr(ck(cr.cudaGraphNodeGetType(nd)),"value",0)); print("type",t, cr.cudaGraphNodeType(t))
    if t==2:
        p=ck(cr.cudaGraphMemsetNodeGetParams(nd))
        print("   memset dst=",hex(int(p.dst)),"value=",p.value,"elementSize=",p.elementSize,"width=",p.width,"height=",p.height)
        print("   y.data_ptr=",hex(y.data_ptr()),"y bytes=",y.numel()*4)
