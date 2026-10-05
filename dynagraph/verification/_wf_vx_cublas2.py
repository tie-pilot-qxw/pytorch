"""Does claim-14's 'capture at the max shape, keep the extra node' trick
generalize to cuBLAS's extra node (splitKreduce)? For ATen reductions the extra
node is a 4B MEMSET semaphore = harmless. splitKreduce is NOT harmless."""
import sys, os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
D=128; K=512; NB=4*1024*1024
bx=torch.randn(NB,device=dev); bo=torch.zeros(NB,device=dev)
W=torch.randn(K,D,device=dev)
def cap(fn,w=3):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g, probe.dump_graph(g.raw_cuda_graph())
def mm(n):
    x=bx[:n*D].view(n,D); o=bo[:n*K].view(n,K)
    return lambda: torch.mm(x,W.t(),out=o)

A,B = 17, 64     # A has 2 nodes (gemm + splitKreduce), B has 1
gA,nA=cap(mm(A)); gB,nB=cap(mm(B))
print("nodes@%d"%A,[x["type"] for x in nA],[demangle(k["name"])[:50] for k in kernels(nA)])
print("nodes@%d"%B,[x["type"] for x in nB],[demangle(k["name"])[:50] for k in kernels(nB)])
gA.instantiate(); exe=gA.raw_cuda_graph_exec()
# patch only the gemm node (node 0) with B's gemm params; leave splitKreduce as recorded
r=probe.copy_node_params(exe,nA[0]["node_handle"],nB[0]["node_handle"],True)
print("patch gemm node ->",r)
ref=(bx[:B*D].view(B,D)@W.t()).clone()
bo.zero_(); torch.cuda.synchronize(); gA.replay(); torch.cuda.synchronize()
got=bo[:B*K].view(B,K); md=(got-ref).abs().max().item()
print(f"capture@{A}(2 nodes) replay as {B}: {'CORRECT' if md/ref.abs().max().item()<1e-4 else 'WRONG'} maxdiff={md:.3e}")
print("  rows written:",(got.abs().sum(-1)>0).sum().item(),"/",B)
