"""Can the 'extra MEMSET at large shapes' structural switch be absorbed by
capturing at the LARGEST shape and patching only the kernel node?"""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
NB=1024*1024
bx=torch.randn(NB,device=dev); bo=torch.zeros(NB,device=dev); D=128
def cap(fn,w=2):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g, probe.dump_graph(g.raw_cuda_graph())

out=bo[:D]
def f(k): 
    t=bx[:k*D].view(k,D); return lambda: torch.sum(t,0,out=out)

BIG, SMALL = 4096, 512
gB,nB = cap(f(BIG))     # has MEMSET
gS,nS = cap(f(SMALL))   # no MEMSET
print("BIG  nodes:",[x["type"] for x in nB])
print("SMALL nodes:",[x["type"] for x in nS])
gB.instantiate()
exe=gB.raw_cuda_graph_exec()
kB=kernels(nB)[0]; kS=kernels(nS)[0]
print("same func:", kB["func"]==kS["func"], "gridBIG",kB["grid"],"gridSMALL",kS["grid"])
r=probe.copy_node_params(exe, kB["node_handle"], kS["node_handle"])
print("patch BIG-graph kernel node with SMALL params ->", r)
ref = torch.sum(bx[:SMALL*D].view(SMALL,D),0).clone()
bo.zero_(); torch.cuda.synchronize(); gB.replay(); torch.cuda.synchronize()
got = bo[:D]
print("capture@4096, replay as 512:", "CORRECT" if torch.allclose(got,ref,atol=1e-4) else "WRONG",
      " maxdiff", (got-ref).abs().max().item())
# and back to BIG
refB = torch.sum(bx[:BIG*D].view(BIG,D),0).clone()
probe.copy_node_params(exe, kB["node_handle"], kB["node_handle"])
bo.zero_(); torch.cuda.synchronize(); gB.replay(); torch.cuda.synchronize()
print("switch back to 4096:", "CORRECT" if torch.allclose(bo[:D],refB,atol=1e-3) else "WRONG",
      " maxdiff", (bo[:D]-refB).abs().max().item())
