import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
NB=4*1024*1024
bx=torch.randn(NB,device=dev); by=torch.randn(NB,device=dev); bo=torch.zeros(NB,device=dev)
D=128
def cap(fn,w=2):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g, probe.dump_graph(g.raw_cuda_graph())

print("## (C) hard limit: force a func change through cuGraphExecKernelNodeSetParams")
def sm(k):
    x=bx[:64*k].view(64,k); o=bo[:64*k].view(64,k)
    return lambda: torch.softmax(x,-1,out=o)
gA,nA=cap(sm(256)); gB,nB=cap(sm(512))
a,b=kernels(nA)[0],kernels(nB)[0]
print("  funcA",hex(a["func"]),"funcB",hex(b["func"]))
print("  A:",demangle(a["name"])[:100]); print("  B:",demangle(b["name"])[:100])
gA.instantiate()
print("  forced SetParams(A.node, B.params) ->", probe.copy_node_params(gA.raw_cuda_graph_exec(), a["node_handle"], b["node_handle"], True))
# same-func control from the same call site
gC,nC=cap(sm(200))
c=kernels(nC)[0]
print("  same-template neighbour dim=200 func", hex(c["func"]), "== A:", c["func"]==a["func"])
print("  SetParams(A.node, C.params) ->", probe.copy_node_params(gA.raw_cuda_graph_exec(), a["node_handle"], c["node_handle"]))
ref=torch.softmax(bx[:64*200].view(64,200),-1).clone()
bo.zero_(); torch.cuda.synchronize(); gA.replay(); torch.cuda.synchronize()
print("  capture@256 replayed as 200:", "CORRECT" if torch.allclose(bo[:64*200].view(64,200),ref,atol=1e-6) else "WRONG")

print()
print("## SYNTHESIS (no capture at the target shape): patch only the naked scalar + grid")
n0, n1 = 1024, 3457          # 3457 never captured
x=bx[:n1*D].view(n1,D); y=by[:n1*D].view(n1,D); o=bo[:n1*D].view(n1,D)
gS,nS = cap(lambda: torch.add(bx[:n0*D].view(n0,D), by[:n0*D].view(n0,D), out=bo[:n0*D].view(n0,D)))
k=kernels(nS)[0]
print("  captured at numel", n0*D, "params:", [(p["index"],p["size"]) for p in k["params"]])
total=max(p["offset"]+p["size"] for p in k["params"])
buf=bytearray(total)
for p in k["params"]: buf[p["offset"]:p["offset"]+p["size"]]=p["bytes"]
newnumel = n1*D
buf[0:4]=newnumel.to_bytes(4,"little")     # p0 = int N, computed host-side from the shape alone
# grid: block_work_size = 128 threads * 4 elems/thread * (vec factor already folded into recorded grid)
import math
recorded_grid = k["grid"][0]; recorded_numel = n0*D
per_block = recorded_numel // recorded_grid
grid = math.ceil(newnumel/per_block)
print(f"  recorded grid={recorded_grid} -> elems/block={per_block}; synthesized grid={grid}")
gS.instantiate()
probe.set_exec_kernel_params(gS.raw_cuda_graph_exec(), k["node_handle"], bytes(buf), grid,1,1, *k["block"], k["smem"])
ref = (bx[:n1*D].view(n1,D)+by[:n1*D].view(n1,D)).clone()
bo.zero_(); torch.cuda.synchronize(); gS.replay(); torch.cuda.synchronize()
got=bo[:n1*D].view(n1,D)
print("  synthesized replay at n=3457:", "CORRECT" if torch.equal(got,ref) else "WRONG",
      " (tail beyond n0 also written:", bool(torch.equal(got[n0:],ref[n0:])), ")")
