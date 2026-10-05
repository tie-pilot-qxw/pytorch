"""INDEPENDENT re-test of claim 2 (ATen kernel-variant rescue via func swap).
Differences from the original probe:
  * identity by MANGLED NAME, never by CUfunction pointer
  * a NEGATIVE CONTROL: patch params only (keep node's own func) -> must be WRONG,
    which is what proves the func swap is the thing doing the work
  * control for pointer stability: capture the SAME shape twice, compare func ptr
"""
import sys, os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
NB=4*1024*1024
bx=torch.randn(NB,device=dev); bo=torch.zeros(NB,device=dev)
bi=torch.randint(0,512,(NB,),device=dev,dtype=torch.long); D=128
W=torch.randn(512,D,device=dev)

def cap(fn,w=2):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g, probe.dump_graph(g.raw_cuda_graph())

def flat(node):
    tot=max(p["offset"]+p["size"] for p in node["params"])
    b=bytearray(tot)
    for p in node["params"]:
        if p["bytes"] is None: return None
        b[p["offset"]:p["offset"]+p["size"]]=p["bytes"]
    return bytes(b)

def sm(k):
    x=bx[:64*k].view(64,k); o=bo[:64*k].view(64,k)
    return lambda: torch.softmax(x,-1,out=o)

print("### CONTROL 0: is the CUfunction pointer stable across two captures of the SAME shape?")
_,n1=cap(sm(256)); _,n2=cap(sm(256))
a,b=kernels(n1)[0],kernels(n2)[0]
print(f"  same shape twice: func {a['func']:#x} vs {b['func']:#x} -> same_ptr={a['func']==b['func']}  same_name={a['name']==b['name']}")

print()
print("### CASE 1: softmax dim 256 -> 512, SAME ARITY. negative control then func swap.")
gA,nA=cap(sm(256)); gB,nB=cap(sm(512))
ka,kb=kernels(nA)[0],kernels(nB)[0]
print("  A name:",demangle(ka['name'])[:100])
print("  B name:",demangle(kb['name'])[:100])
print(f"  NAMES DIFFER = {ka['name']!=kb['name']}   arity {len(ka['params'])} vs {len(kb['params'])}")
print(f"  grid {ka['grid']}->{kb['grid']} block {ka['block']}->{kb['block']} smem {ka['smem']}->{kb['smem']}")
ref=torch.softmax(bx[:64*512].view(64,512),-1).clone()

# --- negative control: same flat bytes + same launch cfg, but KEEP A's func
gA.instantiate(); exe=gA.raw_cuda_graph_exec()
probe.set_exec_kernel_params(exe, ka["node_handle"], flat(kb), kb["grid"][0],kb["grid"][1],kb["grid"][2],
                             kb["block"][0],kb["block"][1],kb["block"][2], kb["smem"])
bo.zero_(); torch.cuda.synchronize(); gA.replay(); torch.cuda.synchronize()
got=bo[:64*512].view(64,512)
md=(got-ref).abs().max().item()
print(f"  [NEG CONTROL params-only, keep func A] {'CORRECT' if md<1e-5 else 'WRONG'} maxdiff={md:.3e}")

# --- now the real thing: force func swap
r=probe.copy_node_params(exe, ka["node_handle"], kb["node_handle"], True)
print("  [func swap] copy_node_params ->", r)
bo.zero_(); torch.cuda.synchronize(); gA.replay(); torch.cuda.synchronize()
got=bo[:64*512].view(64,512); md=(got-ref).abs().max().item()
print(f"  [FUNC SWAP] {'CORRECT' if md<1e-5 else 'WRONG'} maxdiff={md:.3e}")
# and confirm the node in the exec really is B now, by row-sum sanity
print(f"  row sums after swap: min={got.sum(-1).min().item():.6f} max={got.sum(-1).max().item():.6f} (want 1.0)")

print()
print("### CASE 2: softmax 256 -> 4096, DIFFERENT ARITY + different template family")
gA2,nA2=cap(sm(256)); gB2,nB2=cap(sm(4096))
ka2,kb2=kernels(nA2)[0],kernels(nB2)[0]
print("  A:",demangle(ka2['name'])[:100]); print("  B:",demangle(kb2['name'])[:100])
print(f"  NAMES DIFFER={ka2['name']!=kb2['name']} arity {len(ka2['params'])}->{len(kb2['params'])}"
      f" block {ka2['block']}->{kb2['block']} smem {ka2['smem']}->{kb2['smem']}"
      f" module {ka2['module']:#x} vs {kb2['module']:#x} same_module={ka2['module']==kb2['module']}")
gA2.instantiate()
r=probe.copy_node_params(gA2.raw_cuda_graph_exec(), ka2["node_handle"], kb2["node_handle"], True)
print("  copy ->",r)
ref2=torch.softmax(bx[:64*4096].view(64,4096),-1).clone()
bo.zero_(); torch.cuda.synchronize(); gA2.replay(); torch.cuda.synchronize()
got2=bo[:64*4096].view(64,4096); md2=(got2-ref2).abs().max().item()
print(f"  [FUNC SWAP 256->4096] {'CORRECT' if md2<1e-5 else 'WRONG'} maxdiff={md2:.3e}"
      f"  rowsum[0]={got2.sum(-1)[0].item():.6f}")

print()
print("### CASE 3: index_select 16 -> 64, cross-CUmodule check")
def isel(k):
    i=bi[:k]%512; o=bo[:k*D].view(k,D)
    return lambda: torch.index_select(W,0,i,out=o)
gA3,nA3=cap(isel(16)); gB3,nB3=cap(isel(64))
k3a,k3b=kernels(nA3)[0],kernels(nB3)[0]
print("  A:",demangle(k3a['name'])[:100]); print("  B:",demangle(k3b['name'])[:100])
print(f"  NAMES DIFFER={k3a['name']!=k3b['name']} modules {k3a['module']:#x} vs {k3b['module']:#x}"
      f" CROSS_MODULE={k3a['module']!=k3b['module']} arity {len(k3a['params'])}->{len(k3b['params'])}")
gA3.instantiate()
print("  copy ->",probe.copy_node_params(gA3.raw_cuda_graph_exec(), k3a["node_handle"], k3b["node_handle"], True))
ref3=torch.index_select(W,0,bi[:64]%512).clone()
bo.zero_(); torch.cuda.synchronize(); gA3.replay(); torch.cuda.synchronize()
got3=bo[:64*D].view(64,D); md3=(got3-ref3).abs().max().item()
print(f"  [FUNC SWAP index_select] {'CORRECT' if md3<1e-6 else 'WRONG'} maxdiff={md3:.3e} nonzero_rows={(got3.abs().sum(-1)>0).sum().item()}/64")
