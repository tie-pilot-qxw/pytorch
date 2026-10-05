"""(1) claim 11's index_put_ acc=False: script on disk gives WRONG. Is the
    'duplicate indices' explanation right? Re-test with UNIQUE indices.
(2) claim 6's MEMCPY: in their script the MEMCPY came from an added o.copy_(),
    not from layer_norm. Find an op that emits MEMCPY nodes ITSELF (sort) and
    test whether skipping the MEMCPY patch really breaks it."""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
NB=2*1024*1024; D=64
bx=torch.randn(NB,device=dev); bo=torch.zeros(NB,device=dev)
bv=torch.zeros(NB,device=dev); bidx=torch.zeros(NB,device=dev,dtype=torch.long)
def cap(fn,w=3):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g, probe.dump_graph(g.raw_cuda_graph())

print("### (1) index_put_ acc=False, UNIQUE indices, 512 -> 1024")
uniq = torch.randperm(2048, device=dev)   # all distinct
dst  = bo[:2048*D].view(2048,D)
def ip(n): 
    i=uniq[:n]; v=bx[:n*D].view(n,D)
    return lambda: dst.index_put_((i,), v, accumulate=False)
NA,NB_=512,1024
gA,nA=cap(ip(NA)); gB,nB=cap(ip(NB_))
print("  nodes:",[x['type'] for x in nA],[x['type'] for x in nB],
      " same kernel name:",tuple(k['name'] for k in kernels(nA))==tuple(k['name'] for k in kernels(nB)))
gA.instantiate(); exe=gA.raw_cuda_graph_exec()
print("  copy ->",set(probe.copy_node_params(exe,a['node_handle'],b['node_handle'],True) for a,b in zip(nA,nB)))
ref=torch.zeros(2048,D,device=dev); ref.index_put_((uniq[:NB_],), bx[:NB_*D].view(NB_,D))
dst.zero_(); torch.cuda.synchronize(); gA.replay(); torch.cuda.synchronize()
md=(dst-ref).abs().max().item()
print(f"  UNIQUE indices: {'CORRECT' if md<1e-6 else 'WRONG'} maxdiff={md:.3e}")
# control: same test but with DUPLICATE indices (what the saved script does)
dup = torch.randint(0,2048,(2048,),device=dev)
def ipd(n):
    i=dup[:n]; v=bx[:n*D].view(n,D)
    return lambda: dst.index_put_((i,), v, accumulate=False)
gA2,nA2=cap(ipd(NA)); gB2,nB2=cap(ipd(NB_))
gA2.instantiate()
for a,b in zip(nA2,nB2): probe.copy_node_params(gA2.raw_cuda_graph_exec(),a['node_handle'],b['node_handle'],True)
ref2=torch.zeros(2048,D,device=dev); ref2.index_put_((dup[:NB_],), bx[:NB_*D].view(NB_,D))
dst.zero_(); torch.cuda.synchronize(); gA2.replay(); torch.cuda.synchronize()
md2=(dst-ref2).abs().max().item()
# is the EAGER path itself nondeterministic here?
r3=torch.zeros(2048,D,device=dev); r3.index_put_((dup[:NB_],), bx[:NB_*D].view(NB_,D))
r4=torch.zeros(2048,D,device=dev); r4.index_put_((dup[:NB_],), bx[:NB_*D].view(NB_,D))
print(f"  DUPLICATE indices: maxdiff={md2:.3e}  | eager-vs-eager (same call twice) maxdiff="
      f"{(r3-r4).abs().max().item():.3e}  n_dup={NB_-len(set(dup[:NB_].tolist()))}")

print()
print("### (2) sort: does the op ITSELF emit MEMCPY nodes, and do they carry shape?")
def srt(n):
    x=bx[:n]; v=bv[:n]; ii=bidx[:n]
    return lambda: torch.sort(x,out=(v,ii))
for (a,b) in [(1024,2048)]:
    gA,nA=cap(srt(a)); gB,nB=cap(srt(b))
    print(f"  nodes@{a}:",[x['type'] for x in nA])
    print(f"  nodes@{b}:",[x['type'] for x in nB])
    for i,(x,y) in enumerate(zip(nA,nB)):
        if x['type']=='MEMCPY':
            print(f"    node{i} MEMCPY bytes {x.get('memcpy_bytes')} -> {y.get('memcpy_bytes')}  SHAPE-DEPENDENT={x.get('memcpy_bytes')!=y.get('memcpy_bytes')}")
        if x['type']=='MEMSET':
            print(f"    node{i} MEMSET w={x.get('memset_width')} -> {y.get('memset_width')}")
    ref=torch.sort(bx[:b])[0].clone()
    # 2a: patch KERNEL nodes only
    gA.instantiate(); exe=gA.raw_cuda_graph_exec()
    for x,y in zip(nA,nB):
        if x['type']=='KERNEL': probe.copy_node_params(exe,x['node_handle'],y['node_handle'],True)
    bv.zero_(); torch.cuda.synchronize(); gA.replay(); torch.cuda.synchronize()
    md=(bv[:b]-ref).abs().max().item()
    print(f"  [KERNEL nodes only]      {'CORRECT' if md<1e-6 else 'WRONG'} maxdiff={md:.3e}")
    # 2b: patch everything
    for x,y in zip(nA,nB): probe.copy_node_params(exe,x['node_handle'],y['node_handle'],True)
    bv.zero_(); torch.cuda.synchronize(); gA.replay(); torch.cuda.synchronize()
    md=(bv[:b]-ref).abs().max().item()
    print(f"  [all nodes incl MEMCPY]  {'CORRECT' if md<1e-6 else 'WRONG'} maxdiff={md:.3e}")
