"""INDEPENDENT probe of the cuBLAS line, which the verdict claims is 'not opaque
enough to be a blocker'. Questions:
  Q1 can we even NAME a cuBLAS kernel node? (cuFuncGetName)
  Q2 can we READ its params? (cuFuncGetParamInfo + kernelParams vs extra)
  Q3 node count / kernel identity across the variable-length axis (by NAME)
  Q4 does a blind params+func transplant between two shapes actually compute right?
"""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
D=128; K=512
NB=4*1024*1024
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

SH=[17,64,129,512,1000,2048,4096]
print("### Q1/Q2/Q3: cuBLAS mm(n,128)x(128,512) along n")
info={}
for n in SH:
    g,nodes=cap(mm(n))
    ks=kernels(nodes)
    sig=[]
    for k in ks:
        nb=sum(1 for p in k["params"] if p["bytes"] is not None)
        sig.append((k["name"], len(k["params"]), nb, k["has_kernelParams"], k["has_extra"], k["grid"], k["block"], k["smem"]))
    info[n]=( [x["type"] for x in nodes], sig, nodes)
    print(f"  n={n:5d} types={info[n][0]}")
    for s in sig:
        print(f"        name={s[0][:78]}")
        print(f"        nparams={s[1]} readable={s[2]} kernelParams={s[3]} extra={s[4]} grid={s[5]} block={s[6]} smem={s[7]}")
    del g

print()
names={n:tuple(s[0] for s in info[n][1]) for n in SH}
print("  distinct kernel-NAME sets across shapes:", len(set(names.values())))
for v in sorted(set(names.values()), key=str):
    print("   *", [x[:70] for x in v], "<- shapes", [n for n in SH if names[n]==v])
print("  node counts:", sorted(set(len(info[n][0]) for n in SH)))

print()
print("### Q4: blind transplant between two shapes that have the SAME node structure")
# pick two shapes with identical node type lists
pairs=[(a,b) for i,a in enumerate(SH) for b in SH[i+1:] if info[a][0]==info[b][0]]
print("  same-structure pairs:", pairs[:12])
done=0
for (a,b) in pairs:
    if done>=3: break
    gA,nA=cap(mm(a)); gB,nB=cap(mm(b))
    if [x["type"] for x in nA]!=[x["type"] for x in nB]: continue
    same_names = tuple(k["name"] for k in kernels(nA))==tuple(k["name"] for k in kernels(nB))
    gA.instantiate(); exe=gA.raw_cuda_graph_exec()
    rs=[probe.copy_node_params(exe,x["node_handle"],y["node_handle"],True) for x,y in zip(nA,nB)]
    ref=(bx[:b*D].view(b,D)@W.t()).clone()
    bo.zero_(); torch.cuda.synchronize()
    try:
        gA.replay(); torch.cuda.synchronize()
        got=bo[:b*K].view(b,K)
        md=(got-ref).abs().max().item()
        rel=md/ref.abs().max().item()
        print(f"  capture@{a} -> replay as {b}: same_kernel_names={same_names} copy={rs} "
              f"{'CORRECT' if rel<1e-4 else 'WRONG'} maxdiff={md:.3e} rel={rel:.2e} "
              f"tail_written={bool((got[a:].abs().sum()>0).item()) if b>a else 'n/a'}")
    except Exception as e:
        print(f"  capture@{a} -> replay as {b}: EXCEPTION {type(e).__name__} {str(e)[:120]}")
    del gA,gB
    done+=1
