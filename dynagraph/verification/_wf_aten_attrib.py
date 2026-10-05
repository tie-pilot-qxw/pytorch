"""(A) node attribution: can we tell, at capture time, which graph nodes a given
ATen call produced?  Uses cuStreamGetCaptureInfo + cuGraphGetNodes set-difference.
Also pins down the sum/norm structural switch."""
import sys, os, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
NB=1024*1024
bx=torch.randn(NB,device=dev); by=torch.randn(NB,device=dev); bo=torch.zeros(NB,device=dev)
bi=torch.randint(0,512,(NB,),device=dev,dtype=torch.long)
D=128; W=torch.randn(512,D,device=dev); G=torch.randn(D,device=dev); Bt=torch.randn(D,device=dev)
n=512
x=bx[:n*D].view(n,D); y=by[:n*D].view(n,D); o=bo[:n*D].view(n,D); idx=bi[:n]%512

def warm():
    torch.add(x,y,out=o); F.layer_norm(x,(D,),G,Bt,1e-5); torch.index_select(W,0,idx); torch.sort(bx[:4000])
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): warm()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()

g=torch.cuda.CUDAGraph(keep_graph=True)
attrib={}
with torch.cuda.graph(g):
    st = torch.cuda.current_stream().cuda_stream
    def mark(tag, fn):
        before=set(probe.capture_node_set(st))
        r=fn()
        after=set(probe.capture_node_set(st))
        attrib[tag]=after-before
        return r
    mark("add",        lambda: torch.add(x,y,out=o))
    mark("layer_norm", lambda: F.layer_norm(x,(D,),G,Bt,1e-5))
    mark("index_sel",  lambda: torch.index_select(W,0,idx))
    mark("sort4000",   lambda: torch.sort(bx[:4000]))
torch.cuda.synchronize()
nodes=probe.dump_graph(g.raw_cuda_graph())
print(f"whole graph: {len(nodes)} nodes")
allh=set(x["node_handle"] for x in nodes)
tot=0
for tag,hs in attrib.items():
    tot+=len(hs)
    print(f"  {tag:12s}: {len(hs)} node(s), all present in final graph = {hs<=allh}")
    for h in sorted(hs):
        d=probe.describe_node_handle(h)
        nm = demangle(d.get("name","")).replace("at::native::","")[:90] if d["type"]=="KERNEL" else ""
        print(f"        {d['type']:8s} {nm}")
print(f"  attributed {tot}/{len(nodes)} nodes; unattributed = {len(allh - set().union(*attrib.values()))}")

print()
print("## sum(dim=0) / norm() structural switch")
for tag,mk,shapes in [("sum(dim=0)", lambda k:(lambda t=bx[:k*D].view(k,D): t.sum(0)), [512,1000,2048,4096]),
                      ("norm()",     lambda k:(lambda t=bx[:k*D].view(k,D): t.norm()),  [512,1000,2048,4096])]:
    for k in shapes:
        s2=torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
        f=mk(k)
        with torch.cuda.stream(s2):
            for _ in range(2): f()
        torch.cuda.current_stream().wait_stream(s2); torch.cuda.synchronize()
        gg=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(gg): f()
        torch.cuda.synchronize()
        nn=probe.dump_graph(gg.raw_cuda_graph())
        ks=kernels(nn)
        extra=[m for m in nn if m["type"]=="MEMSET"]
        print(f"  {tag} n={k:5d}: {[m['type'] for m in nn]} "
              f"funcs={[hex(m['func']) for m in ks]} grid={[m['grid'] for m in ks]} "
              + (f"memset {extra[0]['memset_width']}B" if extra else ""))
        del gg
