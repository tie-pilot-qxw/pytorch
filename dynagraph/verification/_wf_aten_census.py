"""E9 census: for ~30 common fallback ATen ops, sweep the VARIABLE-LENGTH axis
(the axis DynaGraph actually varies) and classify:
  STABLE       - one kernel signature, identical grid too (rare)
  LAUNCH-ONLY  - same node types & same funcs; only grid/block/smem/params vary  -> patchable
  KERNEL-SWITCH- same node count but some func changes                           -> NOT patchable
  STRUCT-SWITCH- node count / types change                                       -> NOT patchable
Also: for the LAUNCH-ONLY ops, is the shape visible as a naked scalar param?
"""
import os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
NB=4*1024*1024
bx=torch.randn(NB,device=dev); by=torch.randn(NB,device=dev); bo=torch.zeros(NB,device=dev)
bi=torch.randint(0,512,(NB,),device=dev,dtype=torch.long)
bb=(torch.rand(NB,device=dev)>0.5)
D=128
SHAPES=[17,64,129,512,1000,2048,4096]

def cap(fn,w=2):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g, probe.dump_graph(g.raw_cuda_graph())

def X(n):  return bx[:n*D].view(n,D)
def Y(n):  return by[:n*D].view(n,D)
def O(n):  return bo[:n*D].view(n,D)
def I(n):  return bi[:n]%512
def IM(n): return (bi[:n*D]%512).view(n,D)
def B(n):  return bb[:n*D].view(n,D)
W = torch.randn(512,D,device=dev); G=torch.randn(D,device=dev); Bt=torch.randn(D,device=dev)

OPS = {
 "add (elementwise)":       lambda n: (lambda x=X(n),y=Y(n),o=O(n): torch.add(x,y,out=o)),
 "gelu":                    lambda n: (lambda x=X(n): F.gelu(x)),
 "silu*mul":                lambda n: (lambda x=X(n),y=Y(n): F.silu(x)*y),
 "where":                   lambda n: (lambda c=B(n),x=X(n),y=Y(n): torch.where(c,x,y)),
 "masked_fill_":            lambda n: (lambda x=O(n),c=B(n): x.masked_fill_(c,0.5)),
 "clamp":                   lambda n: (lambda x=X(n): x.clamp(-1,1)),
 "embedding":               lambda n: (lambda i=I(n): F.embedding(i,W)),
 "index_select":            lambda n: (lambda i=I(n): torch.index_select(W,0,i)),
 "index_put_ acc=F":        lambda n: (lambda d=bo[:512*D].view(512,D),i=I(n),v=X(n): d.index_put_((i,),v)),
 "index_put_ acc=T":        lambda n: (lambda d=bo[:512*D].view(512,D),i=I(n),v=X(n): d.index_put_((i,),v,accumulate=True)),
 "index_add_":              lambda n: (lambda d=bo[:512*D].view(512,D),i=I(n),v=X(n): d.index_add_(0,i,v)),
 "scatter_add_":            lambda n: (lambda d=bo[:512*D].view(512,D),i=IM(n),v=X(n): d.scatter_add_(0,i,v)),
 "gather":                  lambda n: (lambda i=IM(n): torch.gather(W.expand(512,D).contiguous(),0,i) if False else torch.gather(W,0,i)),
 "sort (len axis)":         lambda n: (lambda x=bx[:n].contiguous(): torch.sort(x)),
 "sort (rows, dim=-1)":     lambda n: (lambda x=X(n): torch.sort(x,dim=-1)),
 "topk k=8 (len axis)":     lambda n: (lambda x=bx[:n].contiguous(): torch.topk(x,8)),
 "argmax (rows)":           lambda n: (lambda x=X(n): x.argmax(dim=-1)),
 "cumsum (len axis)":       lambda n: (lambda x=bx[:n].contiguous(): torch.cumsum(x,0)),
 "layer_norm (rows)":       lambda n: (lambda x=X(n): F.layer_norm(x,(D,),G,Bt,1e-5)),
 "rms_norm (rows)":         lambda n: (lambda x=X(n): F.rms_norm(x,(D,),G,1e-5)),
 "softmax (rows,dim=-1)":   lambda n: (lambda x=X(n): torch.softmax(x,-1)),
 "log_softmax (rows)":      lambda n: (lambda x=X(n): torch.log_softmax(x,-1)),
 "softmax over LEN axis":   lambda n: (lambda x=bx[:n*4].view(4,n): torch.softmax(x,-1)),
 "sum(dim=0)":              lambda n: (lambda x=X(n): x.sum(0)),
 "mean(dim=-1)":            lambda n: (lambda x=X(n): x.mean(-1)),
 "var(dim=-1)":             lambda n: (lambda x=X(n): x.var(-1)),
 "norm()":                  lambda n: (lambda x=X(n): x.norm()),
 "repeat_interleave x3":    lambda n: (lambda x=X(n): torch.repeat_interleave(x,3,dim=0)),
 "cat([x,x])":              lambda n: (lambda x=X(n): torch.cat([x,x],0)),
 "pad (constant)":          lambda n: (lambda x=X(n): F.pad(x,(0,0,0,4))),
 "flip(0)":                 lambda n: (lambda x=X(n): torch.flip(x,[0])),
 "roll(1,0)":               lambda n: (lambda x=X(n): torch.roll(x,1,0)),
 "tril":                    lambda n: (lambda x=X(n): torch.tril(x)),
 "contiguous(transpose)":   lambda n: (lambda x=X(n): x.t().contiguous()),
 "nll_loss":                lambda n: (lambda x=torch.log_softmax(X(n),-1),t=I(n)%D: F.nll_loss(x,t)),
 "cross_entropy":           lambda n: (lambda x=X(n),t=I(n)%D: F.cross_entropy(x,t)),
 "dropout p=.1":            lambda n: (lambda x=X(n): F.dropout(x,0.1,True)),
 "max_pool1d":              lambda n: (lambda x=X(n).t().unsqueeze(0): F.max_pool1d(x,2)),
 "avg_pool1d":              lambda n: (lambda x=X(n).t().unsqueeze(0): F.avg_pool1d(x,2)),
 "bincount-free: eq+sum":   lambda n: (lambda i=I(n): (i==3).sum()),
 "one_hot":                 lambda n: (lambda i=I(n): F.one_hot(i,512)),
 "matmul small (cuBLAS)":   lambda n: (lambda x=X(n): x@W.t()),
}

def classify(sigs):
    types = set(s[0] for s in sigs); funcs = set(s[1] for s in sigs)
    launch = set(s[2] for s in sigs)
    if len(types)>1: return "STRUCT-SWITCH"
    if len(funcs)>1: return "KERNEL-SWITCH"
    if len(launch)>1: return "LAUNCH-ONLY"
    return "STABLE"

rows=[]
for tag, mk in OPS.items():
    sigs=[]; errs=0; nodesets={}
    for n in SHAPES:
        try:
            g,nodes=cap(mk(n))
        except Exception as e:
            errs+=1; continue
        ks=kernels(nodes)
        sigs.append((tuple(x["type"] for x in nodes), tuple(k["func"] for k in ks),
                     tuple((k["grid"],k["block"],k["smem"]) for k in ks)))
        nodesets[n]=nodes
        del g
    if not sigs:
        rows.append((tag,"ALL-ERROR",0,"")); continue
    cls = classify(sigs)
    # naked-scalar analysis: compare first two same-kernel shapes
    detail=""
    keys=sorted(nodesets)
    if cls in ("LAUNCH-ONLY","STABLE") and len(keys)>=2:
        a=kernels(nodesets[keys[0]]); b=kernels(nodesets[keys[-1]])
        naked=0; blob=0
        for ka,kb in zip(a,b):
            for pa,pb in zip(ka["params"],kb["params"]):
                if pa["bytes"]!=pb["bytes"]:
                    if pa["size"]<=8: naked+=1
                    else: blob+=1
        detail=f"changed params: {naked} scalar(<=8B), {blob} struct(>8B)"
    elif cls!="STABLE":
        ncounts=sorted(set(len(s[0]) for s in sigs))
        nf=len(set(s[1] for s in sigs))
        detail=f"node counts {ncounts}, {nf} distinct kernel sets over {len(sigs)} shapes"
    rows.append((tag,cls,errs,detail))

print(f"{'op':28s} {'class':14s} {'detail'}")
print("-"*110)
order={"STABLE":0,"LAUNCH-ONLY":1,"KERNEL-SWITCH":2,"STRUCT-SWITCH":3,"ALL-ERROR":4}
for tag,cls,errs,detail in sorted(rows,key=lambda r:(order.get(r[1],9),r[0])):
    print(f"{tag:28s} {cls:14s} {detail}{'  (errors:%d)'%errs if errs else ''}")
from collections import Counter
c=Counter(r[1] for r in rows)
print("\nTOTALS:", dict(c), " n_ops =", len(rows))
patch = c.get("STABLE",0)+c.get("LAUNCH-ONLY",0)
print(f"patchable-by-params (same node set + same kernels): {patch}/{len(rows)} = {100*patch/len(rows):.0f}%")
