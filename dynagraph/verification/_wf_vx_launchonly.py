"""The census MEASURES 'LAUNCH-ONLY' (structural). The verdict INFERS
'therefore params-transplant adapts it'. The original agent only numerically
verified 8 ops. Here: numerically verify a batch of LAUNCH-ONLY ops that were
NOT in that list, by blind full-node param transplant 512 -> 1000."""
import os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
NB=4*1024*1024; D=128
bx=torch.randn(NB,device=dev); by=torch.randn(NB,device=dev)
bi=torch.randint(0,512,(NB,),device=dev,dtype=torch.long)
bb=(torch.rand(NB,device=dev)>0.5)
W=torch.randn(512,D,device=dev); G=torch.randn(D,device=dev); Bt=torch.randn(D,device=dev)
def X(n): return bx[:n*D].view(n,D)
def Y(n): return by[:n*D].view(n,D)
def I(n): return bi[:n]%512
def B_(n): return bb[:n*D].view(n,D)

def cap(fn,w=3):
    box={}
    def wrapped(): box['o']=fn()
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): wrapped()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): wrapped()
    torch.cuda.synchronize()
    return g, probe.dump_graph(g.raw_cuda_graph()), box['o']

OPS = {
 "where":                 lambda n: torch.where(B_(n),X(n),Y(n)),
 "roll(1,0)":             lambda n: torch.roll(X(n),1,0),
 "flip(0)":               lambda n: torch.flip(X(n),[0]),
 "tril":                  lambda n: torch.tril(X(n)),
 "contiguous(transpose)": lambda n: X(n).t().contiguous(),
 "repeat_interleave x3":  lambda n: torch.repeat_interleave(X(n),3,dim=0),
 "cat([x,x])":            lambda n: torch.cat([X(n),X(n)],0),
 "pad(constant)":         lambda n: F.pad(X(n),(0,0,0,4)),
 "one_hot":               lambda n: F.one_hot(I(n),512),
 "max_pool1d":            lambda n: F.max_pool1d(X(n).t().unsqueeze(0),2),
 "cross_entropy":         lambda n: F.cross_entropy(X(n),I(n)%D),
 "argmax(rows)":          lambda n: X(n).argmax(dim=-1),
 "embedding":             lambda n: F.embedding(I(n),W),
 "clamp":                 lambda n: X(n).clamp(-1,1),
 "silu*mul":              lambda n: F.silu(X(n))*Y(n),
 "layer_norm(rows)":      lambda n: F.layer_norm(X(n),(D,),G,Bt,1e-5),
 "var(dim=-1)":           lambda n: X(n).var(-1),
 "gelu":                  lambda n: F.gelu(X(n)),
}
A,Bn = 512, 1000
print(f"blind transplant {A} -> {Bn}, verified numerically against eager@{Bn}")
print(f"{'op':24s} {'nodesA/B':22s} {'copy':10s} result")
print("-"*100)
nok=nbad=0
for tag,f in OPS.items():
    try:
        gA,nA,oA=cap(lambda: f(A)); gB,nB,oB=cap(lambda: f(Bn))
    except Exception as e:
        print(f"{tag:24s} capture failed: {type(e).__name__} {str(e)[:60]}"); continue
    tA=[x['type'] for x in nA]; tB=[x['type'] for x in nB]
    namesA=tuple(k['name'] for k in kernels(nA)); namesB=tuple(k['name'] for k in kernels(nB))
    if tA!=tB:
        print(f"{tag:24s} {str(tA)[:20]:22s} {'-':10s} STRUCT differs {tB}"); continue
    gA.instantiate(); exe=gA.raw_cuda_graph_exec()
    rs=set(probe.copy_node_params(exe,a['node_handle'],b['node_handle'],True) for a,b in zip(nA,nB))
    ref=f(Bn).clone()
    oB.zero_() if oB.dtype.is_floating_point else oB.fill_(0)
    torch.cuda.synchronize()
    try:
        gA.replay(); torch.cuda.synchronize()
        got=oB
        if got.dtype!=ref.dtype: got=got.to(ref.dtype)
        md=(got.float()-ref.float()).abs().max().item()
        scale=max(ref.float().abs().max().item(),1.0)
        ok = md/scale < 1e-4
        nok+=ok; nbad+=(not ok)
        print(f"{tag:24s} {str(len(tA))+'/'+str(len(tB))+' '+('sameK' if namesA==namesB else 'KSWITCH'):22s} "
              f"{str(rs)[:9]:10s} {'CORRECT' if ok else 'WRONG'} maxdiff={md:.3e}")
    except Exception as e:
        nbad+=1
        print(f"{tag:24s} {'':22s} {'':10s} EXCEPTION {type(e).__name__} {str(e)[:70]}")
    del gA,gB
print(f"\n-> {nok} CORRECT, {nbad} NOT")
