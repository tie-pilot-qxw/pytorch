"""Can the grid of an opaque kernel really be inferred from a few fake runs?
For each op, capture over a series of n, read each kernel node's gridDim, and see how grid relates to n."""
import math, torch, torch.nn.functional as F
import cuda.bindings.driver as D
DEV="cuda"; DT=torch.float16
torch.manual_seed(0)

def chk(r):
    if isinstance(r,(list,tuple)):
        if r[0] != D.CUresult.CUDA_SUCCESS: raise RuntimeError(str(r[0]))
        rest = r[1:]
        return rest[0] if len(rest)==1 else rest
    return r

def capture_grids(fn):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s): fn()
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph(keep_graph=True)
    except TypeError:
        g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s): fn()
    torch.cuda.synchronize()
    cg = D.CUgraph(g.raw_cuda_graph())
    _, cnt = chk(D.cuGraphGetNodes(cg, 0))          # (nodes, numNodes)
    nodes, _ = chk(D.cuGraphGetNodes(cg, cnt))
    out=[]
    for nd in nodes:
        t = chk(D.cuGraphNodeGetType(nd))
        if t != D.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL: 
            out.append(("(non-kernel)", None)); continue
        p = chk(D.cuGraphKernelNodeGetParams(nd))
        try: nm = chk(D.cuFuncGetName(p.func)).decode()
        except Exception: nm = "?"
        out.append((nm.split("<")[0].split("(")[0][:40], (p.gridDimX, p.gridDimY, p.gridDimZ)))
    return out

NS = [4096, 8192, 16384, 65536, 262144, 1048576]
CASES = [
  ("scatter_add",  lambda n: (lambda x=torch.randn(n,device=DEV), i=torch.randint(0,max(n//2,1),(n,),device=DEV), o=torch.zeros(max(n//2,1),device=DEV): (lambda: o.scatter_add(0,i,x)))()),
  ("index_select", lambda n: (lambda x=torch.randn(n,device=DEV), i=torch.randint(0,n,(n,),device=DEV): (lambda: torch.index_select(x,0,i)))()),
  ("layer_norm",   lambda n: (lambda x=torch.randn(n//128,128,device=DEV,dtype=DT), w=torch.randn(128,device=DEV,dtype=DT), b=torch.randn(128,device=DEV,dtype=DT): (lambda: F.layer_norm(x,(128,),w,b)))()),
  ("softmax",      lambda n: (lambda x=torch.randn(n//128,128,device=DEV,dtype=DT): (lambda: F.softmax(x,-1)))()),
  ("embedding_bag",lambda n: (lambda w=torch.randn(4096,128,device=DEV), i=torch.randint(0,4096,(n,),device=DEV), o=torch.arange(0,n,max(n//32,1),device=DEV,dtype=torch.long): (lambda: F.embedding_bag(i,w,o)))()),
  ("conv2d(cuDNN)",lambda n: (lambda x=torch.randn(max(n//(3*64*64),1),3,64,64,device=DEV,dtype=DT), w=torch.randn(32,3,3,3,device=DEV,dtype=DT): (lambda: F.conv2d(x,w,padding=1)))()),
  ("cumsum",       lambda n: (lambda x=torch.randn(n,device=DEV): (lambda: torch.cumsum(x,0)))()),
  ("GEMM(cuBLAS)", lambda n: (lambda a=torch.randn(max(n//512,1),512,device=DEV,dtype=DT), b=torch.randn(512,512,device=DEV,dtype=DT): (lambda: a@b))()),
]

def classify(ns, grids):
    """What function of n is the grid?"""
    gx = [g[0]*g[1]*g[2] for g in grids]
    if len(set(gx))==1: return f"constant {gx[0]} (independent of n)", True
    # try cdiv(n, C)
    for i in range(len(ns)):
        if gx[i]==0: continue
        C = math.ceil(ns[i]/gx[i])
        for C2 in {C, C-1, C+1, 1<<max(0,(C-1).bit_length())}:
            if C2>0 and all(g == -(-nn//C2) for nn,g in zip(ns,gx)):
                return f"cdiv(n, {C2}) -- 2 points fix it, a 3rd verifies", True
    # try linear
    if len(set(gx))==len(gx):
        d = [(gx[i+1]-gx[i])/(ns[i+1]-ns[i]) for i in range(len(ns)-1)]
        if max(d)-min(d) < 1e-9: return f"linear n*{d[0]:.3g}", True
    return "**not in the small hypothesis class**", False

print(f"{'op':<16} {'how grid changes with n':<52} {'recover?':>8}")
print("-"*84)
rec = tot = 0
for name, maker in CASES:
    per_n = []
    try:
        for n in NS: per_n.append(capture_grids(maker(n)))
    except Exception as e:
        print(f"{name:<16} capture failed: {type(e).__name__} {str(e)[:40]}"); continue
    if len(set(len(x) for x in per_n)) != 1:
        print(f"{name:<16} {'topology changes with n (kernel count differs), not applicable':<52} {'N/A':>8}"); continue
    nk = len(per_n[0])
    for ki in range(nk):
        kname = per_n[0][ki][0]
        grids = [per_n[j][ki][1] for j in range(len(NS))]
        if any(g is None for g in grids):
            continue
        desc, ok = classify(NS, grids)
        tot += 1; rec += ok
        gs = " ".join(f"{g[0]}x{g[1]}" for g in grids)
        label = f"{name}.{kname[:12]}" if nk>1 else name
        print(f"{label:<16} {desc:<52} {'yes' if ok else 'no':>8}")
        print(f"{'':<16} measured grid: {gs}")
print(f"\n{rec}/{tot} kernels have a grid in the small hypothesis class \"constant / cdiv(n,C) / linear\"; 2-3 fake runs are enough to pin it down.")
