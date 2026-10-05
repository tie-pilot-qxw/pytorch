import sys, os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
NB=2*1024*1024
base_x=torch.randn(NB,device=dev); base_o=torch.zeros(NB,device=dev)
base_idx=torch.randint(0,1024,(NB,),device=dev,dtype=torch.long)
D=64
def cap(fn,w=3):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g, probe.dump_graph(g.raw_cuda_graph())
def flat(nd):
    t=max(p["offset"]+p["size"] for p in nd["params"]); b=bytearray(t)
    for p in nd["params"]: b[p["offset"]:p["offset"]+p["size"]]=p["bytes"]
    return bytes(b)

case = sys.argv[1]

if case == "sum":
    o=base_o[:1]
    def A(): x=base_x[:512];  return lambda: torch.sum(x,dim=0,out=o)
    def B(): x=base_x[:2048]; return lambda: torch.sum(x,dim=0,out=o)
    gA,nA=cap(A()); gB,nB=cap(B())
    a,b=kernels(nA)[0],kernels(nB)[0]
    print("funcA",hex(a["func"]),"funcB",hex(b["func"]),"same",a["func"]==b["func"])
    print("gridA",a["grid"],"blockA",a["block"],"smemA",a["smem"])
    print("gridB",b["grid"],"blockB",b["block"],"smemB",b["smem"])
    fa,fb=flat(a),flat(b)
    runs=[];i=0
    while i<len(fa):
        if fa[i]!=fb[i]:
            j=i
            while j<len(fa) and fa[j]!=fb[j]: j+=1
            runs.append((i,j)); i=j
        else: i+=1
    print("struct byte-runs that differ:")
    for s0,s1 in runs:
        print(f"   @{s0}..{s1}  A={int.from_bytes(fa[s0:min(s0+8,s1)],'little')}  B={int.from_bytes(fb[s0:min(s0+8,s1)],'little')}"
              f"  Ahex={fa[s0:s1].hex()}  Bhex={fb[s0:s1].hex()}")
    ref=torch.sum(base_x[:2048]).item()
    ref512=torch.sum(base_x[:512]).item()
    gA.instantiate()
    probe.set_exec_kernel_params(gA.raw_cuda_graph_exec(), a["node_handle"], fb, *b["grid"], *b["block"], b["smem"])
    base_o.zero_(); torch.cuda.synchronize(); gA.replay(); torch.cuda.synchronize()
    print(f"got={base_o[0].item():.6f}  ref(2048)={ref:.6f}  ref(512)={ref512:.6f}")
    # control: replay graph B itself
    gB.instantiate(); base_o.zero_(); gB.replay(); torch.cuda.synchronize()
    print(f"graphB own replay = {base_o[0].item():.6f}")

if case == "iput":
    # unique indices -> deterministic
    perm = torch.randperm(1024, device=dev)
    d=base_o[:1024*D].view(1024,D)
    for NA,NB_ in [(512,1024)]:
        def A(): 
            i=perm[:NA].contiguous(); v=base_x[:NA*D].view(NA,D)
            return lambda: d.index_put_((i,), v, accumulate=False)
        def B():
            i=perm[:NB_].contiguous(); v=base_x[:NB_*D].view(NB_,D)
            return lambda: d.index_put_((i,), v, accumulate=False)
        iA=perm[:NA].contiguous(); iB=perm[:NB_].contiguous()
        def A(): 
            v=base_x[:NA*D].view(NA,D); return lambda: d.index_put_((iA,), v, accumulate=False)
        def B():
            v=base_x[:NB_*D].view(NB_,D); return lambda: d.index_put_((iB,), v, accumulate=False)
        gA,nA=cap(A()); gB,nB=cap(B())
        a,b=kernels(nA)[0],kernels(nB)[0]
        print("same func:",a["func"]==b["func"],"gridA",a["grid"],"gridB",b["grid"])
        ref=torch.zeros(1024,D,device=dev); ref.index_put_((iB,), base_x[:NB_*D].view(NB_,D))
        ref=ref.clone()
        gA.instantiate()
        probe.set_exec_kernel_params(gA.raw_cuda_graph_exec(), a["node_handle"], flat(b), *b["grid"], *b["block"], b["smem"])
        base_o.zero_(); torch.cuda.synchronize(); gA.replay(); torch.cuda.synchronize()
        got=base_o[:1024*D].view(1024,D)
        print("unique-index index_put_ patched replay correct:",torch.allclose(got,ref),
              "maxdiff",(got-ref).abs().max().item())
        # control: graphB replay
        gB.instantiate(); base_o.zero_(); gB.replay(); torch.cuda.synchronize()
        print("graphB own replay correct:",torch.allclose(base_o[:1024*D].view(1024,D),ref))
