"""The 22 claims say nothing about cuDNN, which the user explicitly asked about.
Same three gates: (A) node attribution, (B) param readability, (C) kernel/node
stability along a variable-length axis. Small tensors only."""
import os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
def cap(fn,w=3):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g, probe.dump_graph(g.raw_cuda_graph())

print("### conv2d fp32, vary BATCH (the variable-length axis) ; C=16 H=W=32 K=32")
w=torch.randn(32,16,3,3,device=dev)
xs={n: torch.randn(n,16,32,32,device=dev) for n in [1,2,4,8,16,32]}
sigs={}
for n in sorted(xs):
    g,nodes=cap(lambda n=n: F.conv2d(xs[n],w,padding=1))
    ks=kernels(nodes)
    sigs[n]=(tuple(x['type'] for x in nodes), tuple(k['name'] for k in ks))
    rd=[sum(1 for p in k['params'] if p['bytes'] is not None) for k in ks]
    np_=[len(k['params']) for k in ks]
    print(f"  N={n:3d} {len(nodes)} node(s) {sigs[n][0]} nparams={np_} readable={rd} "
          f"extra={[k['has_extra'] for k in ks]}")
    for k in ks: print(f"       {demangle(k['name'])[:96]}  grid={k['grid']} smem={k['smem']}")
    del g
print("  distinct (type,name) signatures:", len(set(sigs.values())), "over", len(sigs), "shapes")

print()
print("### SDPA, vary SEQ LEN ; B=2 H=4 Dh=64, fp16")
for backend,name in [(torch.nn.attention.SDPBackend.CUDNN_ATTENTION,"cudnn"),
                     (torch.nn.attention.SDPBackend.FLASH_ATTENTION,"flash")]:
    print(f"  -- backend {name}")
    sg={}
    for S in [64,128,129,256,512,1000]:
        q=torch.randn(2,4,S,64,device=dev,dtype=torch.float16)
        k_=torch.randn(2,4,S,64,device=dev,dtype=torch.float16)
        v=torch.randn(2,4,S,64,device=dev,dtype=torch.float16)
        try:
            with torch.nn.attention.sdpa_kernel(backend):
                g,nodes=cap(lambda: F.scaled_dot_product_attention(q,k_,v))
        except Exception as e:
            print(f"     S={S:5d} ERROR {type(e).__name__} {str(e)[:70]}"); continue
        ks=kernels(nodes)
        sg[S]=(tuple(x['type'] for x in nodes), tuple(kk['name'] for kk in ks))
        rd=[sum(1 for p in kk['params'] if p['bytes'] is not None) for kk in ks]
        print(f"     S={S:5d} {len(nodes)} node(s) {sg[S][0]} nparams={[len(kk['params']) for kk in ks]} readable={rd}")
        for kk in ks: print(f"          {demangle(kk['name'])[:92]}")
        del g
    print("     distinct signatures:", len(set(sg.values())), "over", len(sg), "shapes")
