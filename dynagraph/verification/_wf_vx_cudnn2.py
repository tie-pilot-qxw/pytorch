"""Numeric transplant test on the cuDNN line (not covered by the 22 claims)."""
import sys, os, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
def cap(fn,w=3):
    box={}
    def wr(): box['o']=fn()
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): wr()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): wr()
    torch.cuda.synchronize(); return g, probe.dump_graph(g.raw_cuda_graph()), box['o']

def run(tag, fA, fB, refB, tol):
    gA,nA,oA=cap(fA); gB,nB,oB=cap(fB)
    tA=[x['type'] for x in nA]; tB=[x['type'] for x in nB]
    same=tuple(k['name'] for k in kernels(nA))==tuple(k['name'] for k in kernels(nB))
    print(f"  {tag}: nodesA={tA} nodesB={tB} same_kernel_names={same}")
    if tA!=tB: print("    STRUCT differs -> skip"); return
    gA.instantiate(); exe=gA.raw_cuda_graph_exec()
    rs=set(probe.copy_node_params(exe,a['node_handle'],b['node_handle'],True) for a,b in zip(nA,nB))
    ref=refB().clone(); oB.zero_(); torch.cuda.synchronize()
    gA.replay(); torch.cuda.synchronize()
    md=(oB.float()-ref.float()).abs().max().item(); sc=max(ref.float().abs().max().item(),1.0)
    print(f"    copy={rs} -> {'CORRECT' if md/sc<tol else 'WRONG'} maxdiff={md:.3e} rel={md/sc:.2e}")

print("### cuDNN conv2d, batch 8 -> 16")
w=torch.randn(32,16,3,3,device=dev)
x8=torch.randn(8,16,32,32,device=dev); x16=torch.randn(16,16,32,32,device=dev)
run("conv2d N=8->16", lambda: F.conv2d(x8,w,padding=1), lambda: F.conv2d(x16,w,padding=1),
    lambda: F.conv2d(x16,w,padding=1), 1e-4)
print("### cuDNN conv2d, batch 16 -> 32 (crosses the tilesize32x32x8 -> 128x32x8 switch)")
x32=torch.randn(32,16,32,32,device=dev)
run("conv2d N=16->32", lambda: F.conv2d(x16,w,padding=1), lambda: F.conv2d(x32,w,padding=1),
    lambda: F.conv2d(x32,w,padding=1), 1e-4)

print("### cuDNN SDPA, seq 128 -> 512")
B_=torch.nn.attention.SDPBackend.CUDNN_ATTENTION
def qkv(S): 
    torch.manual_seed(S)
    return [torch.randn(2,4,S,64,device=dev,dtype=torch.float16) for _ in range(3)]
q1,k1,v1=qkv(128); q2,k2,v2=qkv(512)
def sd(q,k,v):
    with torch.nn.attention.sdpa_kernel(B_):
        return F.scaled_dot_product_attention(q,k,v)
run("sdpa S=128->512", lambda: sd(q1,k1,v1), lambda: sd(q2,k2,v2), lambda: sd(q2,k2,v2), 2e-3)
