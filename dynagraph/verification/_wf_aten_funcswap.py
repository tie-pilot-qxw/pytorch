"""Does exec-graph FUNCTION swapping rescue the ATen 'kernel switches with shape'
cases?  Capture at shape A, install shape B's func+params+launch config, replay."""
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

def trial(tag, mkA, mkB, ref_fn, got_fn):
    print("="*90); print("##", tag)
    gA,nA=cap(mkA()); gB,nB=cap(mkB())
    tA=[x["type"] for x in nA]; tB=[x["type"] for x in nB]
    if tA!=tB:
        print(f"   node structure differs {tA} vs {tB} -> out of scope for func swap"); return
    kA,kB=kernels(nA),kernels(nB)
    for a,b in zip(kA,kB):
        print(f"   A: {demangle(a['name'])[:95]}")
        print(f"   B: {demangle(b['name'])[:95]}")
        print(f"      func {a['func']:#x} -> {b['func']:#x}  nparams {len(a['params'])} -> {len(b['params'])}"
              f"  grid {a['grid']}->{b['grid']} block {a['block']}->{b['block']} smem {a['smem']}->{b['smem']}")
    gA.instantiate(); exe=gA.raw_cuda_graph_exec()
    for a,b in zip(nA,nB):
        r=probe.copy_node_params(exe, a["node_handle"], b["node_handle"], True)
        if r!="ok": print("   SetParams ->", r); return
    print("   SetParams(force func swap) -> ok")
    ref=ref_fn().clone(); bo.zero_(); torch.cuda.synchronize()
    gA.replay(); torch.cuda.synchronize()
    got=got_fn()
    ok=torch.allclose(got,ref,atol=1e-5,rtol=1e-4)
    print(f"   REPLAY: {'CORRECT' if ok else 'WRONG'}  maxdiff={(got-ref).abs().max().item():.3e}")

def sm(k):
    x=bx[:64*k].view(64,k); o=bo[:64*k].view(64,k)
    return lambda: torch.softmax(x,-1,out=o)
trial("softmax dim 256 -> 512   (softmax_warp_forward LOG2 8 -> 9)",
      lambda: sm(256), lambda: sm(512),
      lambda: torch.softmax(bx[:64*512].view(64,512),-1),
      lambda: bo[:64*512].view(64,512))
trial("softmax dim 256 -> 4096  (softmax_warp_forward -> cunn_SoftMaxForwardReg, 8 params -> 3)",
      lambda: sm(256), lambda: sm(4096),
      lambda: torch.softmax(bx[:64*4096].view(64,4096),-1),
      lambda: bo[:64*4096].view(64,4096))
trial("softmax dim 4096 -> 256  (reverse direction)",
      lambda: sm(4096), lambda: sm(256),
      lambda: torch.softmax(bx[:64*256].view(64,256),-1),
      lambda: bo[:64*256].view(64,256))

def isel(k):
    i=bi[:k]%512; o=bo[:k*D].view(k,D)
    return lambda: torch.index_select(W,0,i,out=o)
trial("index_select 16 -> 64  (indexSelectSmallIndex -> vectorized_gather_kernel)",
      lambda: isel(16), lambda: isel(64),
      lambda: torch.index_select(W,0,bi[:64]%512),
      lambda: bo[:64*D].view(64,D))

bv=torch.zeros(NB,device=dev); bidx=torch.zeros(NB,device=dev,dtype=torch.long)
def srt(k):
    x=bx[:k]; v=bv[:k]; ii=bidx[:k]
    return lambda: torch.sort(x,out=(v,ii))
def ref_srt(k): return torch.sort(bx[:k])[0]
def got_srt(k): return bv[:k]
for a,b in [(128,1024),(1024,2048),(2048,4096)]:
    trial(f"sort len {a} -> {b} (out=, persistent buffers)",
          (lambda a=a: srt(a)), (lambda b=b: srt(b)),
          (lambda b=b: ref_srt(b)), (lambda b=b: got_srt(b)))

def tk(k):
    x=bx[:k]; v=bv[:8]; ii=bidx[:8]
    return lambda: torch.topk(x,8,out=(v,ii))
trial("topk len 16384 -> 32768 (sbtopk -> mbtopk, 2 nodes -> 22)",
      lambda: tk(16384), lambda: tk(32768),
      lambda: torch.topk(bx[:32768],8)[0], lambda: bv[:8])
