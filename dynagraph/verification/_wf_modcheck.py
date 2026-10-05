import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
bx=torch.randn(1<<21,device=dev); bo=torch.zeros(1<<21,device=dev)
bi=torch.randint(0,512,(1<<20,),device=dev,dtype=torch.long); D=128
W=torch.randn(512,D,device=dev)
def cap(fn,w=2):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g, probe.dump_graph(g.raw_cuda_graph())
def one(tag,fn):
    g,n=cap(fn); k=kernels(n)[0]
    print(f"  {tag:28s} func={k['func']:#x} module={k['module']:#x}  {demangle(k['name'])[:70]}")
    return k
print("## CUmodule identity of the kernels we swapped between")
one("softmax dim=256", lambda: torch.softmax(bx[:64*256].view(64,256),-1,out=bo[:64*256].view(64,256)))
one("softmax dim=512", lambda: torch.softmax(bx[:64*512].view(64,512),-1,out=bo[:64*512].view(64,512)))
one("softmax dim=4096",lambda: torch.softmax(bx[:64*4096].view(64,4096),-1,out=bo[:64*4096].view(64,4096)))
i16=(bi[:16]%512).contiguous(); i64=(bi[:64]%512).contiguous()
one("index_select n=16", lambda: torch.index_select(W,0,i16,out=bo[:16*D].view(16,D)))
one("index_select n=64", lambda: torch.index_select(W,0,i64,out=bo[:64*D].view(64,D)))
one("sort len=128", lambda: torch.sort(bx[:128],out=(bo[:128],bi[:128])))
one("sort len=1024", lambda: torch.sort(bx[:1024],out=(bo[:1024],bi[:1024])))
one("add", lambda: torch.add(bx[:1024],bx[:1024],out=bo[:1024]))
