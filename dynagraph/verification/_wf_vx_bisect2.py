import sys, os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
bx=torch.randn(2*1024*1024,device=dev)
def cap(fn,w=2):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); n=probe.dump_graph(g.raw_cuda_graph()); del g; return n
print("## exact node counts at the claimed boundaries (names, not pointers)")
for n in [19998,19999,20000,20001]:
    nd=cap(lambda n=n: torch.topk(bx[:n].contiguous(),8,dim=-1))
    fam = "mbtopk" if any("mbtopk" in k["name"] for k in kernels(nd)) else "sbtopk"
    print(f"  topk n={n}: {len(nd)} nodes, family={fam}")
for n in [4094,4095,4096,4097,4098]:
    nd=cap(lambda n=n: torch.sort(bx[:n].contiguous(),dim=-1))
    fam = "onesweep" if any("Onesweep" in k["name"] or "DeviceRadixSort" in k["name"] for k in kernels(nd)) else "radixSortKVInPlace"
    print(f"  sort n={n}: {len(nd)} nodes, family={fam}")
