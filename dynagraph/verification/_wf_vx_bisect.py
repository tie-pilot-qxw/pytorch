"""Independent bisect of the STRUCT-SWITCH boundaries, keyed on mangled NAMES
(not CUfunction pointers), to re-test claims 13 and 20."""
import sys, os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
NB=2*1024*1024
bx=torch.randn(NB,device=dev)
def cap(fn,w=2):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); n=probe.dump_graph(g.raw_cuda_graph()); del g; return n
def sig(n):   # NAME-based signature
    return (tuple(x['type'] for x in n), tuple(k['name'] for k in kernels(n)))
def show(n):
    nd=cap_cache[n]
    return f"{len(nd)} nodes {[x['type'] for x in nd]}"
cap_cache={}
def S(mk,n):
    if n not in cap_cache: cap_cache[n]=cap(mk(n))
    return sig(cap_cache[n])
def bisect(tag,mk,lo,hi):
    cap_cache.clear()
    a,b=S(mk,lo),S(mk,hi)
    if a==b: print(f"  {tag}: NO switch between {lo} and {hi}"); return
    while hi-lo>1:
        m=(lo+hi)//2
        if S(mk,m)==a: lo=m
        else: hi=m
    print(f"  {tag}: switches between n={lo} and n={hi}")
    print(f"      n={lo}: {len(cap_cache[lo])} nodes {[x['type'] for x in cap_cache[lo]]}")
    for k in kernels(cap_cache[lo]): print(f"          {demangle(k['name'])[:95]}")
    print(f"      n={hi}: {len(cap_cache[hi])} nodes {[x['type'] for x in cap_cache[hi]]}")
    for k in kernels(cap_cache[hi]): print(f"          {demangle(k['name'])[:95]}")

def m_topk(n): 
    x=bx[:n].contiguous(); return lambda: torch.topk(x,8,dim=-1)
def m_sort(n):
    x=bx[:n].contiguous(); return lambda: torch.sort(x,dim=-1)
print("## name-keyed bisect")
bisect("topk k=8 single->multi block", m_topk, 1024, 65536)
bisect("sort len (cub onesweep)", m_sort, 2048, 8192)
