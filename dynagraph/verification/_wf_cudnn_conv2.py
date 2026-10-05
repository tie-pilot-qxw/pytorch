"""conv2d: sweep batch sizes, byte-level param diff, channels_last variant."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from _wf_cudnn_lib import describe, graph_nodes

torch.manual_seed(0)
DEV="cuda"; DT=torch.float16
C_IN,C_OUT,HW,K = 64,64,56,3
W_NCHW = torch.randn(C_OUT,C_IN,K,K,device=DEV,dtype=DT)
XBIG = torch.randn(32,C_IN,HW,HW,device=DEV,dtype=DT)
W_CL  = W_NCHW.to(memory_format=torch.channels_last)
XBIG_CL = XBIG.to(memory_format=torch.channels_last)

def capture(batch, cl, bench=False):
    torch.backends.cudnn.benchmark = bench
    x = (XBIG_CL if cl else XBIG)[:batch]
    w = W_CL if cl else W_NCHW
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(4): y = F.conv2d(x,w,padding=1)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        y = F.conv2d(x,w,padding=1)
    torch.cuda.synchronize()
    return g, describe(graph_nodes(g.raw_cuda_graph()))

def ksig(info):
    return tuple((d.get("name","?") if d["type"]=="KERNEL" else d["type"]) for d in info)

for cl in (False, True):
    print(f"\n{'='*90}\n=== memory_format={'channels_last' if cl else 'contiguous(NCHW)'} ===")
    keep = {}
    for b in (1,2,3,4,8,13,16,32):
        try:
            g, info = capture(b, cl)
            keep[b] = info
            sig = ksig(info)
            print(f"\n b={b:3d}: {len(info)} nodes")
            for d in info:
                if d["type"]!="KERNEL": print(f"    {d['type']}"); continue
                pi = d.get("param_info",[])
                tot = (pi[-1][1]+pi[-1][2]) if pi else 0
                print(f"    grid={str(d['grid']):16s} blk={str(d['block']):14s} smem={d['smem']:6d} "
                      f"np={len(pi)} pbytes={tot:5d}  {d['name'][:96]}")
        except Exception as e:
            print(f" b={b}: FAILED {type(e).__name__}: {e}")
    # distinct kernel sets
    sigs = {}
    for b,info in keep.items():
        sigs.setdefault(ksig(info), []).append(b)
    print(f"\n  -> {len(sigs)} distinct kernel-sequences over batches {sorted(keep)}:")
    for s,bs in sigs.items():
        print(f"     batches {bs}: " + " | ".join(n[:70] for n in s))

    # byte-level diff for batch pairs that share the same kernel sequence
    print("\n  --- byte-level param diff within a same-kernel group ---")
    for s,bs in sigs.items():
        if len(bs)<2: continue
        a,bb = keep[bs[0]], keep[bs[1]]
        print(f"   pair b={bs[0]} vs b={bs[1]}")
        for na,nb in zip(a,bb):
            if na["type"]!="KERNEL": continue
            pa = na.get("param_bytes") or []; pb = nb.get("param_bytes") or []
            print(f"     node{na['i']} grid {na['grid']} -> {nb['grid']}  {na['name'][:70]}")
            for (i,off,sz,ba),(_,_,_,bbb) in zip(pa,pb):
                if ba is None or bbb is None or ba==bbb:
                    continue
                # per-byte runs
                runs=[]; j=0
                while j<sz:
                    if ba[j]!=bbb[j]:
                        k=j
                        while k<sz and ba[k]!=bbb[k]: k+=1
                        # widen to 4-byte aligned word
                        lo=(j//4)*4; hi=((k+3)//4)*4
                        runs.append((lo,hi)); j=k
                    else: j+=1
                # merge
                merged=[]
                for lo,hi in runs:
                    if merged and lo<=merged[-1][1]: merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
                    else: merged.append((lo,hi))
                desc=[]
                for lo,hi in merged:
                    va=int.from_bytes(ba[lo:hi],"little"); vb=int.from_bytes(bbb[lo:hi],"little")
                    tag = "PTR" if va>0x1000000000 else "int"
                    desc.append(f"[{lo}:{hi}] {tag} {va} -> {vb}")
                print(f"       param{i} (off={off},sz={sz}) differs at {len(merged)} run(s): " + "; ".join(desc[:12]))
