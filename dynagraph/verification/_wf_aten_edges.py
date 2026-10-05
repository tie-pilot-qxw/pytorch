"""E6 bisect exact kernel-switch boundaries;  E7 32->64 bit index boundary;
   E8 alignment / dim-coalescing induced kernel switches."""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_aten_common import probe, kernels, demangle
torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
NB=4*1024*1024
bx=torch.randn(NB,device=dev); by=torch.randn(NB,device=dev); bo=torch.zeros(NB,device=dev)
bi=torch.randint(0,1024,(NB,),device=dev,dtype=torch.long)
D=64
def cap(fn,w=2):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(w): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g, probe.dump_graph(g.raw_cuda_graph())
def sig(nodes):
    ks=kernels(nodes)
    return (tuple(n["type"] for n in nodes), tuple(k["func"] for k in ks))
def nm(nodes):
    return [demangle(k["name"]).replace("at::native::","").replace("(anonymous namespace)::","")[:120] for k in kernels(nodes)]

mode = sys.argv[1]

if mode=="bisect":
    def probe_sig(mk, n):
        try: return sig(cap(mk(n))[1])
        except Exception as e: return ("ERR",str(e)[:60])
    def bisect(tag, mk, lo, hi):
        slo, shi = probe_sig(mk,lo), probe_sig(mk,hi)
        if slo==shi:
            print(f"  {tag}: no switch between {lo} and {hi}"); return
        while hi-lo>1:
            mid=(lo+hi)//2
            if probe_sig(mk,mid)==slo: lo=mid
            else: hi=mid
        print(f"  {tag}: SWITCHES between n={lo} and n={hi}")
        print(f"      n={lo}: {nm(cap(mk(lo))[1])}")
        print(f"      n={hi}: {nm(cap(mk(hi))[1])}")
    def m_isel(n):
        s=bx[:1024*D].view(1024,D); i=bi[:n]%1024; o=bo[:n*D].view(n,D)
        return lambda: torch.index_select(s,0,i,out=o)
    def m_sortlen(n):
        x=bx[:n].contiguous(); return lambda: torch.sort(x,dim=-1)
    def m_topklen(n):
        x=bx[:n].contiguous(); return lambda: torch.topk(x,8,dim=-1)
    def m_softmax(n):
        r=64; x=bx[:r*n].view(r,n); o=bo[:r*n].view(r,n); return lambda: torch.softmax(x,dim=-1,out=o)
    print("## exact kernel-switch boundaries (bisected)")
    bisect("index_select nidx", m_isel, 2, 1024)
    bisect("sort length (cub onesweep)", m_sortlen, 2048, 8192)
    bisect("sort length (radix block size 32->64)", m_sortlen, 64, 2048)
    bisect("topk length (single->multi block)", m_topklen, 1024, 65536)
    for a,b in [(8,16),(16,32),(32,64),(64,128),(128,256),(256,512),(512,1024),
                (1024,2048),(2048,4096),(4096,8192),(8192,16384)]:
        bisect(f"softmax dim {a}->{b}", m_softmax, a, b)

if mode=="bigindex":
    free,_=torch.cuda.mem_get_info(); print(f"free={free/2**30:.2f} GiB")
    need = 2**31 + (1<<20)
    if free < need + (1<<30):
        print("NOT ENOUGH FREE MEMORY on this shared card, skipping"); sys.exit(0)
    buf = torch.empty(need, dtype=torch.uint8, device=dev)
    print("alloc ok, numel", buf.numel())
    for n,label in [(2**31-1,"2^31-1 (fits int32)"), (2**31,"2^31 (overflows int32)")]:
        v = buf[:n]
        try:
            g,nodes = cap(lambda v=v: v.add_(1), w=1)
        except Exception as e:
            print(f"  {label}: capture error {type(e).__name__} {str(e)[:120]}"); continue
        ks=kernels(nodes)
        print(f"  add_ numel={n} {label}: {len(nodes)} nodes, {len(ks)} kernels")
        for k in ks:
            print(f"      grid={k['grid']} block={k['block']}  {demangle(k['name'])[:130]}")
            for p in k["params"][:2]:
                print(f"         p{p['index']} sz={p['size']} = {int.from_bytes(p['bytes'],'little') if p['bytes'] and p['size']<=8 else 'blob'}")
        del g
    # index_select with a >2^31-element source: canUse32BitIndexMath switch
    for rows,label in [((2**31-1)//64, "src numel < 2^31"), ((2**31)//64 + 1, "src numel > 2^31")]:
        src = buf[:rows*64].view(rows,64)
        idx = bi[:256] % rows
        try:
            g,nodes = cap(lambda s=src,i=idx: torch.index_select(s,0,i), w=1)
            print(f"  index_select {label} (src numel={src.numel()}): {[n['type'] for n in nodes]}")
            for k in kernels(nodes): print(f"      {demangle(k['name'])[:150]}")
            del g
        except Exception as e:
            print(f"  index_select {label}: {type(e).__name__} {str(e)[:150]}")

if mode=="align":
    print("## alignment / contiguity / dim-coalescing induced kernel switches (same numel!)")
    n = 1024*D
    variants = {
      "contig 2D, base-aligned":      (bx[:n].view(1024,D),   by[:n].view(1024,D),  bo[:n].view(1024,D)),
      "contig 2D, +1 float offset":   (bx[1:n+1].view(1024,D),by[1:n+1].view(1024,D),bo[1:n+1].view(1024,D)),
      "contig 2D, +2 float offset":   (bx[2:n+2].view(1024,D),by[2:n+2].view(1024,D),bo[2:n+2].view(1024,D)),
      "contig 2D, +4 float offset":   (bx[4:n+4].view(1024,D),by[4:n+4].view(1024,D),bo[4:n+4].view(1024,D)),
      "row-strided (::2) src":        (bx[:2*n].view(2048,D)[::2], by[:n].view(1024,D), bo[:n].view(1024,D)),
      "transposed src":               (bx[:n].view(D,1024).t(),    by[:n].view(1024,D), bo[:n].view(1024,D)),
      "broadcast src (row)":          (bx[:D].expand(1024,D),      by[:n].view(1024,D), bo[:n].view(1024,D)),
    }
    for tag,(x,y,o) in variants.items():
        try:
            g,nodes=cap(lambda x=x,y=y,o=o: torch.add(x,y,out=o))
            ks=kernels(nodes)
            print(f"  {tag:32s} -> {len(nodes)} node(s) func={[hex(k['func']) for k in ks]}")
            for k in ks:
                print(f"       grid={k['grid']} block={k['block']} nparams={len(k['params'])} "
                      f"psizes={[p['size'] for p in k['params']]}")
                print(f"       {demangle(k['name'])[:140]}")
        except Exception as e:
            print(f"  {tag}: ERROR {type(e).__name__} {str(e)[:120]}")

if mode=="dtype":
    print("## dtype-driven template switch at fixed shape")
    n=1024*D
    for dt in [torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int32, torch.int64]:
        x=bx[:n].to(dt); y=by[:n].to(dt); o=torch.zeros(n,dtype=dt,device=dev)
        g,nodes=cap(lambda x=x,y=y,o=o: torch.add(x,y,out=o))
        k=kernels(nodes)[0]
        print(f"  {str(dt):20s} func={k['func']:#x} grid={k['grid']} psizes={[p['size'] for p in k['params']]}")
        print(f"       {demangle(k['name'])[:130]}")
