"""E1: same-shape-twice control (how noisy is byte diffing?)
E2: wide N sweep -> where does the kernel identity / node count change?
E3: contiguity / alignment / dim-coalescing induced kernel switches
"""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _wf_aten_common import probe, capture, kernels, demangle

torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
NB = 4*1024*1024
base_f  = torch.randn(NB, device=dev)
base_f2 = torch.randn(NB, device=dev)
base_out= torch.zeros(NB, device=dev)
base_i  = torch.randint(0, 4096, (NB,), device=dev, dtype=torch.long)
D = 64

def sig(nodes):
    """structural signature: node types + (func,block,smem) per kernel"""
    return tuple((n["type"], n.get("func"), n.get("block"), n.get("smem")) for n in nodes)

def fsig(nodes):
    return tuple(n.get("func") for n in nodes if n["type"]=="KERNEL")

# ---------------- E1: control ----------------
print("#" * 90)
print("# E1  same shape captured twice -> which param bytes are nondeterministic?")
print("#" * 90)
def ctl(tag, mk):
    ga, na = capture(mk()); gb, nb = capture(mk())
    ka, kb = kernels(na), kernels(nb)
    print(f"  {tag}: nodes {len(na)}/{len(nb)} samefuncs={fsig(na)==fsig(nb)}")
    for i,(a,b) in enumerate(zip(ka,kb)):
        for pa, pb in zip(a["params"], b["params"]):
            if pa["bytes"] != pb["bytes"]:
                nby = sum(1 for x,y in zip(pa["bytes"], pb["bytes"]) if x!=y)
                print(f"     k{i} p{pa['index']} sz={pa['size']}: {nby} of {pa['size']} bytes differ "
                      f"ACROSS IDENTICAL SHAPES")

N = 512
def f_add():
    x=base_f[:N*D].view(N,D); y=base_f2[:N*D].view(N,D); o=base_out[:N*D].view(N,D)
    return lambda: torch.add(x,y,out=o)
def f_gather():
    s=base_f[:4096*D].view(4096,D); i=(base_i[:N*D]%4096).view(N,D)
    return lambda: torch.gather(s,0,i)
def f_iput():
    d=base_out[:4096*D].view(4096,D); i=base_i[:N]%4096; v=base_f[:N*D].view(N,D)
    return lambda: d.index_put_((i,), v, accumulate=False)
def f_ln():
    x=base_f[:N*D].view(N,D); w=base_f2[:D]; b=base_f2[D:2*D]
    return lambda: torch.nn.functional.layer_norm(x,(D,),w,b,1e-5)
def f_sum():
    x=base_f[:N*D].view(N,D)
    return lambda: x.sum(dim=0)
for tag, f in [("add out=",f_add),("gather",f_gather),("index_put_ acc=F",f_iput),
               ("layer_norm",f_ln),("sum dim0",f_sum)]:
    ctl(tag, f)

# ---------------- E2: N sweep ----------------
print()
print("#" * 90)
print("# E2  sweep N: does the kernel identity / node count change with shape?")
print("#" * 90)

def mk(op, n):
    if op=="add":
        x=base_f[:n*D].view(n,D); y=base_f2[:n*D].view(n,D); o=base_out[:n*D].view(n,D)
        return lambda: torch.add(x,y,out=o)
    if op=="index_select":
        s=base_f[:4096*D].view(4096,D); i=base_i[:n]%4096
        return lambda: torch.index_select(s,0,i)
    if op=="index_put_F":
        d=base_out[:4096*D].view(4096,D); i=base_i[:n]%4096; v=base_f[:n*D].view(n,D)
        return lambda: d.index_put_((i,), v, accumulate=False)
    if op=="scatter_add":
        d=base_out[:4096*D].view(4096,D); i=(base_i[:n*D]%4096).view(n,D); s=base_f[:n*D].view(n,D)
        return lambda: d.scatter_add_(0,i,s)
    if op=="sort_rows":     # sort along last dim, n rows of D
        x=base_f[:n*D].view(n,D); return lambda: torch.sort(x,dim=-1)
    if op=="sort_len":      # sort one row of length n
        x=base_f[:n]; return lambda: torch.sort(x,dim=-1)
    if op=="topk_len":
        x=base_f[:n]; return lambda: torch.topk(x, 8, dim=-1)
    if op=="cumsum_len":
        x=base_f[:n]; return lambda: torch.cumsum(x, dim=-1)
    if op=="layer_norm":
        x=base_f[:n*D].view(n,D); w=base_f2[:D]; b=base_f2[D:2*D]
        return lambda: torch.nn.functional.layer_norm(x,(D,),w,b,1e-5)
    if op=="layer_norm_dim":   # vary the normalized dim instead
        r=64; x=base_f[:r*n].view(r,n); w=base_f2[:n]; b=base_f2[n:2*n]
        return lambda: torch.nn.functional.layer_norm(x,(n,),w,b,1e-5)
    if op=="softmax_dim":
        r=64; x=base_f[:r*n].view(r,n); return lambda: torch.softmax(x,dim=-1)
    if op=="sum_dim0":
        x=base_f[:n*D].view(n,D); return lambda: x.sum(dim=0)
    if op=="sum_all":
        x=base_f[:n]; return lambda: x.sum()
    if op=="repeat_interleave":
        x=base_f[:n*D].view(n,D); return lambda: torch.repeat_interleave(x,3,dim=0)
    raise KeyError(op)

SWEEP_N   = [1,2,3,4,7,8,15,16,31,32,64,100,128,255,256,511,512,1000,1024,2048,4096,8192,16384,32768,65536]
SWEEP_LEN = [1,2,3,4,8,15,16,31,32,64,128,256,511,512,1024,2048,4095,4096,4097,8192,16384,32768,65536,262144,1048576,4194304]

OPS = [("add",SWEEP_N),("index_select",SWEEP_N),("index_put_F",SWEEP_N),("scatter_add",SWEEP_N),
       ("sort_rows",SWEEP_N),("sort_len",SWEEP_LEN),("topk_len",SWEEP_LEN),("cumsum_len",SWEEP_LEN),
       ("layer_norm",SWEEP_N),("layer_norm_dim",[8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]),
       ("softmax_dim",[8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536]),
       ("sum_dim0",SWEEP_N),("sum_all",SWEEP_LEN),("repeat_interleave",SWEEP_N)]

for op, sweep in OPS:
    print(f"\n--- {op} ---")
    prev = None
    groups = []
    for n in sweep:
        if op in ("add","index_put_F","scatter_add","layer_norm","sum_dim0","repeat_interleave","index_select") and n*D > NB:
            continue
        if op in ("sort_rows",) and n*D > NB: continue
        if n > NB: continue
        try:
            g, nodes = capture(mk(op, n))
        except Exception as e:
            print(f"   n={n:<8} ERROR {type(e).__name__}: {str(e)[:120]}")
            prev = None; continue
        ks = kernels(nodes)
        s = (tuple(x["type"] for x in nodes), fsig(nodes),
             tuple(x["block"] for x in ks), tuple(x["smem"] for x in ks))
        if s != prev:
            groups.append((n, nodes, s))
            names = [demangle(k["name"]) for k in ks]
            print(f"   n={n:<8} NEW SIG: {len(nodes)} nodes {[x['type'] for x in nodes]} blocks={s[2]} smem={s[3]}")
            for nm in names:
                nm = nm.replace("at::native::","").replace("at_cuda_detail::cub::","cub::")
                print(f"       {nm[:150]}")
            prev = s
    print(f"   -> {len(groups)} distinct kernel signatures across {len(sweep)} shapes")
