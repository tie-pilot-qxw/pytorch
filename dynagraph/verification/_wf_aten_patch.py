"""E4: the real DynaGraph move, on ATen kernels.

For each op: capture graph@shapeA and graph@shapeB (all buffers preallocated &
identical, out= everywhere, so the two captures differ only in shape).
Then take graphB's *recorded kernel-node parameter bytes + grid/block/smem* and
write them into graphA's INSTANTIATED exec graph with
cuGraphExecKernelNodeSetParams. Replay execA. Compare against an eager
reference computed at shape B.

This tests (A)+(B)+(C) at once without needing to understand the structs:
if the kernel is the same and the node set is the same, blind blob transplant
should be sufficient.
"""
import sys, os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _wf_aten_common import probe, kernels, demangle

torch.cuda.init(); dev="cuda"; torch.manual_seed(0)
NB = 2*1024*1024
base_x   = torch.randn(NB, device=dev)
base_y   = torch.randn(NB, device=dev)
base_o   = torch.zeros(NB, device=dev)
base_o2  = torch.zeros(NB, device=dev)
base_idx = torch.randint(0, 1024, (NB,), device=dev, dtype=torch.long)
base_idxm= (base_idx % 1024).contiguous()   # persistent, not a capture-pool temp
base_li  = torch.zeros(NB, device=dev, dtype=torch.long)
D = 64

def cap(fn, warmup=3):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    return g, probe.dump_graph(g.raw_cuda_graph())

def flat_params(node):
    total = max((p["offset"]+p["size"]) for p in node["params"]) if node["params"] else 0
    buf = bytearray(total)
    for p in node["params"]:
        if p["bytes"] is None: return None
        buf[p["offset"]:p["offset"]+p["size"]] = p["bytes"]
    return bytes(buf)

def transplant(gA, nodesA, nodesB, allow_grid=True, patch_block=True):
    """Blindly copy every node's recorded params from graph B into exec A."""
    assert len(nodesA)==len(nodesB)
    gA.instantiate()
    exe = gA.raw_cuda_graph_exec()
    report=[]
    for a,b in zip(nodesA,nodesB):
        if a["type"] != b["type"]: return f"node type mismatch {a['type']}/{b['type']}"
        r = probe.copy_node_params(exe, a["node_handle"], b["node_handle"])
        if r != "ok":
            return f"{a['type']} node: {r}"
        if a["type"]=="KERNEL" and (a["block"]!=b["block"] or a["smem"]!=b["smem"] or a["grid"]!=b["grid"]):
            report.append(f"{a['grid']}/{a['block']}/{a['smem']} -> {b['grid']}/{b['block']}/{b['smem']}")
    return "OK" + (("  [launch cfg changed: " + "; ".join(report) + "]") if report else "")

def check(tag, mkA, mkB, refB, outview_B, note=""):
    print("="*94)
    print(f"## {tag}  {note}")
    try:
        gA, nA = cap(mkA()); gB, nB = cap(mkB())
    except Exception as e:
        print("   capture failed:", type(e).__name__, str(e)[:200]); return
    tA=[n['type'] for n in nA]; tB=[n['type'] for n in nB]
    print(f"   nodes A={tA}\n   nodes B={tB}")
    if tA != tB:
        print("   *** NODE STRUCTURE DIFFERS -> structural fallback required, stop"); return
    r = transplant(gA, nA, nB)
    print("   transplant:", r)
    if not r.startswith("OK"): return
    ref = refB()            # eager reference at shape B
    ref = ref.clone()
    base_o.zero_(); base_o2.zero_(); base_li.zero_()
    torch.cuda.synchronize()
    gA.replay()
    torch.cuda.synchronize()
    got = outview_B()
    ok = torch.allclose(got, ref, rtol=1e-4, atol=1e-5)
    md = (got-ref).abs().max().item() if got.numel() else 0.0
    print(f"   REPLAY-AFTER-PATCH numerically {'CORRECT' if ok else 'WRONG'}  maxdiff={md:.3e}  n={got.numel()}")

NA, NB_ = 512, 1024
WHICH = sys.argv[1] if len(sys.argv)>1 else "all"
_orig_check = check
def check(tag,*a,**k):
    if WHICH!="all" and WHICH not in tag: return
    return _orig_check(tag,*a,**k)

# ---- add, out= ----
def A_add():
    x=base_x[:NA*D].view(NA,D); y=base_y[:NA*D].view(NA,D); o=base_o[:NA*D].view(NA,D)
    return lambda: torch.add(x,y,out=o)
def B_add():
    x=base_x[:NB_*D].view(NB_,D); y=base_y[:NB_*D].view(NB_,D); o=base_o[:NB_*D].view(NB_,D)
    return lambda: torch.add(x,y,out=o)
check("add (out=)", A_add, B_add,
      lambda: (base_x[:NB_*D].view(NB_,D)+base_y[:NB_*D].view(NB_,D)),
      lambda: base_o[:NB_*D].view(NB_,D))

# ---- index_select, out= ----
def A_is():
    s=base_x[:1024*D].view(1024,D); i=base_idx[:NA]; o=base_o[:NA*D].view(NA,D)
    return lambda: torch.index_select(s,0,i,out=o)
def B_is():
    s=base_x[:1024*D].view(1024,D); i=base_idx[:NB_]; o=base_o[:NB_*D].view(NB_,D)
    return lambda: torch.index_select(s,0,i,out=o)
check("index_select (out=)", A_is, B_is,
      lambda: torch.index_select(base_x[:1024*D].view(1024,D),0,base_idx[:NB_]),
      lambda: base_o[:NB_*D].view(NB_,D))

# ---- cumsum, out= ----
def A_cs():
    x=base_x[:NA*D]; o=base_o[:NA*D]; return lambda: torch.cumsum(x,0,out=o)
def B_cs():
    x=base_x[:NB_*D]; o=base_o[:NB_*D]; return lambda: torch.cumsum(x,0,out=o)
check("cumsum (out=)", A_cs, B_cs,
      lambda: torch.cumsum(base_x[:NB_*D],0),
      lambda: base_o[:NB_*D])

# ---- scatter_add_ in place ----
def A_sa():
    d=base_o[:1024*D].view(1024,D); i=base_idxm[:NA*D].view(NA,D); s=base_x[:NA*D].view(NA,D)
    return lambda: d.scatter_add_(0,i,s)
def B_sa():
    d=base_o[:1024*D].view(1024,D); i=base_idxm[:NB_*D].view(NB_,D); s=base_x[:NB_*D].view(NB_,D)
    return lambda: d.scatter_add_(0,i,s)
def ref_sa():
    t=torch.zeros(1024,D,device=dev)
    t.scatter_add_(0,base_idxm[:NB_*D].view(NB_,D), base_x[:NB_*D].view(NB_,D)); return t
check("scatter_add_ (inplace)", A_sa, B_sa, ref_sa, lambda: base_o[:1024*D].view(1024,D),
      note="persistent index buffer")

# ---- index_put_ accumulate=False ----
def A_ip():
    d=base_o[:1024*D].view(1024,D); i=base_idx[:NA]; v=base_x[:NA*D].view(NA,D)
    return lambda: d.index_put_((i,), v, accumulate=False)
def B_ip():
    d=base_o[:1024*D].view(1024,D); i=base_idx[:NB_]; v=base_x[:NB_*D].view(NB_,D)
    return lambda: d.index_put_((i,), v, accumulate=False)
def ref_ip():
    t=torch.zeros(1024,D,device=dev); t.index_put_((base_idx[:NB_],), base_x[:NB_*D].view(NB_,D)); return t
check("index_put_ acc=False (inplace)", A_ip, B_ip, ref_ip, lambda: base_o[:1024*D].view(1024,D))

# ---- sum over all: block+smem change ----
def A_sum():
    x=base_x[:512]; o=base_o[:1]; return lambda: torch.sum(x, dim=0, out=o)
def B_sum():
    x=base_x[:2048]; o=base_o[:1]; return lambda: torch.sum(x, dim=0, out=o)
check("sum() 512->2048 (block 128->512, smem 512->2048)", A_sum, B_sum,
      lambda: torch.sum(base_x[:2048],dim=0,keepdim=True),
      lambda: base_o[:1])

# ---- softmax over a power-of-2 dim: DIFFERENT template ----
def A_sm():
    x=base_x[:64*256].view(64,256); o=base_o[:64*256].view(64,256)
    return lambda: torch.softmax(x,dim=-1,out=o)
def B_sm():
    x=base_x[:64*512].view(64,512); o=base_o[:64*512].view(64,512)
    return lambda: torch.softmax(x,dim=-1,out=o)
check("softmax dim 256->512 (LOG2 template param differs)", A_sm, B_sm,
      lambda: torch.softmax(base_x[:64*512].view(64,512),dim=-1),
      lambda: base_o[:64*512].view(64,512))

# ---- gather (out=) ----
def A_g():
    s=base_x[:1024*D].view(1024,D); i=base_idxm[:NA*D].view(NA,D); o=base_o[:NA*D].view(NA,D)
    return lambda: torch.gather(s,0,i,out=o)
def B_g():
    s=base_x[:1024*D].view(1024,D); i=base_idxm[:NB_*D].view(NB_,D); o=base_o[:NB_*D].view(NB_,D)
    return lambda: torch.gather(s,0,i,out=o)
check("gather (out=)", A_g, B_g,
      lambda: torch.gather(base_x[:1024*D].view(1024,D),0,base_idxm[:NB_*D].view(NB_,D)),
      lambda: base_o[:NB_*D].view(NB_,D))

# ---- layer_norm: vary rows (out via native_layer_norm has no out=, use F.layer_norm into graph pool) ----
# instead vary rows on a fixed out buffer using torch.nn.functional.layer_norm + copy_
def A_ln():
    x=base_x[:NA*D].view(NA,D); w=base_y[:D]; b=base_y[D:2*D]; o=base_o[:NA*D].view(NA,D)
    return lambda: o.copy_(torch.nn.functional.layer_norm(x,(D,),w,b,1e-5))
def B_ln():
    x=base_x[:NB_*D].view(NB_,D); w=base_y[:D]; b=base_y[D:2*D]; o=base_o[:NB_*D].view(NB_,D)
    return lambda: o.copy_(torch.nn.functional.layer_norm(x,(D,),w,b,1e-5))
check("layer_norm rows 512->1024 (+copy_)", A_ln, B_ln,
      lambda: torch.nn.functional.layer_norm(base_x[:NB_*D].view(NB_,D),(D,),base_y[:D],base_y[D:2*D],1e-5),
      lambda: base_o[:NB_*D].view(NB_,D))

# ---- index_select across the 16/31 kernel boundary ----
def A_is16():
    s=base_x[:1024*D].view(1024,D); i=base_idxm[:16]; o=base_o[:16*D].view(16,D)
    return lambda: torch.index_select(s,0,i,out=o)
def B_is64():
    s=base_x[:1024*D].view(1024,D); i=base_idxm[:64]; o=base_o[:64*D].view(64,D)
    return lambda: torch.index_select(s,0,i,out=o)
check("index_select 16->64 (crosses indexSelectSmallIndex boundary)", A_is16, B_is64,
      lambda: torch.index_select(base_x[:1024*D].view(1024,D),0,base_idxm[:64]),
      lambda: base_o[:64*D].view(64,D))

# ---- sort across the 4096 boundary (node count changes) ----
def A_s():
    x=base_x[:1024].clone(); return lambda: torch.sort(x,dim=-1)
def B_s():
    x=base_x[:8192].clone(); return lambda: torch.sort(x,dim=-1)
check("sort len 1024->8192 (crosses single-tile / cub-onesweep boundary)", A_s, B_s,
      lambda: torch.sort(base_x[:8192])[0], lambda: base_o[:8192])
