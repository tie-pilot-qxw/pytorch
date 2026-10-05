"""Sweep real ATen CUDA ops: capture one graph per (op, shape), compare node
count / kernel identity / parameter bytes across two shapes.

To make the param diff meaningful, every input tensor is a *view of a
preallocated base buffer*, so the data pointers are identical across the two
shapes and the only param deltas that can appear are shape-derived (or
allocator-assigned output pointers, which we label).
"""
import os, sys, json, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _wf_aten_common import probe, capture, kernels, demangle, pshow

torch.cuda.init()
dev = "cuda"
torch.manual_seed(0)

FREE, TOT = torch.cuda.mem_get_info()
print(f"# free={FREE/2**30:.1f}GB total={TOT/2**30:.1f}GB", flush=True)

# ---- preallocated bases (all small: ~ 4M float = 16MB each) ----
NB = 4 * 1024 * 1024
base_f  = torch.randn(NB, device=dev)
base_f2 = torch.randn(NB, device=dev)
base_out= torch.zeros(NB, device=dev)
base_h  = torch.randn(NB, device=dev, dtype=torch.half)
base_i  = torch.randint(0, 64, (NB,), device=dev, dtype=torch.long)
base_i32= torch.randint(0, 64, (NB,), device=dev, dtype=torch.int32)

PTRS = {}
def reg(name, t):
    PTRS[t.data_ptr()] = name
    return t
for n, t in [("base_f", base_f), ("base_f2", base_f2), ("base_out", base_out),
             ("base_h", base_h), ("base_i", base_i), ("base_i32", base_i32)]:
    reg(n, t)

def classify(b, shape_vals):
    """Try to name a raw param byte-string."""
    if b is None:
        return "?"
    n = len(b)
    v = int.from_bytes(b, "little") if n <= 8 else None
    outs = []
    if n == 8:
        if v in PTRS: return f"ptr[{PTRS[v]}]"
        # near a known base?
        for p, nm in PTRS.items():
            if 0 <= v - p < NB * 8:
                return f"ptr[{nm}+{v-p}]"
        if v > 2**40:  # looks like a device pointer
            return f"devptr:{v:#x}"
    if n in (1, 2, 4, 8):
        for k, sv in shape_vals.items():
            if v == sv: return f"{k}={v}"
        return f"u{n*8}:{v}"
    return f"blob[{n}B]"


def _interp(buf, off, sz):
    return int.from_bytes(buf[off:off+sz], "little")

def blob_detail(ba, bb, sa, sb, maxrun=24):
    """Report byte-runs inside an opaque struct param that differ, and try to
    name each run by matching against shape values / known pointers."""
    n = min(len(ba), len(bb))
    runs, i = [], 0
    while i < n:
        if ba[i] != bb[i]:
            j = i
            while j < n and ba[j] != bb[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    # merge runs separated by <=3 equal bytes
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 3:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(list(r) if False else (r[0], r[1]))
        merged[-1] = tuple(merged[-1])
    named, unnamed = [], 0
    for (s0, s1) in merged:
        tag = None
        for width in (8, 4):
            base = s0 - (s0 % width)
            for start in {base, s0}:
                if start + width > len(ba): continue
                va = _interp(ba, start, width); vb = _interp(bb, start, width)
                ka = [k for k, v in sa.items() if v == va]
                kb = [k for k, v in sb.items() if v == vb]
                if ka and kb and ka[0] == kb[0]:
                    tag = f"@{start} u{width*8} {ka[0]}: {va} -> {vb}"
                    break
                if va in PTRS or vb in PTRS or (va > 2**40 and vb > 2**40 and width == 8):
                    tag = f"@{start} ptr {va:#x} -> {vb:#x}"
                    break
            if tag: break
        if tag:
            named.append(tag)
        else:
            va = _interp(ba, s0, min(8, s1-s0)); vb = _interp(bb, s0, min(8, s1-s0))
            named.append(f"@{s0}..{s1} OPAQUE {va} -> {vb}")
            unnamed += 1
    print(f"         blob diff: {len(merged)} runs ({unnamed} unnamed)")
    for t in named[:maxrun]:
        print("           " + t)
    if len(named) > maxrun:
        print(f"           ... {len(named)-maxrun} more")

def run(tag, fn_a, fn_b, shapes_a, shapes_b, note=""):
    print("=" * 100)
    print(f"## {tag}   A={shapes_a}  B={shapes_b}  {note}", flush=True)
    try:
        ga, na = capture(fn_a)
        gb, nb = capture(fn_b)
    except Exception as e:
        print("   CAPTURE FAILED:", type(e).__name__, str(e)[:300])
        return
    ka, kb = kernels(na), kernels(nb)
    print(f"   nodes: A={len(na)} ({[x['type'] for x in na]})")
    print(f"          B={len(nb)} ({[x['type'] for x in nb]})")
    if len(ka) != len(kb):
        print("   *** KERNEL COUNT DIFFERS ***")
    for i in range(max(len(ka), len(kb))):
        a = ka[i] if i < len(ka) else None
        b = kb[i] if i < len(kb) else None
        print(f"   --- kernel[{i}] ---")
        if a is None or b is None:
            one = a or b
            print(f"     only in {'A' if a else 'B'}: {demangle(one['name'])}")
            continue
        same_fn = a["func"] == b["func"]
        print(f"     name A: {demangle(a['name'])}")
        if a["name"] != b["name"]:
            print(f"     name B: {demangle(b['name'])}")
        print(f"     samefunc={same_fn}  funcA={a['func']:#x} funcB={b['func']:#x}")
        print(f"     gridA={a['grid']} blockA={a['block']} smemA={a['smem']}")
        print(f"     gridB={b['grid']} blockB={b['block']} smemB={b['smem']}")
        if not same_fn:
            print("     *** DIFFERENT KERNEL -> cudaGraphExecKernelNodeSetParams NOT ENOUGH ***")
            print(f"     paramsA n={len(a['params'])} paramsB n={len(b['params'])}")
            continue
        nd = 0
        for pa, pb in zip(a["params"], b["params"]):
            ca = classify(pa["bytes"], shapes_a)
            cb = classify(pb["bytes"], shapes_b)
            chg = pa["bytes"] != pb["bytes"]
            if chg: nd += 1
            mark = "CHG" if chg else "   "
            print(f"     {mark} p{pa['index']:<2} off={pa['offset']:<4} sz={pa['size']:<3} A={ca:<28} B={cb}")
            if chg and pa["size"] > 8 and pa["bytes"] and pb["bytes"]:
                blob_detail(pa["bytes"], pb["bytes"], shapes_a, shapes_b)
        print(f"     changed params: {nd}/{len(a['params'])}")

# ============================ ops ============================
D = 64

def mk_add(n):
    x = base_f[:n*D].view(n, D); y = base_f2[:n*D].view(n, D); o = base_out[:n*D].view(n, D)
    return lambda: torch.add(x, y, out=o)

def mk_index_select(n):
    src = base_f[:4096*D].view(4096, D); idx = base_i[:n] % 4096
    return lambda: torch.index_select(src, 0, idx)

def mk_embedding(n):
    w = base_f[:4096*D].view(4096, D); idx = base_i[:n] % 4096
    return lambda: torch.nn.functional.embedding(idx, w)

def mk_index_put(n, acc):
    dst = base_out[:4096*D].view(4096, D); idx = base_i[:n] % 4096
    val = base_f[:n*D].view(n, D)
    return lambda: dst.index_put_((idx,), val, accumulate=acc)

def mk_scatter_add(n):
    dst = base_out[:4096*D].view(4096, D); idx = (base_i[:n*D] % 4096).view(n, D)
    src = base_f[:n*D].view(n, D)
    return lambda: dst.scatter_add_(0, idx, src)

def mk_gather(n):
    src = base_f[:4096*D].view(4096, D); idx = (base_i[:n*D] % 4096).view(n, D)
    return lambda: torch.gather(src, 0, idx)

def mk_sort(n, d=D):
    x = base_f[:n*d].view(n, d)
    return lambda: torch.sort(x, dim=-1)

def mk_topk(n, d=D, k=8):
    x = base_f[:n*d].view(n, d)
    return lambda: torch.topk(x, k, dim=-1)

def mk_cumsum(n, d=D):
    x = base_f[:n*d].view(n, d)
    return lambda: torch.cumsum(x, dim=-1)

def mk_layernorm(n, d=D):
    x = base_f[:n*d].view(n, d); w = base_f2[:d]; b = base_f2[d:2*d]
    return lambda: torch.nn.functional.layer_norm(x, (d,), w, b, 1e-5)

def mk_softmax(n, d=D):
    x = base_f[:n*d].view(n, d)
    return lambda: torch.softmax(x, dim=-1)

def mk_repeat_interleave(n):
    x = base_f[:n*D].view(n, D)
    return lambda: torch.repeat_interleave(x, 3, dim=0)

def mk_sum(n):
    x = base_f[:n*D].view(n, D)
    return lambda: x.sum(dim=0)

N1, N2 = 512, 1024
sh = lambda n, **kw: dict(N=n, ND=n*D, D=D, **kw)

CASES = [
 ("add (TensorIterator elementwise)", mk_add(N1), mk_add(N2), sh(N1), sh(N2)),
 ("index_select",        mk_index_select(N1), mk_index_select(N2), sh(N1), sh(N2)),
 ("embedding",           mk_embedding(N1), mk_embedding(N2), sh(N1), sh(N2)),
 ("index_put_ acc=False",mk_index_put(N1, False), mk_index_put(N2, False), sh(N1), sh(N2)),
 ("index_put_ acc=True", mk_index_put(N1, True), mk_index_put(N2, True), sh(N1), sh(N2)),
 ("scatter_add_",        mk_scatter_add(N1), mk_scatter_add(N2), sh(N1), sh(N2)),
 ("gather",              mk_gather(N1), mk_gather(N2), sh(N1), sh(N2)),
 ("sort dim=-1",         mk_sort(N1), mk_sort(N2), sh(N1), sh(N2)),
 ("topk k=8",            mk_topk(N1), mk_topk(N2), sh(N1), sh(N2)),
 ("cumsum dim=-1",       mk_cumsum(N1), mk_cumsum(N2), sh(N1), sh(N2)),
 ("layer_norm",          mk_layernorm(N1), mk_layernorm(N2), sh(N1), sh(N2)),
 ("softmax",             mk_softmax(N1), mk_softmax(N2), sh(N1), sh(N2)),
 ("repeat_interleave x3",mk_repeat_interleave(N1), mk_repeat_interleave(N2), sh(N1), sh(N2)),
 ("sum dim=0",           mk_sum(N1), mk_sum(N2), sh(N1), sh(N2)),
]

for tag, fa, fb, sa, sb in CASES:
    try:
        run(tag, fa, fb, sa, sb)
    except Exception as e:
        print(f"## {tag}: ERROR {type(e).__name__}: {str(e)[:300]}")
