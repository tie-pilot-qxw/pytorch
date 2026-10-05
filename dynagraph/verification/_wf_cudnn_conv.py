"""(A)(B)(C) for cuDNN convolution: capture a graph containing only F.conv2d at
several batch sizes, enumerate the nodes, compare kernels, diff param bytes."""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from _wf_cudnn_lib import describe, demangle, graph_nodes, libcuda

torch.manual_seed(0)
DEV = "cuda"; DT = torch.float16
C_IN, C_OUT, HW, K = 64, 64, 56, 3

print(f"torch {torch.__version__}  cudnn {torch.backends.cudnn.version()}  "
      f"enabled={torch.backends.cudnn.enabled}")

W = torch.randn(C_OUT, C_IN, K, K, device=DEV, dtype=DT)
# one big input storage so that narrow views share the SAME data_ptr across batch sizes
XBIG = torch.randn(64, C_IN, HW, HW, device=DEV, dtype=DT)

def capture_conv(batch, bench):
    torch.backends.cudnn.benchmark = bench
    x = XBIG[:batch]
    # warmup on a side stream (required before capture; also runs cudnn plan selection)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            y = F.conv2d(x, W, padding=1)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        y = F.conv2d(x, W, padding=1)
    torch.cuda.synchronize()
    raw = g.raw_cuda_graph()
    nodes = graph_nodes(raw)
    info = describe(nodes)
    return g, info, x, y

def show(tag, info):
    print(f"\n##### {tag}: {len(info)} nodes")
    for d in info:
        if d["type"] != "KERNEL":
            print(f"  [{d['i']}] {d['type']}")
            continue
        nm = d.get("name", "?")
        pi = d.get("param_info", [])
        tot = (pi[-1][1] + pi[-1][2]) if pi else 0
        print(f"  [{d['i']}] KERNEL grid={d['grid']} block={d['block']} smem={d['smem']} "
              f"nparams={len(pi)} parambytes={tot} kernelParams={d['has_kernelParams']} extra={d['has_extra']} mod={hex(d['module'] or 0)}")
        print(f"        {nm[:150]}")
        dm = demangle(nm)
        if dm != nm:
            print(f"        demangled: {dm[:220]}")

RESULTS = {}
for bench in (False, True):
    for batch in (2, 8):
        try:
            g, info, x, y = capture_conv(batch, bench)
            RESULTS[(bench, batch)] = info
            show(f"conv2d batch={batch} cudnn.benchmark={bench}  out={tuple(y.shape)} "
                 f"x.ptr={hex(x.data_ptr())} y.ptr={hex(y.data_ptr())}", info)
        except Exception as e:
            print(f"\n##### conv2d batch={batch} bench={bench} FAILED: {type(e).__name__}: {e}")

print("\n\n########## (C) does the kernel change with batch? ##########")
for bench in (False, True):
    a = RESULTS.get((bench, 2)); b = RESULTS.get((bench, 8))
    if not a or not b:
        continue
    ka = [d.get("name") for d in a if d["type"] == "KERNEL"]
    kb = [d.get("name") for d in b if d["type"] == "KERNEL"]
    print(f"\nbenchmark={bench}: batch2 kernels={len(ka)}  batch8 kernels={len(kb)}  same_list={ka==kb}")
    for i, (x_, y_) in enumerate(zip(ka, kb)):
        mark = "SAME " if x_ == y_ else "DIFF!"
        print(f"   {mark} node{i}")
        if x_ != y_:
            print(f"      b2: {x_[:130]}")
            print(f"      b8: {y_[:130]}")

print("\n\n########## (B) param byte diff, batch 2 vs 8 ##########")
for bench in (False, True):
    a = RESULTS.get((bench, 2)); b = RESULTS.get((bench, 8))
    if not a or not b:
        continue
    print(f"\n--- benchmark={bench} ---")
    ka = [d for d in a if d["type"] == "KERNEL"]
    kb = [d for d in b if d["type"] == "KERNEL"]
    for na, nb in zip(ka, kb):
        if na.get("name") != nb.get("name"):
            print(f"  node{na['i']}: kernel differs, byte diff meaningless")
            continue
        pa = na.get("param_bytes") or []
        pb = nb.get("param_bytes") or []
        print(f"  node{na['i']} {na.get('name','?')[:80]}")
        if not pa:
            print(f"    no kernelParams array (extra={na['has_extra']}) -> params are opaque to us")
            continue
        for (i, off, sz, ba), (_, _, _, bb) in zip(pa, pb):
            if ba == bb:
                continue
            def asint(x):
                return int.from_bytes(x, "little") if x else None
            va, vb = asint(ba), asint(bb)
            kind = "ptr?" if (va and va > 0x100000000) else "scalar"
            print(f"    idx={i:2d} off={off:4d} sz={sz}  {kind}  b2={va} (0x{va:x})  b8={vb} (0x{vb:x})"
                  + (f"  ratio={vb/va:.4g}" if va and vb and va != 0 else ""))
