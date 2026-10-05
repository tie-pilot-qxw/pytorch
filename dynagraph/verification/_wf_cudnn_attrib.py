"""(A) Can we tell, at capture time, which nodes a single extern (cuDNN) call produced?
Two candidate mechanisms, both tested live:
  A1. cuStreamGetCaptureInfo_v3 -> graph_out, then cuGraphGetNodes on the *in-progress*
      graph, before and after the call; the new handles are that call's nodes.
  A2. cuStreamGetCaptureInfo_v3 -> dependencies_out before/after (the capture frontier).
"""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from _wf_cudnn_lib import libcuda, graph_nodes, node_type, kernel_params, func_name

def safe_nodes(graph):
    """cuGraphGetNodes on a possibly in-progress graph; returns (list, note)."""
    n = ctypes.c_size_t(0)
    rc = libcuda.cuGraphGetNodes(ctypes.c_void_p(graph), None, ctypes.byref(n))
    if rc != 0:
        return None, f"count rc={rc}"
    if n.value == 0:
        return [], "empty"
    arr = (ctypes.c_void_p * n.value)()
    rc = libcuda.cuGraphGetNodes(ctypes.c_void_p(graph), arr, ctypes.byref(n))
    if rc != 0:
        return None, f"fetch rc={rc} (count said {n.value})"
    return [arr[i] for i in range(n.value)], "ok"

def capture_info(stream):
    status = ctypes.c_int(0); cid = ctypes.c_uint64(0)
    graph = ctypes.c_void_p(); deps = ctypes.POINTER(ctypes.c_void_p)()
    edge = ctypes.c_void_p(); ndeps = ctypes.c_size_t(0)
    rc = libcuda.cuStreamGetCaptureInfo_v3(ctypes.c_void_p(stream), ctypes.byref(status),
            ctypes.byref(cid), ctypes.byref(graph), ctypes.byref(deps),
            ctypes.byref(edge), ctypes.byref(ndeps))
    if rc != 0:
        return ("rc=%d" % rc, None, None, None)
    dl = [deps[i] for i in range(ndeps.value)] if deps else []
    return (status.value, cid.value, graph.value, dl)

DEV="cuda"; DT=torch.float16
x  = torch.randn(4, 64, 56, 56, device=DEV, dtype=DT).to(memory_format=torch.channels_last)
w  = torch.randn(64, 64, 3, 3, device=DEV, dtype=DT).to(memory_format=torch.channels_last)
x2 = torch.randn(4, 64, 56, 56, device=DEV, dtype=DT)          # NCHW -> 4-node conv
w2 = torch.randn(64, 64, 3, 3, device=DEV, dtype=DT)
q  = torch.randn(2, 8, 1024, 64, device=DEV, dtype=torch.bfloat16)

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s), sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    for _ in range(4):
        F.conv2d(x, w, padding=1); F.conv2d(x2, w2, padding=1)
        F.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch.relu(x)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()

g = torch.cuda.CUDAGraph(keep_graph=True)
LOG = []
with torch.cuda.graph(g):
    st = torch.cuda.current_stream().cuda_stream
    def snap(tag):
        stt, cid, graph, deps = capture_info(st)
        nodes, note = safe_nodes(graph) if graph else (None, "no graph")
        LOG.append((tag, stt, graph, set(nodes) if nodes is not None else None, list(deps), note))
        return nodes
    snap("begin")
    ya = F.relu(x)                # a plain aten kernel, control
    snap("after relu")
    yb = F.conv2d(x, w, padding=1)          # channels_last cuDNN conv
    snap("after conv_cl")
    yc = F.conv2d(x2, w2, padding=1)        # NCHW cuDNN conv (multi-node)
    snap("after conv_nchw")
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        yd = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    snap("after sdpa_cudnn")
torch.cuda.synchronize()

print("capture status codes (1 == ACTIVE):", [l[1] for l in LOG])
print("graph handle stable during capture:", len({l[2] for l in LOG}) == 1,
      [hex(l[2] or 0) for l in LOG])
print("cuGraphGetNodes on the IN-PROGRESS capture graph:", [l[5] for l in LOG])
print()
prev = LOG[0][3] or set()
for tag, stt, graph, nodes, deps, note in LOG[1:]:
    if nodes is None:
        print(f"--- {tag}: cuGraphGetNodes unavailable ({note}); frontier={len(deps)}")
        for nd in deps:
            t = node_type(nd); p = kernel_params(nd) if t=="KERNEL" else None
            print(f"        FRONTIER {t:8s} {(func_name(p.func) if p and p.func else '')[:95]}")
        continue
    new = nodes - prev
    print(f"--- {tag}: total nodes {len(prev)} -> {len(nodes)}   (+{len(new)} new)")
    print(f"      capture frontier (dependencies_out) = {len(deps)} node(s)")
    for nd in sorted(new, key=lambda n: n):
        t = node_type(nd)
        nm = ""
        if t == "KERNEL":
            p = kernel_params(nd)
            nm = func_name(p.func) if p and p.func else "?"
        infront = " <-FRONTIER" if nd in deps else ""
        print(f"        {t:8s} {nm[:95]}{infront}")
    prev = nodes

final = graph_nodes(g.raw_cuda_graph())
print(f"\nfinal graph after EndCapture: {len(final)} nodes; "
      f"handles identical to the ones seen mid-capture: {set(final) == (LOG[-1][3] or set())}")
print("final node types:", [node_type(n) for n in final])
