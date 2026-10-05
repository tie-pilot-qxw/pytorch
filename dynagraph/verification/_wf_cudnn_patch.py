"""End-to-end: take a graph captured at shape S1, host-patch every node's params to the
values a capture at shape S2 produced (cuGraphExecKernelNodeSetParams /
cuGraphExecMemsetNodeSetParams), replay, and check the result against eager at S2.
This is the real test of "can a cuDNN node be re-parameterized instead of re-captured"."""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from _wf_cudnn_lib import (libcuda, graph_nodes, node_type, kernel_params,
                           func_name, param_info, KNP)

class MSP(ctypes.Structure):
    _fields_ = [("dst", ctypes.c_ulonglong), ("pitch", ctypes.c_size_t),
                ("value", ctypes.c_uint), ("elementSize", ctypes.c_uint),
                ("width", ctypes.c_size_t), ("height", ctypes.c_size_t)]

def memset_params(node):
    p = MSP()
    rc = libcuda.cuGraphMemsetNodeGetParams(ctypes.c_void_p(node), ctypes.byref(p))
    return p if rc == 0 else None

def err(rc):
    s = ctypes.c_char_p(); libcuda.cuGetErrorName(ctypes.c_int(rc), ctypes.byref(s))
    return f"{rc} {s.value.decode()}"

KEEPALIVE = []
def patch_node(exec_handle, node_dst, node_src):
    """copy node_src's params onto node_dst inside exec_handle."""
    t = node_type(node_src)
    if t != node_type(node_dst):
        return f"type mismatch {node_type(node_dst)} vs {t}"
    if t == "MEMSET":
        ps = memset_params(node_src)
        rc = libcuda.cuGraphExecMemsetNodeSetParams(ctypes.c_void_p(exec_handle),
                ctypes.c_void_p(node_dst), ctypes.byref(ps), ctypes.c_void_p(0))
        return "ok" if rc == 0 else "MEMSET " + err(rc)
    if t != "KERNEL":
        return f"skipped ({t})"
    ps = kernel_params(node_src)
    pi = param_info(ps.func)
    n = len(pi)
    bufs = []
    ptrs = (ctypes.c_void_p * n)()
    for i, (_, off, sz) in enumerate(pi):
        b = (ctypes.c_ubyte * sz).from_buffer_copy(
            bytes((ctypes.c_ubyte * sz).from_address(ps.kernelParams[i])))
        bufs.append(b); ptrs[i] = ctypes.cast(b, ctypes.c_void_p)
    KEEPALIVE.append((bufs, ptrs))
    new = KNP()
    new.func = ps.func
    new.gridDimX, new.gridDimY, new.gridDimZ = ps.gridDimX, ps.gridDimY, ps.gridDimZ
    new.blockDimX, new.blockDimY, new.blockDimZ = ps.blockDimX, ps.blockDimY, ps.blockDimZ
    new.sharedMemBytes = ps.sharedMemBytes
    new.kernelParams = ctypes.cast(ptrs, ctypes.POINTER(ctypes.c_void_p))
    new.extra = None; new.kern = None; new.ctx = None
    rc = libcuda.cuGraphExecKernelNodeSetParams_v2(ctypes.c_void_p(exec_handle),
            ctypes.c_void_p(node_dst), ctypes.byref(new))
    return "ok" if rc == 0 else err(rc)

HOLD = []
def cap(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(4): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    HOLD.append((g, out))
    return g, out, graph_nodes(g.raw_cuda_graph())

def trial(name, fnA, fnB, ref_fn, rtol=2e-2, atol=2e-2):
    print(f"\n{'='*90}\n### {name}")
    gA, outA, nA = cap(fnA)
    gB, outB, nB = cap(fnB)
    kindsA = [node_type(n) for n in nA]; kindsB = [node_type(n) for n in nB]
    namesA = [func_name(kernel_params(n).func) if node_type(n)=="KERNEL" else "-" for n in nA]
    namesB = [func_name(kernel_params(n).func) if node_type(n)=="KERNEL" else "-" for n in nB]
    print(f"  S1 nodes {kindsA}\n  S2 nodes {kindsB}")
    print(f"  same kernel list: {namesA == namesB}")
    if namesA != namesB:
        for a, b in zip(namesA, namesB):
            if a != b: print(f"    S1 {a[:80]}\n    S2 {b[:80]}")
    if len(nA) != len(nB):
        print("  -> node COUNT differs; SetParams cannot express this at all."); return
    gA.instantiate()
    ex = gA.raw_cuda_graph_exec()
    ref = ref_fn()
    torch.cuda.synchronize()
    outB.copy_(torch.full_like(outB, float("nan")))
    torch.cuda.synchronize()
    ok = True
    for i, (da, db) in enumerate(zip(nA, nB)):
        r = patch_node(ex, da, db)
        print(f"    patch node{i} ({node_type(da)}): {r}")
        if r != "ok": ok = False
    if not ok:
        print("  -> patch rejected by the driver."); return
    gA.replay(); torch.cuda.synchronize()
    good = torch.allclose(outB.float(), ref.float(), rtol=rtol, atol=atol)
    nan = bool(torch.isnan(outB).any())
    md = (outB.float() - ref.float()).abs().max().item() if not nan else float("nan")
    print(f"  -> replay of the PATCHED S1 graph vs eager at S2: allclose={good} "
          f"any_nan={nan} max_abs_diff={md:.4g}")

DEV="cuda"; DTH=torch.float16
W  = torch.randn(64,64,3,3,device=DEV,dtype=DTH).to(memory_format=torch.channels_last)
XB = torch.randn(32,64,56,56,device=DEV,dtype=DTH).to(memory_format=torch.channels_last)
def conv(b): return lambda: F.conv2d(XB[:b], W, padding=1)

# 1. conv, same kernel group (b=13 and b=16 both pick sm90_xmma ...256x64x32)
trial("conv2d channels_last  b=13 -> b=16  (same cuDNN kernel)",
      conv(13), conv(16), lambda: F.conv2d(XB[:16], W, padding=1))
# 2. conv, kernel changes
trial("conv2d channels_last  b=4 -> b=8   (cuDNN picks a DIFFERENT kernel)",
      conv(4), conv(8), lambda: F.conv2d(XB[:8], W, padding=1))
# 3. conv, kernel changes AND node count changes
trial("conv2d channels_last  b=16 -> b=32  (different kernel + extra MEMSET)",
      conv(16), conv(32), lambda: F.conv2d(XB[:32], W, padding=1))

DTB = torch.bfloat16
QB = torch.randn(2,8,2048,64,device=DEV,dtype=DTB)
def sdpa(bk, L):
    def f():
        with sdpa_kernel(bk):
            return F.scaled_dot_product_attention(QB[:,:,:L], QB[:,:,:L], QB[:,:,:L], is_causal=True)
    return f
trial("SDPA EFFICIENT (PyTorch cutlass mem-eff)  L=512 -> 1024",
      sdpa(SDPBackend.EFFICIENT_ATTENTION,512), sdpa(SDPBackend.EFFICIENT_ATTENTION,1024),
      sdpa(SDPBackend.EFFICIENT_ATTENTION,1024))
trial("SDPA CUDNN  L=512 -> 1024",
      sdpa(SDPBackend.CUDNN_ATTENTION,512), sdpa(SDPBackend.CUDNN_ATTENTION,1024),
      sdpa(SDPBackend.CUDNN_ATTENTION,1024))
trial("SDPA FLASH (pytorch_flash)  L=1024 -> 1536  (same kernel)",
      sdpa(SDPBackend.FLASH_ATTENTION,1024), sdpa(SDPBackend.FLASH_ATTENTION,1536),
      sdpa(SDPBackend.FLASH_ATTENTION,1536))
trial("SDPA FLASH  L=1024 -> 512  (splitkv path: 1 node -> 2 nodes)",
      sdpa(SDPBackend.FLASH_ATTENTION,1024), sdpa(SDPBackend.FLASH_ATTENTION,512),
      sdpa(SDPBackend.FLASH_ATTENTION,512))
