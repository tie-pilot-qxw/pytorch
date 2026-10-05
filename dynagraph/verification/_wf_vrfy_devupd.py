"""Probe the verdict's ARCHITECTURAL blocker:
 'the Triton deviceUpdatable path (device-side graph edit inside the same launch)
  is unavailable for extern nodes.'
The cited doc restriction is about device-side cudaGraphLaunch()/CDP, NOT about
device-UPDATABLE nodes.  CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE is
documented as 'Valid for graph nodes, launches' -> try to set it on an ALREADY
CAPTURED cuDNN node via cuGraphKernelNodeSetAttribute, before instantiation.
"""
import ctypes
import torch, torch.nn.functional as F

libcuda = ctypes.CDLL("libcuda.so.1")
CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13

def errname(rc):
    s = ctypes.c_char_p(); libcuda.cuGetErrorName(ctypes.c_int(rc), ctypes.byref(s))
    return f"{rc}({s.value.decode() if s.value else '?'})"

def nodes_of(graph):
    n = ctypes.c_size_t(0)
    libcuda.cuGraphGetNodes(ctypes.c_void_p(graph), None, ctypes.byref(n))
    arr = (ctypes.c_void_p*n.value)()
    libcuda.cuGraphGetNodes(ctypes.c_void_p(graph), arr, ctypes.byref(n))
    return [arr[i] for i in range(n.value)]

def ntype(nd):
    t = ctypes.c_int(0); libcuda.cuGraphNodeGetType(ctypes.c_void_p(nd), ctypes.byref(t)); return t.value

class KNP(ctypes.Structure):
    _fields_ = [("func", ctypes.c_void_p),("gx",ctypes.c_uint),("gy",ctypes.c_uint),("gz",ctypes.c_uint),
                ("bx",ctypes.c_uint),("by",ctypes.c_uint),("bz",ctypes.c_uint),("smem",ctypes.c_uint),
                ("kernelParams",ctypes.POINTER(ctypes.c_void_p)),("extra",ctypes.POINTER(ctypes.c_void_p)),
                ("kern",ctypes.c_void_p),("ctx",ctypes.c_void_p)]

def fname(f):
    s = ctypes.c_char_p()
    return s.value.decode(errors="replace") if libcuda.cuFuncGetName(ctypes.byref(s), ctypes.c_void_p(f))==0 else "?"

HOLD=[]
def cap_no_instantiate(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): out = fn()
    torch.cuda.synchronize(); HOLD.append((g,out))
    already = True
    try: g.raw_cuda_graph_exec()
    except Exception as e: already = False
    return g, out, nodes_of(g.raw_cuda_graph()), already

DEV="cuda"
W = torch.randn(64,64,3,3,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)
X = torch.randn(8,64,56,56,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)

for label, fn in [("aten relu (baseline, also not a Triton kernel)", lambda: torch.relu(torch.ones(1024, device=DEV))),
                  ("cuDNN conv2d channels_last b=8", lambda: F.conv2d(X, W, padding=1))]:
    print("="*90); print("###", label)
    g,out,nds,already = cap_no_instantiate(fn)
    print(f"  already instantiated by torch at capture_end: {already}")
    kn = [z for z in nds if ntype(z)==0]
    print(f"  nodes={len(nds)} kernel nodes={len(kn)}  first kernel={fname(KNP().func) if not kn else ''}")
    p = KNP(); libcuda.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(kn[0]), ctypes.byref(p))
    print(f"  kernel: {fname(p.func)[:90]}")
    buf = (ctypes.c_ubyte*256)()
    ctypes.memset(buf, 0, 256)
    ctypes.cast(buf, ctypes.POINTER(ctypes.c_int))[0] = 1   # deviceUpdatable = 1
    rc = libcuda.cuGraphKernelNodeSetAttribute(ctypes.c_void_p(kn[0]),
            ctypes.c_int(CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE), ctypes.byref(buf))
    print(f"  cuGraphKernelNodeSetAttribute(DEVICE_UPDATABLE_KERNEL_NODE=1) rc={errname(rc)}")
    if rc == 0:
        devnode = ctypes.cast(ctypes.byref(buf, 8), ctypes.POINTER(ctypes.c_void_p))[0]
        print(f"    devNode handle returned = 0x{(devnode or 0):x}")
        rb = (ctypes.c_ubyte*256)(); ctypes.memset(rb,0,256)
        rc2 = libcuda.cuGraphKernelNodeGetAttribute(ctypes.c_void_p(kn[0]),
                ctypes.c_int(CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE), ctypes.byref(rb))
        print(f"    readback rc={errname(rc2)} deviceUpdatable={ctypes.cast(rb,ctypes.POINTER(ctypes.c_int))[0]} "
              f"devNode=0x{(ctypes.cast(ctypes.byref(rb,8),ctypes.POINTER(ctypes.c_void_p))[0] or 0):x}")
        try:
            g.instantiate(); print("    instantiate() after opting in: OK")
            ref = fn(); torch.cuda.synchronize()
            g.replay(); torch.cuda.synchronize()
            ok = torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
            print(f"    replay still correct: {bool(ok)}  any_nan={bool(torch.isnan(out).any())}")
        except Exception as e:
            print(f"    instantiate/replay FAILED: {type(e).__name__}: {str(e)[:200]}")
    print()
