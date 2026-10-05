"""Shared helpers: enumerate CUDA-graph nodes and read kernel-node params byte-wise."""
from __future__ import annotations
import ctypes, subprocess

libcuda = ctypes.CDLL("libcuda.so.1")

CU_GRAPH_NODE_TYPE = {
    0: "KERNEL", 1: "MEMCPY", 2: "MEMSET", 3: "HOST", 4: "GRAPH", 5: "EMPTY",
    6: "WAIT_EVENT", 7: "EVENT_RECORD", 8: "EXT_SEMAS_SIGNAL", 9: "EXT_SEMAS_WAIT",
    10: "MEM_ALLOC", 11: "MEM_FREE", 12: "BATCH_MEM_OP", 13: "CONDITIONAL",
}

class KNP(ctypes.Structure):
    _fields_ = [("func", ctypes.c_void_p),
                ("gridDimX", ctypes.c_uint), ("gridDimY", ctypes.c_uint), ("gridDimZ", ctypes.c_uint),
                ("blockDimX", ctypes.c_uint), ("blockDimY", ctypes.c_uint), ("blockDimZ", ctypes.c_uint),
                ("sharedMemBytes", ctypes.c_uint),
                ("kernelParams", ctypes.POINTER(ctypes.c_void_p)),
                ("extra", ctypes.POINTER(ctypes.c_void_p)),
                ("kern", ctypes.c_void_p),
                ("ctx", ctypes.c_void_p)]

def _chk(rc, what):
    if rc != 0:
        s = ctypes.c_char_p()
        libcuda.cuGetErrorName(ctypes.c_int(rc), ctypes.byref(s))
        raise RuntimeError(f"{what} -> {rc} {s.value}")

def graph_nodes(graph_handle):
    n = ctypes.c_size_t(0)
    _chk(libcuda.cuGraphGetNodes(ctypes.c_void_p(graph_handle), None, ctypes.byref(n)), "cuGraphGetNodes(count)")
    arr = (ctypes.c_void_p * n.value)()
    _chk(libcuda.cuGraphGetNodes(ctypes.c_void_p(graph_handle), arr, ctypes.byref(n)), "cuGraphGetNodes")
    return [arr[i] for i in range(n.value)]

def node_type(node):
    t = ctypes.c_int(0)
    _chk(libcuda.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(t)), "cuGraphNodeGetType")
    return CU_GRAPH_NODE_TYPE.get(t.value, f"?{t.value}")

def kernel_params(node):
    p = KNP()
    rc = libcuda.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(node), ctypes.byref(p))
    if rc != 0:
        return None
    return p

def func_name(func):
    s = ctypes.c_char_p()
    rc = libcuda.cuFuncGetName(ctypes.byref(s), ctypes.c_void_p(func))
    if rc != 0:
        return f"<cuFuncGetName rc={rc}>"
    return s.value.decode(errors="replace")

def func_module(func):
    m = ctypes.c_void_p()
    rc = libcuda.cuFuncGetModule(ctypes.byref(m), ctypes.c_void_p(func))
    return None if rc != 0 else m.value

def param_info(func, n_max=64):
    """[(idx, offset, size)] from cuFuncGetParamInfo."""
    out = []
    off = ctypes.c_size_t(); sz = ctypes.c_size_t()
    for i in range(n_max):
        rc = libcuda.cuFuncGetParamInfo(ctypes.c_void_p(func), ctypes.c_size_t(i),
                                        ctypes.byref(off), ctypes.byref(sz))
        if rc != 0:
            break
        out.append((i, off.value, sz.value))
    return out

def read_param_bytes(p, pi):
    """Read the raw bytes of every kernel param. Returns [(idx, off, size, bytes)]."""
    out = []
    if not p.kernelParams:
        return out
    for (i, off, size) in pi:
        addr = p.kernelParams[i]
        if not addr:
            out.append((i, off, size, None)); continue
        buf = (ctypes.c_ubyte * size).from_address(addr)
        out.append((i, off, size, bytes(buf)))
    return out

def demangle(name):
    try:
        r = subprocess.run(["c++filt", name], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or name
    except Exception:
        return name

def describe(nodes, want_params=True):
    """-> list of dicts"""
    info = []
    for idx, nd in enumerate(nodes):
        d = {"i": idx, "type": node_type(nd), "node": nd}
        if d["type"] == "KERNEL":
            p = kernel_params(nd)
            if p is None:
                d["err"] = "cuGraphKernelNodeGetParams failed"
            else:
                d["func"] = p.func
                d["name"] = func_name(p.func) if p.func else "<null func>"
                d["grid"] = (p.gridDimX, p.gridDimY, p.gridDimZ)
                d["block"] = (p.blockDimX, p.blockDimY, p.blockDimZ)
                d["smem"] = p.sharedMemBytes
                d["has_kernelParams"] = bool(p.kernelParams)
                d["has_extra"] = bool(p.extra)
                d["module"] = func_module(p.func) if p.func else None
                if want_params and p.func:
                    pi = param_info(p.func)
                    d["param_info"] = pi
                    d["param_bytes"] = read_param_bytes(p, pi) if p.kernelParams else []
        info.append(d)
    return info
