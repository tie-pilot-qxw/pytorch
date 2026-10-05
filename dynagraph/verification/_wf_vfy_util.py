"""Independent graph-node introspection via the CUDA *driver* API (no dot parsing)."""
import ctypes, torch

_cu = ctypes.CDLL("libcuda.so.1")

NODE_TYPES = {0:"KERNEL",1:"MEMCPY",2:"MEMSET",3:"HOST",4:"GRAPH",5:"EMPTY",6:"WAIT_EVENT",
              7:"EVENT_RECORD",8:"SEM_SIGNAL",9:"SEM_WAIT",10:"MEM_ALLOC",11:"MEM_FREE",
              12:"BATCH_MEM_OP",13:"CONDITIONAL"}

class KNP(ctypes.Structure):
    _fields_ = [("func", ctypes.c_void_p),
                ("gridDimX", ctypes.c_uint), ("gridDimY", ctypes.c_uint), ("gridDimZ", ctypes.c_uint),
                ("blockDimX", ctypes.c_uint), ("blockDimY", ctypes.c_uint), ("blockDimZ", ctypes.c_uint),
                ("sharedMemBytes", ctypes.c_uint),
                ("kernelParams", ctypes.POINTER(ctypes.c_void_p)),
                ("extra", ctypes.POINTER(ctypes.c_void_p)),
                ("kern", ctypes.c_void_p), ("ctx", ctypes.c_void_p)]

def _chk(r, what):
    if r != 0:
        s = ctypes.c_char_p()
        _cu.cuGetErrorName(r, ctypes.byref(s))
        raise RuntimeError(f"{what} -> {r} {s.value}")

def func_name(f):
    p = ctypes.c_char_p()
    r = _cu.cuFuncGetName(ctypes.byref(p), ctypes.c_void_p(f))
    return p.value.decode() if r == 0 else f"<err{r}>"

def func_module(f):
    m = ctypes.c_void_p()
    r = _cu.cuFuncGetModule(ctypes.byref(m), ctypes.c_void_p(f))
    return m.value if r == 0 else None

def func_params(f):
    """[(offset,size)] via cuFuncGetParamInfo until it errors out."""
    out = []
    for i in range(256):
        off = ctypes.c_size_t(); sz = ctypes.c_size_t()
        r = _cu.cuFuncGetParamInfo(ctypes.c_void_p(f), ctypes.c_size_t(i),
                                   ctypes.byref(off), ctypes.byref(sz))
        if r != 0:
            break
        out.append((off.value, sz.value))
    return out

_ATTRS = {0:"MAX_THREADS_PER_BLOCK",1:"SHARED_SIZE_BYTES",2:"CONST_SIZE_BYTES",
          3:"LOCAL_SIZE_BYTES",4:"NUM_REGS",5:"PTX_VERSION",6:"BINARY_VERSION"}
def func_attrs(f):
    d = {}
    for k, n in _ATTRS.items():
        v = ctypes.c_int()
        if _cu.cuFuncGetAttribute(ctypes.byref(v), ctypes.c_int(k), ctypes.c_void_p(f)) == 0:
            d[n] = v.value
    return d

def graph_nodes(raw_graph):
    n = ctypes.c_size_t(0)
    _chk(_cu.cuGraphGetNodes(ctypes.c_void_p(raw_graph), None, ctypes.byref(n)), "cuGraphGetNodes(count)")
    cnt = n.value
    arr = (ctypes.c_void_p * cnt)()
    _chk(_cu.cuGraphGetNodes(ctypes.c_void_p(raw_graph), arr, ctypes.byref(n)), "cuGraphGetNodes")
    out = []
    for i in range(cnt):
        t = ctypes.c_int()
        _chk(_cu.cuGraphNodeGetType(arr[i], ctypes.byref(t)), "cuGraphNodeGetType")
        rec = {"node": arr[i], "type": NODE_TYPES.get(t.value, str(t.value))}
        if t.value == 0:
            p = KNP()
            _chk(_cu.cuGraphKernelNodeGetParams_v2(arr[i], ctypes.byref(p)), "KernelNodeGetParams")
            rec.update(func=p.func, grid=(p.gridDimX,p.gridDimY,p.gridDimZ),
                       block=(p.blockDimX,p.blockDimY,p.blockDimZ), smem=p.sharedMemBytes,
                       kernelParams=bool(p.kernelParams), extra=bool(p.extra),
                       name=func_name(p.func), module=func_module(p.func))
            # raw parameter bytes
            pi = func_params(p.func)
            rec["paraminfo"] = pi
            blobs = []
            if p.kernelParams:
                for j,(off,sz) in enumerate(pi):
                    try:
                        ptr = p.kernelParams[j]
                        blobs.append(bytes((ctypes.c_ubyte*sz).from_address(ptr)) if ptr else None)
                    except Exception:
                        blobs.append(None)
            rec["parambytes"] = blobs
            # cuBLAS/cuDNN launch with `extra` = CU_LAUNCH_PARAM_BUFFER_POINTER form
            rec["extrablob"] = None
            if p.extra:
                buf = None; blen = None
                k = 0
                while k < 16:
                    tag = p.extra[k]
                    if tag is None or tag == 0:
                        break
                    if tag == 1:
                        buf = p.extra[k+1]
                    elif tag == 2:
                        szp = p.extra[k+1]
                        blen = ctypes.c_size_t.from_address(szp).value if szp else None
                    k += 2
                if buf is not None:
                    n_ = blen if blen is not None else (pi[0][0] + pi[0][1] if pi else 0)
                    try:
                        rec["extrablob"] = bytes((ctypes.c_ubyte*n_).from_address(buf))
                    except Exception as e:
                        rec["extrablob"] = None
        out.append(rec)
    return out


def capture(fn, warmup=3):
    """Warm up on a side stream, then capture fn() into a fresh graph. Returns node list."""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        fn()
    nodes = graph_nodes(g.raw_cuda_graph())
    torch.cuda.synchronize()
    return nodes, g

def short(n):
    if n["type"] != "KERNEL":
        return n["type"]
    return f'{n["name"][:78]} grid={n["grid"]} blk={n["block"]} smem={n["smem"]}'
