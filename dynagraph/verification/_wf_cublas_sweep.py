#!/usr/bin/env python3
"""(B)(C) Sweep M/dtype/op and record each cuBLAS node's kernel name, func pointer,
grid/block/smem and full kernel parameter bytes, for diffing."""
import ctypes, json, sys, os
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

K = N = 512
KEEP = []          # keep graphs / tensors from being freed


def param_info(func):
    infos = []
    i = 0
    while True:
        try:
            off, size = ck(cd.cuFuncGetParamInfo(func, i))
        except RuntimeError:
            break
        infos.append((int(off), int(size)))
        i += 1
        if i > 64:
            break
    return infos


def read_params(dp, infos):
    kp = int(dp.kernelParams)
    if kp == 0:
        return None
    arr = (ctypes.c_void_p * len(infos)).from_address(kp)
    blobs = []
    for idx, (off, size) in enumerate(infos):
        ptr = arr[idx]
        if not ptr:
            blobs.append(None)
            continue
        blobs.append(bytes((ctypes.c_ubyte * size).from_address(ptr)))
    return blobs


def capture(op, M, dtype):
    dev = "cuda"
    x = torch.randn(M, K, device=dev, dtype=dtype)
    w = torch.randn(K, N, device=dev, dtype=dtype)
    b = torch.randn(N, device=dev, dtype=dtype)
    fn = (lambda: torch.addmm(b, x, w)) if op == "addmm" else (lambda: torch.mm(x, w))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        y = fn()
    torch.cuda.synchronize()
    KEEP.append((g, x, w, b, y))
    return g


def describe(g):
    raw = g.raw_cuda_graph()
    n = ck(cr.cudaGraphGetNodes(raw))[1]
    nodes = ck(cr.cudaGraphGetNodes(raw, n))[0]
    recs = []
    for nd in nodes:
        t = int(getattr(ck(cr.cudaGraphNodeGetType(nd)), "value", 0))
        r = {"type": t}
        if t == 0:
            dp = ck(cd.cuGraphKernelNodeGetParams(nd))
            func = dp.func
            nm = ck(cd.cuFuncGetName(func))
            r["name"] = nm.decode() if isinstance(nm, bytes) else str(nm)
            r["func"] = int(func)
            r["grid"] = (dp.gridDimX, dp.gridDimY, dp.gridDimZ)
            r["block"] = (dp.blockDimX, dp.blockDimY, dp.blockDimZ)
            r["smem"] = int(dp.sharedMemBytes)
            r["pinfo"] = param_info(func)
            r["extra"] = int(dp.extra)
            blobs = read_params(dp, r["pinfo"])
            r["params_hex"] = [None if b is None else b.hex() for b in (blobs or [])]
        recs.append(r)
    return recs


def main():
    torch.cuda.init()
    out = {}
    Ms = [16, 64, 128, 129, 256, 512, 947, 1024, 4096]
    for op in ("addmm", "mm"):
        for dtname, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            for M in Ms:
                g = capture(op, M, dt)
                key = f"{op}|{dtname}|M{M}"
                out[key] = describe(g)
                ks = [r for r in out[key] if r["type"] == 0]
                print(f"{key:<24} nodes={len(out[key])} types={[r['type'] for r in out[key]]}")
                for r in ks:
                    print(f"    func=0x{r['func']:x} grid={r['grid']} block={r['block']} "
                          f"smem={r['smem']} nparams={len(r['pinfo'])} pinfo={r['pinfo']}")
                    print(f"    {r['name']}")
    dg_out = os.environ.get("DG_OUT", "/tmp/dynagraph_out")
    os.makedirs(dg_out, exist_ok=True)
    with open(os.path.join(dg_out, "_wf_cublas_sweep.json"), "w") as f:
        json.dump(out, f)
    print("\nwrote _wf_cublas_sweep.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
