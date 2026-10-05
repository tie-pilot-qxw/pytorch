#!/usr/bin/env python3
"""(A) How many nodes one torch.addmm produces in a CUDA Graph / of what type / which kernel."""
import ctypes, sys
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck


def capture_addmm(M, K=512, N=512, dtype=torch.float32):
    dev = "cuda"
    x = torch.randn(M, K, device=dev, dtype=dtype)
    w = torch.randn(K, N, device=dev, dtype=dtype)
    b = torch.randn(N, device=dev, dtype=dtype)
    # warmup on side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            y = torch.addmm(b, x, w)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        y = torch.addmm(b, x, w)
    torch.cuda.synchronize()
    return g, (x, w, b, y)


def node_report(g):
    raw = g.raw_cuda_graph()
    n = ck(cr.cudaGraphGetNodes(raw))[1]
    nodes = ck(cr.cudaGraphGetNodes(raw, n))[0]
    out = []
    for i, nd in enumerate(nodes):
        t = ck(cr.cudaGraphNodeGetType(nd))
        tv = int(getattr(t, "value", t))
        tname = getattr(cr.cudaGraphNodeType(tv), "name", str(tv))
        rec = {"idx": i, "node": int(nd), "type": f"{tname}({tv})"}
        if tv == 0:
            try:
                p = ck(cr.cuGraphKernelNodeGetParams(nd)) if hasattr(cr, "cuGraphKernelNodeGetParams") else None
            except Exception:
                p = None
            dp = ck(cd.cuGraphKernelNodeGetParams(nd))
            rec["drv_params"] = dp
            func = dp.func
            rec["func_int"] = int(func)
            try:
                nm = ck(cd.cuFuncGetName(func))
                rec["name"] = nm.decode() if isinstance(nm, bytes) else str(nm)
            except Exception as e:
                rec["name"] = f"<err {e}>"
            rec["grid"] = (dp.gridDimX, dp.gridDimY, dp.gridDimZ)
            rec["block"] = (dp.blockDimX, dp.blockDimY, dp.blockDimZ)
            rec["smem"] = dp.sharedMemBytes
            rec["kernelParams_repr"] = repr(dp.kernelParams)[:200]
            rec["extra_repr"] = repr(dp.extra)[:200]
        out.append(rec)
    return out


def main():
    torch.cuda.init()
    print("torch", torch.__version__)
    for M in (64,):
        g, _t = capture_addmm(M)
        print(f"\n=== addmm M={M} fp32 ===")
        for rec in node_report(g):
            print(f"  node[{rec['idx']}] type={rec['type']}")
            if "name" in rec:
                print(f"      name={rec['name']}")
                print(f"      func=0x{rec['func_int']:x} grid={rec['grid']} block={rec['block']} smem={rec['smem']}")
                print(f"      kernelParams={rec['kernelParams_repr']}")
                print(f"      extra={rec['extra_repr']}")
        del g
    return 0


if __name__ == "__main__":
    sys.exit(main())
