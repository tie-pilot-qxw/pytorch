#!/usr/bin/env python3
"""(C) How far can M change before the kernel switches: fine-grained M sweep, recording kernel name/func/node count."""
import gc, sys, json
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

K = N = 512
MMAX = 4096
which = sys.argv[1] if len(sys.argv) > 1 else "fp32"
DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[which]
op = sys.argv[2] if len(sys.argv) > 2 else "addmm"

x_full = torch.randn(MMAX, K, device="cuda", dtype=DT)
w = torch.randn(K, N, device="cuda", dtype=DT)
b = torch.randn(N, device="cuda", dtype=DT)
y_full = torch.zeros(MMAX, N, device="cuda", dtype=DT)

def run(M):
    if op == "addmm":
        return torch.addmm(b, x_full[:M], w, out=y_full[:M])
    return torch.mm(x_full[:M], w, out=y_full[:M])

def probe(M):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): run(M)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        run(M)
    torch.cuda.synchronize()
    raw = g.raw_cuda_graph()
    nn = ck(cr.cudaGraphGetNodes(raw))[1]
    nodes = ck(cr.cudaGraphGetNodes(raw, nn))[0]
    out = []
    for nd in nodes:
        t = int(getattr(ck(cr.cudaGraphNodeGetType(nd)), "value", 0))
        if t != 0:
            out.append(("<type%d>" % t, 0, None)); continue
        dp = ck(cd.cuGraphKernelNodeGetParams(nd))
        nm = ck(cd.cuFuncGetName(dp.func)).decode()
        out.append((nm, int(dp.func), (dp.gridDimX, dp.gridDimY, dp.gridDimZ)))
    del g; gc.collect()
    return out

Ms = sorted(set(list(range(1, 33)) + list(range(32, 273, 8)) +
                list(range(288, 1057, 32)) + [947, 1000, 1023, 1025] +
                list(range(1152, 4097, 256))))
prev = None
rows = []
for M in Ms:
    r = probe(M)
    sig = tuple(x[0] for x in r)
    rows.append((M, r))
    if sig != prev:
        print(f"M>={M:<6} nodes={len(r)}  " + " | ".join(
            f"{n.replace('_ZN7cutlass7Kernel2I','').replace('EEvNT_6ParamsE','')[:74]}"
            f" grid={gd}" for n, f, gd in r))
        prev = sig
print(f"\nScanned {len(Ms)} M values; the segments with distinct kernel signatures are listed above.")
names = {}
for M, r in rows:
    for n, f, gd in r:
        names.setdefault(n, []).append(M)
print(f"{len(names)} distinct kernel names; "
      f"{len({f for _,r in rows for _,f,_ in r})} unique func pointers")
for n, ms in names.items():
    print(f"   {len(ms):>4} M values -> {n[:90]}")
