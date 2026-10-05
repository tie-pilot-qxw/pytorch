#!/usr/bin/env python3
"""Q4: does the same M pick a different kernel at a different time / in a different order / in a different process?"""
import gc, os, sys, json, random
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

DT = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("WF_DT", "bf16")]
K = N = 512
MMAX = 4096
x_full = torch.randn(MMAX, K, device="cuda", dtype=DT)
w = torch.randn(K, N, device="cuda", dtype=DT)
b = torch.randn(N, device="cuda", dtype=DT)
y_full = torch.zeros(MMAX, N, device="cuda", dtype=DT)

def kern_for(M):
    def run():
        return torch.addmm(b, x_full[:M], w, out=y_full[:M])
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): run()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): run()
    torch.cuda.synchronize()
    raw = g.raw_cuda_graph()
    nn = ck(cr.cudaGraphGetNodes(raw))[1]
    nodes = ck(cr.cudaGraphGetNodes(raw, nn))[0]
    out = []
    for nd in nodes:
        t = int(getattr(ck(cr.cudaGraphNodeGetType(nd)), "value", 0))
        if t != 0: out.append(f"<t{t}>"); continue
        dp = ck(cd.cuGraphKernelNodeGetParams(nd))
        out.append(f"{ck(cd.cuFuncGetName(dp.func)).decode()}@0x{int(dp.func):x}")
    del g; gc.collect()
    return tuple(out)

Ms = [16, 104, 236, 512, 947, 1024, 2048, 4096]
r1 = {M: kern_for(M) for M in Ms}                    # ascending, first pass
r2 = {M: kern_for(M) for M in Ms}                    # ascending, second pass (same process)
sh = Ms[:]; random.Random(7).shuffle(sh)
r3 = {M: kern_for(M) for M in sh}                    # shuffled
r4 = {M: kern_for(M) for M in reversed(Ms)}          # descending
print(json.dumps({"tag": os.environ.get("WF_TAG", "run"),
                  "ws": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<default>"),
                  "r1": {str(k): v for k, v in r1.items()},
                  "same_2nd": all(r1[M] == r2[M] for M in Ms),
                  "same_shuffled": all(r1[M] == r3[M] for M in Ms),
                  "same_desc": all(r1[M] == r4[M] for M in Ms),
                  "diffs": {str(M): [r1[M], r2[M], r3[M], r4[M]]
                            for M in Ms if not (r1[M] == r2[M] == r3[M] == r4[M])}}))
