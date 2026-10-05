#!/usr/bin/env python3
"""Can we swap just **one** extern node in a whole large graph, and across clusters?
Method: capture the cuBLAS call as a small 1-node graph, insert it into an outer graph as a child-graph node,
and after the outer graph is instantiated, replace that child graph wholesale with cuGraphExecChildGraphNodeSetParams."""
import ctypes, os, torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

which = os.environ.get("DT", "bf16")
DT = {"fp32": torch.float32, "bf16": torch.bfloat16}[which]
SENT = 12288.0
K = N = 512
BASE = int(os.environ.get("BASE", "947"))
TARGETS = [int(v) for v in os.environ.get("TARGETS", "16,104,1024,4096,947").split(",")]
FIXZERO = bool(os.environ.get("FIXZERO"))
MMAX = max(TARGETS + [BASE])
torch.manual_seed(0)
x = torch.randn(MMAX, K, device="cuda", dtype=DT)
w = torch.randn(K, N, device="cuda", dtype=DT)
b = torch.randn(N, device="cuda", dtype=DT)
y = torch.empty(MMAX, N, device="cuda", dtype=DT)
HOLD = []

OP = os.environ.get("OP", "addmm")
def run(M):
    if OP == "addmm": return torch.addmm(b, x[:M], w, out=y[:M])
    return torch.mm(x[:M], w, out=y[:M])

def cap(M):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): run(M)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): run(M)
    torch.cuda.synchronize(); HOLD.append(g)
    raw = g.raw_cuda_graph()
    c = ck(cr.cudaGraphGetNodes(raw))[1]
    nds = ck(cr.cudaGraphGetNodes(raw, c))[0]
    info = []
    for nd in nds:
        dp = ck(cd.cuGraphKernelNodeGetParams(nd))
        cl = ck(cd.cuGraphKernelNodeGetAttribute(nd, cd.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION))
        cur = (int(cl.clusterDim.x), int(cl.clusterDim.y), int(cl.clusterDim.z))
        if FIXZERO and cur == (0, 0, 0):
            v = cd.CUkernelNodeAttrValue(); v.clusterDim.x = v.clusterDim.y = v.clusterDim.z = 1
            ck(cd.cuGraphKernelNodeSetAttribute(nd, cd.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, v))
            cur = (1, 1, 1)
        info.append((ck(cd.cuFuncGetName(dp.func)).decode(), cur, int(dp.sharedMemBytes)))
    return g, raw, info

tg = {M: cap(M) for M in TARGETS}
gb, rawb, infob = cap(BASE)

outer = ck(cd.cuGraphCreate(0))
child = ck(cd.cuGraphAddChildGraphNode(outer, [], 0, rawb))
exec_ = ck(cd.cuGraphInstantiate(outer, 0))
st = torch.cuda.current_stream().cuda_stream
print(f"DT={which} outer graph = 1 child-graph node, host M={BASE} {infob[0][0][:52]} cluster={infob[0][1]}")
for M in TARGETS:
    _, rawt, infot = tg[M]
    y.fill_(SENT)
    try:
        ck(cd.cuGraphExecChildGraphNodeSetParams(exec_, child, rawt))
    except RuntimeError as e:
        print(f"M={M:<6} {infot[0][1]} SetChildGraphParams failed: {str(e)[:70]}"); continue
    ck(cd.cuGraphLaunch(exec_, st)); torch.cuda.synchronize()
    ref = (torch.addmm(b, x[:M], w) if OP == "addmm" else torch.mm(x[:M], w)).float()
    err = (y[:M].float() - ref).abs().max().item()
    nw = int((y[:M] == SENT).sum().item())
    tail = 0 if M == MMAX else int((y[M:] != SENT).sum().item())
    ok = "OK" if (err == 0 and nw == 0 and tail == 0) else "**BAD**"
    print(f"M={M:<6} {infot[0][0][:50]:<50} cluster={str(infot[0][1]):<10} smem={infot[0][2]:<7} {ok} "
          f"max|err|={err:.4g} unwritten={nw} oob={tail}")
