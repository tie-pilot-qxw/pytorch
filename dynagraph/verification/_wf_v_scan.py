#!/usr/bin/env python3
"""Independent cross-check: put N GEMMs with different M into one capture and use
cudaStreamGetCaptureInfo to incrementally identify "which nodes this call produced". Prints each M's
node count / real kernel name / grid / smem / cluster. The method differs from the "capture each M
separately" approach of the agent being verified, so the two can corroborate each other."""
import gc, json, os, sys
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

which = os.environ.get("DT", "fp32")
op    = os.environ.get("OP", "addmm")
DT = {"fp32": torch.float32, "bf16": torch.bfloat16}[which]
K = N = int(os.environ.get("KN", "512"))
MS = [int(v) for v in os.environ["MS"].split(",")]
MMAX = max(MS)

torch.manual_seed(0)
x_full = torch.randn(MMAX, K, device="cuda", dtype=DT)
w = torch.randn(K, N, device="cuda", dtype=DT)
b = torch.randn(N, device="cuda", dtype=DT)
y_full = torch.zeros(MMAX, N, device="cuda", dtype=DT)

def run(M):
    if op == "addmm":
        return torch.addmm(b, x_full[:M], w, out=y_full[:M])
    return torch.mm(x_full[:M], w, out=y_full[:M])

# warmup (so heuristics / JIT happen outside the capture)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for M in MS:
        for _ in range(2): run(M)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

def node_info(nd):
    t = int(getattr(ck(cr.cudaGraphNodeGetType(nd)), "value", 0))
    if t != 0:
        return dict(type=t, name="<type%d>" % t)
    dp = ck(cd.cuGraphKernelNodeGetParams(nd))
    try:
        cl = ck(cd.cuGraphKernelNodeGetAttribute(
            nd, cd.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION))
        cluster = (int(cl.clusterDim.x), int(cl.clusterDim.y), int(cl.clusterDim.z))
    except RuntimeError as e:
        cluster = "err:" + str(e)[:30]
    npar = 0
    pinfo = []
    while npar < 64:
        try:
            o, sz = ck(cd.cuFuncGetParamInfo(dp.func, npar))
        except RuntimeError:
            break
        pinfo.append((int(o), int(sz))); npar += 1
    return dict(type=0, name=ck(cd.cuFuncGetName(dp.func)).decode(),
                func=int(dp.func),
                grid=(int(dp.gridDimX), int(dp.gridDimY), int(dp.gridDimZ)),
                block=(int(dp.blockDimX), int(dp.blockDimY), int(dp.blockDimZ)),
                smem=int(dp.sharedMemBytes), cluster=cluster,
                kp=int(dp.kernelParams), extra=int(dp.extra), pinfo=pinfo)

g = torch.cuda.CUDAGraph(keep_graph=True)
seen = set()
per_M = {}
with torch.cuda.graph(g):
    st = torch.cuda.current_stream().cuda_stream
    for M in MS:
        run(M)
        info = cr.cudaStreamGetCaptureInfo(st)
        ck(info)
        status, cid, graph_cap = info[1], info[2], info[3]
        cnt = ck(cr.cudaGraphGetNodes(graph_cap))[1]
        nds = ck(cr.cudaGraphGetNodes(graph_cap, cnt))[0]
        new = [nd for nd in nds if int(nd) not in seen]
        for nd in nds: seen.add(int(nd))
        per_M[M] = [int(x) for x in new]
torch.cuda.synchronize()

raw = g.raw_cuda_graph()
cnt = ck(cr.cudaGraphGetNodes(raw))[1]
nds = ck(cr.cudaGraphGetNodes(raw, cnt))[0]
by_h = {int(nd): node_info(nd) for nd in nds}
print(f"# dtype={which} op={op} K=N={K}  total nodes={cnt}  num M={len(MS)}")

rows = []
prev = None
for M in MS:
    infos = [by_h[h] for h in per_M[M]]
    infos.sort(key=lambda d: d["name"])
    sig = tuple((d["name"], d.get("cluster"), d.get("smem")) for d in infos)
    rows.append(dict(M=M, n=len(infos), infos=infos))
    if sig != prev:
        print(f"M>={M:<6} nodes={len(infos)} " + " | ".join(
            f"{d['name'][:72]} grid={d.get('grid')} smem={d.get('smem')} cluster={d.get('cluster')} pinfo={d.get('pinfo')}"
            for d in infos))
        prev = sig
names = {}
clusters = {}
for r in rows:
    for d in r["infos"]:
        names.setdefault(d["name"], []).append(r["M"])
        if d["type"] == 0:
            clusters.setdefault(str(d["cluster"]), []).append(r["M"])
print(f"\ndistinct kernel names = {len(names)}   distinct func pointers = "
      f"{len({d['func'] for r in rows for d in r['infos'] if d['type']==0})}")
for n, ms in sorted(names.items(), key=lambda kv: -len(kv[1])):
    print(f"   {len(ms):>4} M  [{min(ms)}..{max(ms)}]  {n[:88]}")
print("cluster groups:")
for c, ms in clusters.items():
    print(f"   {c}: {len(ms)} M values, [{min(ms)}..{max(ms)}]")
nodecnt = {}
for r in rows: nodecnt.setdefault(r["n"], []).append(r["M"])
print("node count distribution:", {k: (len(v), min(v), max(v)) for k, v in nodecnt.items()})
if os.environ.get("OUT"):
    json.dump([dict(M=r["M"], n=r["n"],
                    infos=[{k: v for k, v in d.items() if k != "func"} for d in r["infos"]])
               for r in rows], open(os.environ["OUT"], "w"))
