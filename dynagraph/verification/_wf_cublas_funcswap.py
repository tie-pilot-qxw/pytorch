#!/usr/bin/env python3
"""(C) Key experiment: can cuGraphExecKernelNodeSetParams swap the func as well?
First capture once per M on the same set of buffers, collecting (func, grid, smem, param blob),
then install each whole set into **one** already-instantiated graph and check the numbers one by one."""
import ctypes, gc, math, os, struct, sys

HOST_SMEM = {"v": 0}
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

DT = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[1] if len(sys.argv) > 1 else "bf16"]
K = N = 512
MMAX = 4096
torch.manual_seed(0)
x_full = torch.randn(MMAX, K, device="cuda", dtype=DT)
w = torch.randn(K, N, device="cuda", dtype=DT)
b = torch.randn(N, device="cuda", dtype=DT)
y_full = torch.zeros(MMAX, N, device="cuda", dtype=DT)
GRAPHS = []

def run(M):
    return torch.addmm(b, x_full[:M], w, out=y_full[:M])

def read_args(dp):
    kp, ex = int(dp.kernelParams), int(dp.extra)
    npar = 0
    while True:
        try: ck(cd.cuFuncGetParamInfo(dp.func, npar))
        except RuntimeError: break
        npar += 1
        if npar > 64: break
    if ex:
        ents = (ctypes.c_void_p * 8).from_address(ex)
        vals = [ents[i] or 0 for i in range(8)]
        ptr = size = None
        i = 0
        while i < 8 and vals[i]:
            if vals[i] == 1: ptr = vals[i+1]; i += 2
            elif vals[i] == 2: size = ctypes.c_size_t.from_address(vals[i+1]).value; i += 2
            else: i += 1
        return ("extra", [bytes((ctypes.c_ubyte*size).from_address(ptr))])
    if kp:
        arr = (ctypes.c_void_p * npar).from_address(kp)
        blobs = []
        for i in range(npar):
            _, sz = ck(cd.cuFuncGetParamInfo(dp.func, i))
            blobs.append(bytes((ctypes.c_ubyte*int(sz)).from_address(arr[i])))
        return ("kernelParams", blobs)
    return ("none", [])

def harvest(M):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): run(M)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): run(M)
    torch.cuda.synchronize()
    GRAPHS.append(g)
    raw = g.raw_cuda_graph()
    nn = ck(cr.cudaGraphGetNodes(raw))[1]
    nodes = ck(cr.cudaGraphGetNodes(raw, nn))[0]
    ks = []
    for nd in nodes:
        t = int(getattr(ck(cr.cudaGraphNodeGetType(nd)), "value", 0))
        if t != 0: ks.append(None); continue
        dp = ck(cd.cuGraphKernelNodeGetParams(nd))
        mode, blobs = read_args(dp)
        try:
            cl = ck(cd.cuGraphKernelNodeGetAttribute(nd, cd.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION))
            cluster = (cl.clusterDim.x, cl.clusterDim.y, cl.clusterDim.z)
        except RuntimeError as e:
            cluster = f"<err {str(e)[:40]}>"
        ks.append(dict(M=M, name=ck(cd.cuFuncGetName(dp.func)).decode(), func=dp.func, dp=dp, cluster=cluster,
                       funci=int(dp.func), kern=int(dp.kern), ctx=int(dp.ctx),
                       grid=(dp.gridDimX, dp.gridDimY, dp.gridDimZ),
                       block=(dp.blockDimX, dp.blockDimY, dp.blockDimZ),
                       smem=int(dp.sharedMemBytes), mode=mode, blobs=blobs))
    return g, nodes, ks

KEEP = []
def make_params(rec):
    p = cd.CUDA_KERNEL_NODE_PARAMS()
    p.func = rec["func"]
    p.kern = rec["dp"].kern
    p.ctx = rec["dp"].ctx
    p.gridDimX, p.gridDimY, p.gridDimZ = rec["grid"]
    p.blockDimX, p.blockDimY, p.blockDimZ = rec["block"]
    # WF_SMEM=keep: swap the func but pin smem to the host graph's value.
    # Separates two confounded variables, "smem changed" and "func changed" (in _wf_swap2.log,
    # not included in this repo, all three fp32 func swaps blew up, yet the bf16 func swap with
    # smem up by 16KB was bit-exact).
    p.sharedMemBytes = HOST_SMEM["v"] if os.environ.get("WF_SMEM") == "keep" else rec["smem"]
    if rec["mode"] == "extra":
        buf = bytearray(rec["blobs"][0]); sz = len(buf)
        cb = (ctypes.c_ubyte*sz).from_buffer(buf); cs = ctypes.c_size_t(sz)
        ea = (ctypes.c_void_p*5)(ctypes.c_void_p(1), ctypes.cast(cb, ctypes.c_void_p),
                                 ctypes.c_void_p(2), ctypes.cast(ctypes.pointer(cs), ctypes.c_void_p),
                                 ctypes.c_void_p(0))
        KEEP.append((buf, cb, cs, ea))
        p.kernelParams = 0; p.extra = ctypes.addressof(ea)
    else:
        bufs = [bytearray(x) for x in rec["blobs"]]
        cbs = [(ctypes.c_ubyte*len(x)).from_buffer(x) for x in bufs]
        arr = (ctypes.c_void_p*len(cbs))(*[ctypes.cast(c, ctypes.c_void_p) for c in cbs])
        KEEP.append((bufs, cbs, arr))
        p.kernelParams = ctypes.addressof(arr); p.extra = 0
    return p

Ms = [int(v) for v in os.environ.get('WF_MS', '16,104,236,248,512,947,1024,2048,4096').split(',')]
table = {}
print(f"dtype={DT}  harvesting {Ms}")
for M in Ms:
    _g, _n, ks = harvest(M)
    table[M] = ks
    for r in ks:
        if r: print(f"  M={M:<5} {r['name'][:64]:<64} func=0x{r['funci']:x} cluster={r['cluster']} grid={r['grid']} smem={r['smem']} mode={r['mode']} blob={[len(x) for x in r['blobs']]}")
        else: print(f"  M={M:<5} <non-kernel node>")

BASE = int(sys.argv[2]) if len(sys.argv) > 2 else 947
gb, nodes_b, ks_b = harvest(BASE)
gb.replay(); torch.cuda.synchronize()
gexec = gb.raw_cuda_graph_exec()
HOST_SMEM["v"] = ks_b[0]["smem"]
print(f"\nhost graph: M={BASE}, {len(nodes_b)} nodes, kernel={ks_b[0]['name'][:60]} smem={ks_b[0]['smem']} cluster={ks_b[0]['cluster']} mode={ks_b[0]['mode']}")
print(f"WF_SMEM={os.environ.get('WF_SMEM', 'target')}")
print("\ntarget  swapped cluster     smem     mode          SetParams         numerics")
for M in Ms:
    ks = table[M]
    if len(ks) != len(ks_b) or any(r is None for r in ks):
        print(f"{M:<7} node count/types differ ({len(ks)} vs {len(ks_b)}), skipped"); continue
    swapped = ks[0]["funci"] != ks_b[0]["funci"]
    try:
        for nd, rec in zip(nodes_b, ks):
            ck(cd.cuGraphExecKernelNodeSetParams(gexec, nd, make_params(rec)))
        y_full.zero_()
        gb.replay(); torch.cuda.synchronize()
        ref = torch.addmm(b, x_full[:M], w).float()
        err = (y_full[:M].float() - ref).abs().max().item()
        rel = err / max(ref.abs().max().item(), 1e-9)
        tail = y_full[M:].abs().max().item() if M < MMAX else 0.0
        print(f"{M:<7} {str(swapped):<8}{str(ks[0]['cluster']):<12} {ks[0]['smem']:<9}{ks[0]['mode']:<14}OK                max|err|={err:<10.4g} rel={rel:<10.3g} oob={tail:.3g}"
              + ("" if rel < 5e-2 else "   **WRONG**"))
    except RuntimeError as e:
        print(f"{M:<7} {str(swapped):<8}{str(ks[0]['cluster']):<12} {ks[0]['smem']:<9}{ks[0]['mode']:<14}failed: {str(e)[:70]}")
