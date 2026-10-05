#!/usr/bin/env python3
"""Independent cross-check of claims 14/15/16: can an already-instantiated graph be switched to another M's cuBLAS kernel?
Two methods compared:
  METHOD=setparams  -- cuGraphExecKernelNodeSetParams(func+param blob+grid+smem)
  METHOD=execupdate -- cudaGraphExecUpdate(gexec, template graph captured on the spot for the target M)
Numerics are checked with a sentinel: catches both out-of-bounds writes and "should have written but didn't"."""
import ctypes, gc, os, statistics, sys, time
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

which = os.environ.get("DT", "bf16")
DT = {"fp32": torch.float32, "bf16": torch.bfloat16}[which]
SENT = 12288.0
K = N = 512
BASE = int(os.environ.get("BASE", "947"))
TARGETS = [int(v) for v in os.environ.get("TARGETS", "16,104,236,512,947").split(",")]
METHOD = os.environ.get("METHOD", "setparams")
OP = os.environ.get("OP", "addmm")
MMAX = max(TARGETS + [BASE])
torch.manual_seed(0)
x_full = torch.randn(MMAX, K, device="cuda", dtype=DT)
w = torch.randn(K, N, device="cuda", dtype=DT)
b = torch.randn(N, device="cuda", dtype=DT)
y_full = torch.empty(MMAX, N, device="cuda", dtype=DT)
HOLD = []

def run(M):
    if OP == "addmm": return torch.addmm(b, x_full[:M], w, out=y_full[:M])
    return torch.mm(x_full[:M], w, out=y_full[:M])

def blob_of(dp):
    kp, ex = int(dp.kernelParams), int(dp.extra)
    if ex:
        ents = (ctypes.c_void_p * 8).from_address(ex)
        vals = [ents[i] or 0 for i in range(8)]
        ptr = size = None; i = 0
        while i < 8 and vals[i] not in (0, None):
            if vals[i] == 1: ptr = vals[i + 1]; i += 2
            elif vals[i] == 2: size = ctypes.c_size_t.from_address(vals[i + 1]).value; i += 2
            else: break
        return "extra", [bytes((ctypes.c_ubyte * size).from_address(ptr))]
    n = 0
    while n < 64:
        try: ck(cd.cuFuncGetParamInfo(dp.func, n))
        except RuntimeError: break
        n += 1
    arr = (ctypes.c_void_p * n).from_address(kp)
    out = []
    for i in range(n):
        _, sz = ck(cd.cuFuncGetParamInfo(dp.func, i))
        out.append(bytes((ctypes.c_ubyte * int(sz)).from_address(arr[i])))
    return "kernelParams", out

def harvest(M):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): run(M)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): run(M)
    torch.cuda.synchronize()
    HOLD.append(g)
    raw = g.raw_cuda_graph()
    cnt = ck(cr.cudaGraphGetNodes(raw))[1]
    nds = ck(cr.cudaGraphGetNodes(raw, cnt))[0]
    recs = []
    for nd in nds:
        if os.environ.get("FIXZERO"):
            cl0 = ck(cd.cuGraphKernelNodeGetAttribute(nd, cd.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION))
            if (cl0.clusterDim.x, cl0.clusterDim.y, cl0.clusterDim.z) == (0, 0, 0):
                v = cd.CUkernelNodeAttrValue()
                v.clusterDim.x, v.clusterDim.y, v.clusterDim.z = 1, 1, 1
                try:
                    ck(cd.cuGraphKernelNodeSetAttribute(nd, cd.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, v))
                    print(f"   [FIXZERO] M={M} template node cluster set to (1,1,1)")
                except RuntimeError as e:
                    print(f"   [FIXZERO] M={M} set failed {str(e)[:60]}")
        dp = ck(cd.cuGraphKernelNodeGetParams(nd))
        cl = ck(cd.cuGraphKernelNodeGetAttribute(nd, cd.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION))
        mode, blobs = blob_of(dp)
        recs.append(dict(M=M, name=ck(cd.cuFuncGetName(dp.func)).decode(), dp=dp,
                         funci=int(dp.func), smem=int(dp.sharedMemBytes),
                         grid=(int(dp.gridDimX), int(dp.gridDimY), int(dp.gridDimZ)),
                         block=(int(dp.blockDimX), int(dp.blockDimY), int(dp.blockDimZ)),
                         cluster=(int(cl.clusterDim.x), int(cl.clusterDim.y), int(cl.clusterDim.z)),
                         mode=mode, blobs=blobs))
    return g, nds, recs

KEEP = []
def mk(rec):
    p = cd.CUDA_KERNEL_NODE_PARAMS()
    dp = rec["dp"]
    p.func = dp.func; p.kern = dp.kern; p.ctx = dp.ctx
    p.gridDimX, p.gridDimY, p.gridDimZ = rec["grid"]
    p.blockDimX, p.blockDimY, p.blockDimZ = rec["block"]
    p.sharedMemBytes = rec["smem"]
    if rec["mode"] == "extra":
        buf = bytearray(rec["blobs"][0]); sz = len(buf)
        cb = (ctypes.c_ubyte * sz).from_buffer(buf); cs = ctypes.c_size_t(sz)
        ea = (ctypes.c_void_p * 5)(ctypes.c_void_p(2), ctypes.cast(ctypes.pointer(cs), ctypes.c_void_p),
                                   ctypes.c_void_p(1), ctypes.cast(cb, ctypes.c_void_p),
                                   ctypes.c_void_p(0))
        KEEP.append((buf, cb, cs, ea))
        p.kernelParams = 0; p.extra = ctypes.addressof(ea)
    else:
        bufs = [bytearray(x) for x in rec["blobs"]]
        cbs = [(ctypes.c_ubyte * len(x)).from_buffer(x) for x in bufs]
        arr = (ctypes.c_void_p * len(cbs))(*[ctypes.cast(c, ctypes.c_void_p) for c in cbs])
        KEEP.append((bufs, cbs, arr))
        p.kernelParams = ctypes.addressof(arr); p.extra = 0
    return p

tgt = {M: harvest(M) for M in TARGETS}
gb, nodes_b, recs_b = harvest(BASE)
gb.replay(); torch.cuda.synchronize()
gexec = gb.raw_cuda_graph_exec()
print(f"DT={which} OP={OP} METHOD={METHOD} host M={BASE} nodes={len(nodes_b)} "
      f"kernel={recs_b[0]['name'][:56]} smem={recs_b[0]['smem']} cluster={recs_b[0]['cluster']}")
for M in TARGETS:
    _, _, r = tgt[M]
    print(f"  target M={M:<6} {r[0]['name'][:58]:<58} cluster={r[0]['cluster']} smem={r[0]['smem']} "
          f"grid={r[0]['grid']} swap_func={r[0]['funci'] != recs_b[0]['funci']}")
print()
REPS = int(os.environ.get("REPS", "0"))


def _switch(M):
    """Switch exec to M and return the time taken (us). Times only the switch itself, not replay or checks.
    The template graphs are captured ahead of time (`tgt`), so the ExecUpdate number is "the price of
    each switch once branches are pre-captured" -- the form DynaGraph would use in practice."""
    gt, nds_t, recs_t = tgt[M]
    t0 = time.perf_counter_ns()
    if METHOD == "setparams":
        for nd, rec in zip(nodes_b, recs_t):
            ck(cd.cuGraphExecKernelNodeSetParams(gexec, nd, mk(rec)))
    else:
        r = cr.cudaGraphExecUpdate(gexec, gt.raw_cuda_graph())
        if int(r[0]) != 0:
            raise RuntimeError(f"ExecUpdate err={int(r[0])}")
    return (time.perf_counter_ns() - t0) / 1000.0


if REPS:
    # Interleaved: each round switches to every target in turn, instead of hitting one target N times in a row.
    # The machine always has other jobs running, so steady-state numbers mean nothing (see docs/METHODOLOGY.md).
    times = {M: [] for M in TARGETS}
    for _ in range(REPS):
        for M in TARGETS:
            try:
                times[M].append(_switch(M))
            except RuntimeError as e:
                print(f"M={M} switch failed: {str(e)[:70]}")
                times[M] = None
                break
        else:
            continue
        break
    print(f"\nSwitch time ({METHOD}, {len(nodes_b)} nodes, REPS={REPS}, template graphs pre-captured)")
    print("target  cluster       swapfn  median_us  min_us    p90us")
    for M in TARGETS:
        v = times.get(M)
        if not v:
            continue
        _, _, r = tgt[M]
        v2 = sorted(v)
        print(f"{M:<7} {str(r[0]['cluster']):<13} {str(r[0]['funci'] != recs_b[0]['funci']):<7} "
              f"{statistics.median(v2):<10.2f} {v2[0]:<9.2f} {v2[int(len(v2) * 0.9)]:.2f}")
    print()

for M in TARGETS:
    gt, nds_t, recs_t = tgt[M]
    same_cluster = recs_t[0]["cluster"] == recs_b[0]["cluster"]
    y_full.fill_(SENT)
    try:
        if METHOD == "setparams":
            if len(recs_t) != len(recs_b): print(f"M={M} node count differs, skipped"); continue
            for nd, rec in zip(nodes_b, recs_t):
                ck(cd.cuGraphExecKernelNodeSetParams(gexec, nd, mk(rec)))
        else:
            r = cr.cudaGraphExecUpdate(gexec, gt.raw_cuda_graph())
            if int(r[0]) != 0:
                info = r[1]
                print(f"M={M:<6} cluster={recs_t[0]['cluster']} ExecUpdate failed err={int(r[0])} "
                      f"result={getattr(info, 'result', info)}")
                continue
        gb.replay(); torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"M={M:<6} cluster={recs_t[0]['cluster']} same_cluster={same_cluster} failed: {str(e)[:90]}")
        continue
    ref = (torch.addmm(b, x_full[:M], w) if OP == "addmm" else torch.mm(x_full[:M], w)).float()
    got = y_full[:M].float()
    err = (got - ref).abs().max().item()
    nw = int((y_full[:M] == SENT).sum().item())
    tail = 0 if M == MMAX else int((y_full[M:] != SENT).sum().item())
    ok = "OK" if (err == 0 and nw == 0 and tail == 0) else "**BAD**"
    print(f"M={M:<6} cluster={recs_t[0]['cluster']} same_cluster={same_cluster} {ok} "
          f"max|err|={err:.4g} unwritten={nw} oob={tail}")
