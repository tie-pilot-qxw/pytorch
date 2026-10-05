#!/usr/bin/env python3
"""DeepGEMM bf16_gemm_nt on the four (N,K) of Qwen3-0.6B, M=1..MMAX:
numerics, first-call time (JIT), and the captured launch (kernel/grid/cluster/smem/param bytes).
Answers: what tier 3 must express to register a GEMM, and how many variants there are."""
import os, time, collections, ctypes
import torch
import vllm.third_party.deep_gemm as dg
from cuda.bindings import runtime as cr, driver as cu
from torch.cuda._utils import _check_cuda_bindings as ck

MMAX = int(os.environ.get("MMAX", "512"))
SHAPES = {"qkv": (4096, 1024), "o": (1024, 2048), "gate_up": (6144, 1024), "down": (1024, 3072)}
torch.manual_seed(0)

def launch_of(g):
    raw = g.raw_cuda_graph()
    n = int(ck(cr.cudaGraphGetNodes(raw))[1])
    nodes = ck(cr.cudaGraphGetNodes(raw, n))[0]
    out = []
    for nd in nodes:
        t = ck(cr.cudaGraphNodeGetType(nd))
        if t != cr.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            out.append(("non-kernel", str(t)))
            continue
        p = ck(cu.cuGraphKernelNodeGetParams(cu.CUgraphNode(int(nd))))
        name = ck(cu.cuFuncGetName(p.func)).decode()
        try:
            cl = ck(cu.cuGraphKernelNodeGetAttribute(cu.CUgraphNode(int(nd)),
                    cu.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION))
            cluster = (cl.clusterDim.x, cl.clusterDim.y, cl.clusterDim.z)
        except Exception:
            cluster = None
        nparam, size = 0, 0
        while True:
            try:
                off, sz = ck(cu.cuFuncGetParamInfo(p.func, nparam))
                size = max(size, int(off) + int(sz)); nparam += 1
            except Exception:
                break
        out.append((name[:90], (p.gridDimX, p.gridDimY, p.gridDimZ),
                    (p.blockDimX, p.blockDimY, p.blockDimZ), int(p.sharedMemBytes), cluster,
                    nparam, size, "extra" if not int(p.kernelParams or 0) else "kp"))
    return out

for role, (N, K) in SHAPES.items():
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K ** 0.5
    a_full = torch.randn(MMAX, K, device="cuda", dtype=torch.bfloat16)
    d_full = torch.empty(MMAX, N, device="cuda", dtype=torch.bfloat16)
    first = []; bad = []; variants = collections.OrderedDict(); prev = None; changes = 0
    for M in range(1, MMAX + 1):
        a, d = a_full[:M], d_full[:M]
        t0 = time.perf_counter(); dg.bf16_gemm_nt(a, b, d); torch.cuda.synchronize()
        first.append((time.perf_counter() - t0) * 1e3)
        ref = (a.float() @ b.float().t())
        err = ((d.float() - ref).abs().max() / ref.abs().max()).item()
        if not err < 2e-2:
            bad.append((M, err))
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            g = torch.cuda.CUDAGraph(keep_graph=True)
            g.capture_begin(); dg.bf16_gemm_nt(a, b, d); g.capture_end()
        L = launch_of(g)
        key = tuple((x[0], x[3], x[4], x[1]) if len(x) > 2 else x for x in L)
        variants.setdefault(key, []).append(M)
        if prev is not None and key != prev:
            changes += 1
        prev = key
    t_sorted = sorted(first)
    print(f"\n== {role} N={N} K={K}: M=1..{MMAX} numeric errors {len(bad)} {bad[:3]}")
    print(f"   first call ms: median {statistics_median(first) if False else t_sorted[len(t_sorted)//2]:.2f} max {max(first):.0f} ({sum(x>5 for x in first)} calls >5ms)")
    print(f"   {len(variants)} distinct launches, changed {changes} times between adjacent M")
    clusters = collections.Counter()
    for key, Ms in variants.items():
        for x in key:
            clusters[(x[2] if len(x) > 2 else None, len(key))] += len(Ms)
    print(f"   (cluster, node count) distribution: {dict(clusters)}")
    for key, Ms in list(variants.items())[:8]:
        print(f"   M {Ms[0]}..{Ms[-1]} ({len(Ms)}): {key}")
    ex = launch_of(g)
    print(f"   example M={MMAX}: {ex}")
