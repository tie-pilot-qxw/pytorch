"""Is it safe for SetParams to switch a DeepGEMM node from one config to another (different BLOCK_M / thread count)?

Capture bf16_gemm_nt once at M_big (node = variant A), describe at M_small (variant B), write B's launch
into the same node with cuGraphExecKernelNodeSetParams, launch, and check the result and whether an
illegal instruction occurs. Also the reverse: capture at small M, swap in big M.
"""
import ctypes as ct
import sys

import deep_gemm
import torch
from cuda.bindings import driver as cu
from cuda.bindings import runtime as cr
from torch.cuda._utils import _check_cuda_bindings as ck
from torch.utils import _capture_launch as cl

K = N = 480
A = torch.randn(8192, K, device="cuda", dtype=torch.bfloat16)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
D = torch.empty(8192, N, device="cuda", dtype=torch.bfloat16)


def run(M):
    deep_gemm.bf16_gemm_nt(A[:M], W, D[:M])


def describe(M):
    deep_gemm._C.describe_begin()
    run(M)
    return deep_gemm._C.describe_end()


def test(m_cap, m_new):
    run(m_cap), run(m_new)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g.capture_begin()
        run(m_cap)
        g.capture_end()
    g.instantiate()
    raw = g.raw_cuda_graph()
    nodes = ck(cr.cudaGraphGetNodes(raw, 1))[0]
    node = int(nodes[0])
    cap = cl.read_node(node)
    func, grid, block, smem, cluster, pdl, args = describe(m_new)[0]
    print(f"capture M={m_cap}: {cl._name(cap.func)[40:90]} block {cap.block} smem {cap.smem} cluster {cap.cluster}")
    print(f"new     M={m_new}: {cl._name(func)[40:90]} block {tuple(block)} smem {smem} cluster {cluster}")
    bufs = [ct.create_string_buffer(b, len(b)) for b in args]
    arr = (ct.c_void_p * len(bufs))(*[ct.addressof(b) for b in bufs])
    p = cu.CUDA_KERNEL_NODE_PARAMS()
    p.func = func
    p.gridDimX, p.gridDimY, p.gridDimZ = grid
    p.blockDimX, p.blockDimY, p.blockDimZ = block
    p.sharedMemBytes = smem
    p.kernelParams = ct.addressof(arr)
    p.extra = 0
    ck(cu.cuGraphExecKernelNodeSetParams(g.raw_cuda_graph_exec(), node, p))
    D.zero_()
    g.replay()
    try:
        torch.cuda.synchronize()
    except Exception as e:
        print("   FAILED:", str(e).split("\n")[0])
        sys.exit(1)
    ref = A[:m_new].float() @ W.float().t()
    err = ((D[:m_new].float() - ref).abs().max() / ref.abs().max()).item()
    print(f"   ok, rel err {err:.2e}")


which = sys.argv[1] if len(sys.argv) > 1 else "big2small"
if which == "big2small":
    test(8000, 200)
else:
    test(200, 8000)
