"""Minimal prototype, step 2: the minimum host time for each thing DG "must do" on a new shape (C++ implementation / the library itself).

  setparams  N kernel nodes in one exec, a C++ loop of cudaGraphExecKernelNodeSetParams (each call changes the grid and one argument)
  outputs    C++ builds N (offset, sizes, strides) tensors on one arena and hands them back to Python
  describe   DeepGEMM describe, the real GEMM shapes of ESM-2 35M (fwd + dX + dW)
  launch     cudaGraphLaunch of one exec
"""
import time

import torch
from torch.utils.cpp_extension import load_inline

CPP = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void k_noop(float* p, int n) { if (threadIdx.x == 0 && blockIdx.x == 0 && n < 0) p[0] = 1; }

struct G { cudaGraph_t g; cudaGraphExec_t e; std::vector<cudaGraphNode_t> nodes; };
static G* build(int n) {
    auto* s = new G();
    cudaGraphCreate(&s->g, 0);
    cudaGraphNode_t prev = nullptr;
    static float* buf = nullptr; if (!buf) cudaMalloc(&buf, 16);
    for (int i = 0; i < n; ++i) {
        cudaKernelNodeParams p{}; int nn = i; void* args[] = {&buf, &nn};
        p.func = (void*)k_noop; p.gridDim = dim3(1); p.blockDim = dim3(32); p.kernelParams = args;
        cudaGraphNode_t node;
        cudaGraphAddKernelNode(&node, s->g, prev ? &prev : nullptr, prev ? 1 : 0, &p);
        s->nodes.push_back(node); prev = node;
    }
    cudaGraphInstantiate(&s->e, s->g, 0);
    return s;
}

// ns per node for one pass of SetParams over n nodes, averaged over reps passes
double bench_setparams(int n, int reps) {
    G* s = build(n);
    static float* buf = nullptr; if (!buf) cudaMalloc(&buf, 16);
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
        for (int i = 0; i < n; ++i) {
            cudaKernelNodeParams p{}; int nn = i + r; void* args[] = {&buf, &nn};
            p.func = (void*)k_noop; p.gridDim = dim3(1 + (r & 7)); p.blockDim = dim3(32); p.kernelParams = args;
            cudaGraphExecKernelNodeSetParams(s->e, s->nodes[i], &p);
        }
    auto t1 = std::chrono::steady_clock::now();
    cudaGraphExecDestroy(s->e); cudaGraphDestroy(s->g); delete s;
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / (double(n) * reps);
}

// ns per launch of one exec with n nodes
double bench_launch(int n, int reps) {
    G* s = build(n);
    cudaStream_t st; cudaStreamCreate(&st);
    cudaGraphLaunch(s->e, st); cudaStreamSynchronize(st);
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) cudaGraphLaunch(s->e, st);
    auto t1 = std::chrono::steady_clock::now();
    cudaStreamSynchronize(st);
    cudaGraphExecDestroy(s->e); cudaGraphDestroy(s->g); delete s;
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
}

// n tensors viewing `arena` at the given element offsets / sizes / strides, returned to Python
std::vector<at::Tensor> make_outputs(const at::Tensor& arena, const std::vector<int64_t>& offs,
                                     const std::vector<std::vector<int64_t>>& sizes,
                                     const std::vector<std::vector<int64_t>>& strides) {
    std::vector<at::Tensor> out; out.reserve(offs.size());
    auto opts = arena.options();
    for (size_t i = 0; i < offs.size(); ++i)
        out.push_back(at::empty({0}, opts).set_(arena.storage(), offs[i], sizes[i], strides[i]));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bench_setparams", &bench_setparams);
    m.def("bench_launch", &bench_launch);
    m.def("make_outputs", &make_outputs);
}
"""


def main():
    ext = load_inline("dg_proto_units", cpp_sources="", cuda_sources=CPP, verbose=False,
                      extra_cuda_cflags=["-O2"])
    torch.cuda.init()
    for n in (200, 600):
        ext.bench_setparams(n, 5)
        print(f"setparams  C++: {ext.bench_setparams(n, 50) / 1e3:.2f} us/node (exec of {n} nodes)")
    for n in (200, 600):
        print(f"launch     C++: {ext.bench_launch(n, 200) / 1e3:.1f} us/launch (exec of {n} nodes)")

    arena = torch.empty(1 << 28, dtype=torch.bfloat16, device="cuda")
    N = 500
    offs = [i * 4096 for i in range(N)]
    sizes = [[64, 64] for _ in range(N)]
    strides = [[64, 1] for _ in range(N)]
    ext.make_outputs(arena, offs, sizes, strides)
    t0 = time.perf_counter()
    for _ in range(20):
        ext.make_outputs(arena, offs, sizes, strides)
    print(f"outputs    C++: {(time.perf_counter() - t0) / 20 / N * 1e6:.2f} us/tensor ({N} tensors, including the handoff to Python)")

    import deep_gemm

    H, I, V = 480, 1920, 40
    e = lambda *sh: torch.empty(*sh, device="cuda", dtype=torch.bfloat16)

    def gemms(T):
        out = []
        for n, k in ((H, H), (H, H), (H, H), (H, H), (I, H), (H, I), (H, H), (V, H)):
            out.append((e(T, k), e(n, k), e(T, n), "nk"))          # fwd  y = x W^T
            out.append((e(T, n), e(n, k).t(), e(T, k), "nk"))      # dX = dy W
            out.append((e(T, n).t(), e(T, k).t(), e(n, k), "n"))   # dW = dy^T x   (K = T)
        return out

    for a_, b_, d_, dims in gemms(8000):
        deep_gemm.bf16_gemm_nt(a_, b_, d_, compiled_dims=dims)
    torch.cuda.synchronize()
    ts = []
    for T in (7936, 7872, 7808, 7744):
        for a_, b_, d_, dims in gemms(T):
            t0 = time.perf_counter()
            deep_gemm._C.describe_begin()
            deep_gemm.bf16_gemm_nt(a_, b_, d_, compiled_dims=dims)
            deep_gemm._C.describe_end()
            ts.append(time.perf_counter() - t0)
    shapes = gemms(8)
    ts.sort()
    print(f"describe   DeepGEMM (Python call): median {ts[len(ts) // 2] * 1e6:.1f} us/site, {len(shapes)} shapes")


if __name__ == "__main__":
    main()
