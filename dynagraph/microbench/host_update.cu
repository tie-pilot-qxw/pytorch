// Host-side exec update costs, the numbers a C++ patcher would pay per call:
//   cudaGraphExecKernelNodeSetParams x N (grid + one scalar changed)
//   cudaGraphExecUpdate whole graph after editing N nodes in the cudaGraph_t
//   cudaGraphExecChildGraphNodeSetParams x M
// All host time; GPU does nothing but instantiate. Build: nvcc -O2 -o host_update host_update.cu
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <thread>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s failed: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void k(float* p, long long n) { long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) p[i] += 1.f; }

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
  const int N = 48, M = 24, REPS = 200;
  float* buf; CK(cudaMalloc(&buf, 1 << 20));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  // main graph: N kernel nodes + M child graph nodes, captured
  std::vector<cudaGraph_t> kids(M);
  for (int j = 0; j < M; ++j) {
    cudaStream_t cs; CK(cudaStreamCreate(&cs));
    CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
    k<<<64, 128, 0, cs>>>(buf, 8192);
    CK(cudaStreamEndCapture(cs, &kids[j]));
  }
  cudaGraph_t g;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < N; ++i) {
    k<<<64, 128, 0, s>>>(buf, 8192);
    if (i < M) {
      cudaStreamCaptureStatus st; cudaGraph_t cap; const cudaGraphNode_t* deps; const cudaGraphEdgeData* ed; size_t nd;
      CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &cap, &deps, &ed, &nd));
      cudaGraphNode_t cn; CK(cudaGraphAddChildGraphNode(&cn, cap, deps, nd, kids[i]));
      CK(cudaStreamUpdateCaptureDependencies(s, &cn, nullptr, 1, cudaStreamSetCaptureDependencies));
    }
  }
  CK(cudaStreamEndCapture(s, &g));
  size_t nn = 0; CK(cudaGraphGetNodes(g, nullptr, &nn));
  std::vector<cudaGraphNode_t> nodes(nn); CK(cudaGraphGetNodes(g, nodes.data(), &nn));
  std::vector<cudaGraphNode_t> knodes, cnodes;
  for (auto nd : nodes) { cudaGraphNodeType t; CK(cudaGraphNodeGetType(nd, &t)); (t == cudaGraphNodeTypeKernel ? knodes : cnodes).push_back(nd); }
  cudaGraphExec_t ex; CK(cudaGraphInstantiate(&ex, g, 0));
  CK(cudaGraphUpload(ex, s)); CK(cudaStreamSynchronize(s));
  printf("graph: %zu kernel nodes, %zu child nodes\n", knodes.size(), cnodes.size());

  // 1. per-node exec update: change grid + the scalar arg
  std::vector<cudaKernelNodeParams> params(knodes.size());
  for (size_t i = 0; i < knodes.size(); ++i) CK(cudaGraphKernelNodeGetParams(knodes[i], &params[i]));
  double best = 1e9;
  for (int r = 0; r < REPS; ++r) {
    double t0 = now_us();
    for (size_t i = 0; i < knodes.size(); ++i) {
      params[i].gridDim.x = 63 + (r & 1);
      *(long long*)params[i].kernelParams[1] = 8000 + r;
      CK(cudaGraphExecKernelNodeSetParams(ex, knodes[i], &params[i]));
    }
    double t = now_us() - t0; if (t < best) best = t;
  }
  printf("cudaGraphExecKernelNodeSetParams x %zu: %.1f us (%.2f us/node)\n", knodes.size(), best, best / knodes.size());

  // 2. edit the cudaGraph_t, then one cudaGraphExecUpdate
  best = 1e9;
  for (int r = 0; r < REPS; ++r) {
    double t0 = now_us();
    for (size_t i = 0; i < knodes.size(); ++i) {
      params[i].gridDim.x = 63 + (r & 1);
      *(long long*)params[i].kernelParams[1] = 8000 + r;
      CK(cudaGraphKernelNodeSetParams(knodes[i], &params[i]));
    }
    cudaGraphExecUpdateResultInfo info;
    CK(cudaGraphExecUpdate(ex, g, &info));
    double t = now_us() - t0; if (t < best) best = t;
  }
  printf("edit graph x %zu + cudaGraphExecUpdate: %.1f us\n", knodes.size(), best);

  // 3. child swaps
  std::vector<cudaGraph_t> kids2(M);
  for (int j = 0; j < M; ++j) {
    cudaStream_t cs; CK(cudaStreamCreate(&cs));
    CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
    k<<<32, 128, 0, cs>>>(buf, 4096);
    CK(cudaStreamEndCapture(cs, &kids2[j]));
  }
  best = 1e9;
  for (int r = 0; r < REPS; ++r) {
    double t0 = now_us();
    for (size_t j = 0; j < cnodes.size(); ++j) CK(cudaGraphExecChildGraphNodeSetParams(ex, cnodes[j], (r & 1) ? kids2[j] : kids[j]));
    double t = now_us() - t0; if (t < best) best = t;
  }
  printf("cudaGraphExecChildGraphNodeSetParams x %zu: %.1f us (%.2f us/child)\n", cnodes.size(), best, best / cnodes.size());

  // 4. child swaps via graph edit + one ExecUpdate
  best = 1e9;
  for (int r = 0; r < REPS; ++r) {
    double t0 = now_us();
    for (size_t j = 0; j < cnodes.size(); ++j) { cudaGraphNodeParams np{}; np.type = cudaGraphNodeTypeGraph; np.graph.graph = (r & 1) ? kids2[j] : kids[j]; CK(cudaGraphNodeSetParams(cnodes[j], &np)); }
    cudaGraphExecUpdateResultInfo info;
    CK(cudaGraphExecUpdate(ex, g, &info));
    double t = now_us() - t0; if (t < best) best = t;
  }
  printf("edit %zu child nodes + cudaGraphExecUpdate: %.1f us\n", cnodes.size(), best);

  // 4b. the same per-node updates split over T host threads: do exec updates
  // scale across cores, or serialize on the driver's lock?
  for (int T : {2, 4, 8}) {
    best = 1e9;
    for (int r = 0; r < REPS; ++r) {
      double t0 = now_us();
      std::vector<std::thread> th;
      for (int t = 0; t < T; ++t) th.emplace_back([&, t, r]() {
        for (size_t i = t; i < knodes.size(); i += T) {
          params[i].gridDim.x = 63 + (r & 1);
          *(long long*)params[i].kernelParams[1] = 8000 + r;
          cudaGraphExecKernelNodeSetParams(ex, knodes[i], &params[i]);
        }
      });
      for (auto& x : th) x.join();
      double tt = now_us() - t0; if (tt < best) best = tt;
    }
    printf("cudaGraphExecKernelNodeSetParams x %zu over %d threads: %.1f us\n", knodes.size(), T, best);
  }
  // 5. a launch, for scale
  best = 1e9;
  for (int r = 0; r < REPS; ++r) { double t0 = now_us(); CK(cudaGraphLaunch(ex, s)); double t = now_us() - t0; if (t < best) best = t; CK(cudaStreamSynchronize(s)); }
  printf("cudaGraphLaunch host: %.1f us\n", best);
  return 0;
}
