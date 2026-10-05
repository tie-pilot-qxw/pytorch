// Time the REAL generated planner (dumped from the 12-block bench model) on a
// graph of dummy device-updatable nodes with param buffers big enough for the
// offsets it patches. Attribution by variant: full / no pointer patches /
// no scalar patches / grid only. Compile the planner source exactly as
// _compile_module does (-arch=sm_90a -cubin) and load it with the driver API.
//   nvcc -O2 -arch=sm_90a planner_real.cu -o planner_real -lcuda
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)
#define CU(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s; cuGetErrorString(r, &s); printf("CUERR %s @%d: %s\n", #x, __LINE__, s); return 1; } } while (0)

struct Big { long long v[128]; };   // 1 KiB of params: covers any offset the planner uses
__global__ void dummy(Big b, int id) { if (threadIdx.x == 0 && blockIdx.x == 0 && b.v[0] == 12345) b.v[1] = id; }

static int run(const char* cubin, int N, bool with_planner, float* us) {
  CUmodule mod; CUfunction fpl = 0, fly = 0;
  if (with_planner) {
    CU(cuModuleLoad(&mod, cubin));
    CU(cuModuleGetFunction(&fpl, mod, "dynagraph_planner"));
    CU(cuModuleGetFunction(&fly, mod, "dynagraph_layout"));
  }
  long long *ctx, *slot; char* arena; cudaGraphDeviceNode_t* h;
  CK(cudaMalloc(&ctx, 8 * (4 + N))); CK(cudaMalloc(&slot, 8 * 512)); CK(cudaMalloc(&arena, 64 << 20));
  CK(cudaMalloc(&h, sizeof(cudaGraphDeviceNode_t) * N));
  std::vector<long long> hc(4 + N, 1); hc[0] = 512; hc[1] = 1;   // s77=512, flag=1, enabled=1...
  CK(cudaMemcpy(ctx, hc.data(), 8 * (4 + N), cudaMemcpyHostToDevice));
  CK(cudaMemset(slot, 0, 8 * 512));
  cudaStream_t s; CK(cudaStreamCreate(&s)); cudaGraph_t g; cudaGraphExec_t ge;
  std::vector<cudaGraphDeviceNode_t> hs(N);
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  if (with_planner) {
    void* a1[] = {&ctx, &slot};
    CU(cuLaunchKernel(fly, 1, 1, 1, 1, 1, 1, 0, s, a1, nullptr));
    void* a2[] = {&h, &ctx, &arena, &slot};
    CU(cuLaunchKernel(fpl, (N + 127) / 128, 1, 1, 128, 1, 1, 0, s, a2, nullptr));
  }
  for (int i = 0; i < N; ++i) {
    cudaLaunchAttribute attr{}; attr.id = cudaLaunchAttributeDeviceUpdatableKernelNode;
    attr.val.deviceUpdatableKernelNode.deviceUpdatable = 1;
    cudaLaunchConfig_t cfg{}; cfg.gridDim = dim3(1); cfg.blockDim = dim3(32); cfg.stream = s;
    cfg.attrs = &attr; cfg.numAttrs = with_planner ? 1 : 0;
    Big b{}; CK(cudaLaunchKernelEx(&cfg, dummy, b, i));
    if (with_planner) hs[i] = attr.val.deviceUpdatableKernelNode.devNode;
  }
  CK(cudaStreamEndCapture(s, &g));
  if (with_planner) CK(cudaMemcpy(h, hs.data(), sizeof(cudaGraphDeviceNode_t) * N, cudaMemcpyHostToDevice));
  CK(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));
  for (int i = 0; i < 10; ++i) CK(cudaGraphLaunch(ge, s));
  CK(cudaStreamSynchronize(s));
  std::vector<float> ms; cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int i = 0; i < 40; ++i) {
    CK(cudaStreamSynchronize(s)); CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s));
    CK(cudaEventRecord(e1, s)); CK(cudaStreamSynchronize(s)); float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end()); *us = ms[ms.size() / 2] * 1000.0f;
  CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g)); CK(cudaStreamDestroy(s));
  CK(cudaFree(ctx)); CK(cudaFree(slot)); CK(cudaFree(arena)); CK(cudaFree(h));
  return 0;
}
int main(int argc, char** argv) {
  int N = atoi(argv[1]);
  float base; if (run(nullptr, N, false, &base)) return 1;
  printf("floor (no planner): %.1f us\n", base);
  for (int i = 2; i < argc; ++i) {
    float us; if (run(argv[i], N, true, &us)) return 1;
    printf("%-28s %7.1f us  -> planner+layout = %6.1f us\n", argv[i], us, us - base);
  }
  return 0;
}
