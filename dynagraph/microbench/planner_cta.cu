// Does spreading the planner's update calls over more threads / more CTAs
// make them cheaper, or do they serialize inside the device runtime?
//
// The real planner is one thread per node, each thread making K ~ 6 calls in
// sequence (SetEnabled, SetGridDim, a scalar SetParam, 3-4 pointer SetParams),
// all 48 threads in one block. Three layouts of the same 48*K calls:
//   A  1 thread/node, K calls each, 1 block                 (today)
//   B  K threads/node, 1 call each, 1 block  (48*K threads)
//   C  K threads/node, 1 call each, one block PER NODE (48 blocks)
// If B ~ A the calls do not overlap across threads; if C ~ B they do not
// overlap across SMs either -- then "more CTAs" cannot help and the only
// lever is fewer calls.
//
//   nvcc -O2 -arch=sm_90a -rdc=true planner_cta.cu -o planner_cta -lcudadevrt
#include <algorithm>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

constexpr int K = 6;   // calls per node

__global__ void worker(int* out, int a, int b, int c, int d, int e, int f, int id) {
  if (threadIdx.x == 0 && blockIdx.x == 0) out[id] = a + b + c + d + e + f + gridDim.x;
}

// One call: slot j of node i. j==0 is SetGridDim, the rest are SetParam on
// successive int args (offsets 4, 8, ... in worker's param buffer).
__device__ __forceinline__ void one_call(cudaGraphDeviceNode_t h, int j, int L, int i) {
  if (j == 0) {
    cudaGraphKernelNodeSetGridDim(h, dim3((unsigned)((L + 255) / 256), 1, 1));
  } else {
    // worker's param buffer: int* out at 0 (8 bytes), then ints a..f at 8,12,...
    // offset 4 would land inside the pointer -- that was an illegal access.
    int v = L * 3 + i + j;
    cudaGraphKernelNodeSetParam(h, (size_t)(8 + 4 * (j - 1)), &v, sizeof(int));
  }
}

__global__ void planner_A(const cudaGraphDeviceNode_t* h, int N, const long long* ctx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int L = (int)ctx[0];
  for (int j = 0; j < K; ++j) one_call(h[i], j, L, i);
}
__global__ void planner_B(const cudaGraphDeviceNode_t* h, int N, const long long* ctx) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t / K, j = t % K;
  if (i >= N) return;
  one_call(h[i], j, (int)ctx[0], i);
}
__global__ void planner_C(const cudaGraphDeviceNode_t* h, int N, const long long* ctx) {
  int i = blockIdx.x, j = threadIdx.x;
  if (i >= N || j >= K) return;
  one_call(h[i], j, (int)ctx[0], i);
}

static int run(int N, int mode, float* us_out) {
  int *d_out; long long* d_ctx; cudaGraphDeviceNode_t* d_h;
  CK(cudaMalloc(&d_ctx, 8)); CK(cudaMalloc(&d_out, 4 * N));
  CK(cudaMalloc(&d_h, sizeof(cudaGraphDeviceNode_t) * N));
  long long L = 4096; CK(cudaMemcpy(d_ctx, &L, 8, cudaMemcpyHostToDevice));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaGraph_t g; cudaGraphExec_t ge;
  std::vector<cudaGraphDeviceNode_t> hs(N);
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  if (mode == 1) planner_A<<<(N + 127) / 128, 128, 0, s>>>(d_h, N, d_ctx);
  if (mode == 2) planner_B<<<(N * K + 127) / 128, 128, 0, s>>>(d_h, N, d_ctx);
  if (mode == 3) planner_C<<<N, 32, 0, s>>>(d_h, N, d_ctx);
  for (int i = 0; i < N; ++i) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeDeviceUpdatableKernelNode;
    attr.val.deviceUpdatableKernelNode.deviceUpdatable = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(1); cfg.blockDim = dim3(32); cfg.stream = s;
    cfg.attrs = &attr; cfg.numAttrs = mode ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, worker, d_out, 0, 0, 0, 0, 0, 0, i));
    if (mode) hs[i] = attr.val.deviceUpdatableKernelNode.devNode;
  }
  CK(cudaStreamEndCapture(s, &g));
  if (mode) CK(cudaMemcpy(d_h, hs.data(), sizeof(cudaGraphDeviceNode_t) * N, cudaMemcpyHostToDevice));
  CK(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));
  for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(ge, s));
  CK(cudaStreamSynchronize(s));
  std::vector<float> ms; cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int i = 0; i < 60; ++i) {
    CK(cudaStreamSynchronize(s)); CK(cudaEventRecord(e0, s));
    CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s)); CK(cudaStreamSynchronize(s));
    float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end()); *us_out = ms[ms.size() / 2] * 1000.0f;
  CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g)); CK(cudaStreamDestroy(s));
  CK(cudaFree(d_ctx)); CK(cudaFree(d_out)); CK(cudaFree(d_h));
  return 0;
}

int main() {
  const char* names[4] = {"no planner (floor)", "A: 1 thr/node x K calls, 1 blk",
                          "B: K thr/node x 1 call, 1 blk", "C: K thr/node x 1 call, N blks"};
  printf("K=%d calls per node\n%6s  %-34s %10s %12s\n", K, "nodes", "layout", "us/replay", "us/call");
  for (int N : {48, 128, 512}) {
    float base = 0;
    for (int mode = 0; mode < 4; ++mode) {
      float us; if (run(N, mode, &us)) return 1;
      if (mode == 0) base = us;
      if (mode) printf("%6d  %-34s %10.1f %12.4f\n", N, names[mode], us, (us - base) / (N * K));
      else      printf("%6d  %-34s %10.1f %12s\n", N, names[mode], us, "-");
    }
    printf("\n");
  }
  return 0;
}
