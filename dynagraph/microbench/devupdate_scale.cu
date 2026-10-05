// cost of a planner kernel re-parameterizing N device-updatable nodes (grid + one scalar param each) per launch
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 1;}}while(0)
__global__ void worker(int* out, int n, int id) { if (threadIdx.x==0 && blockIdx.x==0) out[id] = n + gridDim.x; }
__global__ void planner(const cudaGraphDeviceNode_t* handles, int N, const int* in) {
  int L = in[0];
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    __align__(16) char buf[2 * sizeof(cudaGraphKernelNodeUpdate)]; cudaGraphKernelNodeUpdate* u = (cudaGraphKernelNodeUpdate*)buf;
    u[0].node = handles[i]; u[0].field = cudaGraphKernelNodeFieldGridDim; u[0].updateData.gridDim = dim3((L + 255) / 256, 1, 1);
    int nn = L * 3 + i;
    u[1].node = handles[i]; u[1].field = cudaGraphKernelNodeFieldParam; u[1].updateData.param.offset = 8; u[1].updateData.param.pValue = &nn; u[1].updateData.param.size = sizeof(int);
    cudaGraphKernelNodeUpdatesApply(u, 2);
  }
}
int run(int N, bool patch) {
  int *d_in, *d_out; cudaGraphDeviceNode_t* d_handles;
  CK(cudaMalloc(&d_in, 4)); CK(cudaMalloc(&d_out, 4 * N)); CK(cudaMalloc(&d_handles, sizeof(cudaGraphDeviceNode_t) * N));
  cudaStream_t s; CK(cudaStreamCreate(&s)); cudaGraph_t g; cudaGraphExec_t ge;
  std::vector<cudaGraphDeviceNode_t> hs(N);
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  if (patch) planner<<<1, 1024, 0, s>>>(d_handles, N, d_in);
  for (int i = 0; i < N; ++i) {
    cudaLaunchAttribute attr{}; attr.id = cudaLaunchAttributeDeviceUpdatableKernelNode; attr.val.deviceUpdatableKernelNode.deviceUpdatable = 1;
    cudaLaunchConfig_t cfg{}; cfg.gridDim = dim3(1); cfg.blockDim = dim3(256); cfg.stream = s; cfg.attrs = &attr; cfg.numAttrs = patch ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, worker, d_out, 0, i));
    if (patch) hs[i] = attr.val.deviceUpdatableKernelNode.devNode;
  }
  CK(cudaStreamEndCapture(s, &g));
  if (patch) CK(cudaMemcpy(d_handles, hs.data(), sizeof(cudaGraphDeviceNode_t) * N, cudaMemcpyHostToDevice));
  CK(cudaGraphInstantiate(&ge, g, 0)); CK(cudaGraphUpload(ge, s));
  int L = 7000; CK(cudaMemcpy(d_in, &L, 4, cudaMemcpyHostToDevice));
  CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
  if (patch) { std::vector<int> out(N); CK(cudaMemcpy(out.data(), d_out, 4 * N, cudaMemcpyDeviceToHost)); int bad = 0; for (int i = 0; i < N; ++i) bad += (out[i] != L * 3 + i + (L + 255) / 256); printf("  N=%d patched nodes: %d mismatches\n", N, bad); }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0, s);
  for (int i = 0; i < 200; ++i) CK(cudaGraphLaunch(ge, s));
  cudaEventRecord(e1, s); CK(cudaStreamSynchronize(s)); float ms; cudaEventElapsedTime(&ms, e0, e1);
  printf("  N=%d %-13s GPU time/launch %.1f us  (%.3f us/node)\n", N, patch ? "device-patch" : "static", ms * 1000 / 200, ms * 1000 / 200 / N);
  return 0;
}
int main() { for (int N : {500, 3000}) { run(N, false); run(N, true); } return 0; }
