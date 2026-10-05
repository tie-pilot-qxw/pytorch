#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 1;}}while(0)

__global__ void worker(int* out, int n) {
  if (threadIdx.x == 0 && blockIdx.x == 0) { out[0] = gridDim.x; out[1] = n; }
}
// "planner": reads this step's L from device memory, re-parameterizes the downstream node in the SAME graph launch
__global__ void planner(const cudaGraphDeviceNode_t* handles, const int* in) {
  if (threadIdx.x == 0) {
    int L = in[0];
    cudaGraphKernelNodeSetGridDim(handles[0], dim3((L + 255) / 256, 1, 1));
    int nn = L * 3;
    cudaGraphKernelNodeSetParam(handles[0], 8 /*offset of n after the 8-byte pointer*/, &nn, sizeof(int));
  }
}
int main() {
  int *d_in, *d_out; cudaGraphDeviceNode_t* d_handles;
  CK(cudaMalloc(&d_in, 4)); CK(cudaMalloc(&d_out, 8)); CK(cudaMalloc(&d_handles, sizeof(cudaGraphDeviceNode_t)));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  planner<<<1, 32, 0, s>>>(d_handles, d_in);
  cudaLaunchAttribute attr{}; attr.id = cudaLaunchAttributeDeviceUpdatableKernelNode; attr.val.deviceUpdatableKernelNode.deviceUpdatable = 1;
  cudaLaunchConfig_t cfg{}; cfg.gridDim = dim3(1); cfg.blockDim = dim3(256); cfg.stream = s; cfg.attrs = &attr; cfg.numAttrs = 1;
  CK(cudaLaunchKernelEx(&cfg, worker, d_out, 0));
  CK(cudaStreamEndCapture(s, &g));
  cudaGraphDeviceNode_t h = attr.val.deviceUpdatableKernelNode.devNode;
  printf("devNode handle = %p\n", (void*)h);
  CK(cudaMemcpy(d_handles, &h, sizeof(h), cudaMemcpyHostToDevice));
  CK(cudaGraphInstantiate(&ge, g, 0));
  int Ls[] = {100, 5000, 70000, 1};
  for (int i = 0; i < 4; ++i) {
    int L = Ls[i]; CK(cudaMemcpy(d_in, &L, 4, cudaMemcpyHostToDevice));
    if (i < 2) CK(cudaGraphUpload(ge, s));          // test: is upload needed every launch?
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    int out[2]; CK(cudaMemcpy(out, d_out, 8, cudaMemcpyDeviceToHost));
    printf("L=%6d upload=%d -> worker saw gridDim.x=%d (expect %d), n=%d (expect %d)  %s\n", L, i<2, out[0], (L+255)/256, out[1], L*3,
           (out[0]==(L+255)/256 && out[1]==L*3) ? "SAME-LAUNCH OK" : "MISMATCH");
  }
  // timing: launch cost with device-side updates (with and without per-launch upload)
  for (int mode = 0; mode < 2; ++mode) {
    CK(cudaStreamSynchronize(s));
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 2000; ++i) { if (mode) CK(cudaGraphUpload(ge, s)); CK(cudaGraphLaunch(ge, s)); }
    CK(cudaStreamSynchronize(s));
    auto t1 = std::chrono::steady_clock::now();
    printf("avg per graph launch (%s): %.2f us\n", mode ? "upload+launch" : "launch only", std::chrono::duration<double, std::micro>(t1 - t0).count() / 2000);
  }
  return 0;
}
