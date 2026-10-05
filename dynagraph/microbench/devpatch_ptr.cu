// Can the planner patch **pointer** arguments on the device side?
//
// This is the precondition for the whole "re-lay out buffers on the device" path. Scenario: the total is fixed but each sample differs
// (variable-length sequences packed to a token budget), so two buffers change in opposite directions,
// no single "max shape" covers both at once, and record-at-max breaks down.
//
// The way out is to have the planner re-lay out each buffer inside one arena at replay time according to the runtime sizes,
// then patch each kernel's pointer arguments to the new locations. Order and lifetime reuse are fixed at compile time,
// so it is bump allocation rather than a general-purpose allocator.
//
// But that requires pointer arguments to really be patchable within the same launch. Scalars are known to work (measured);
// pointers are 8 bytes and the driver may validate them differently, so they must be verified separately.
//
// Build: nvcc -arch=sm_90a -o devpatch_ptr devpatch_ptr.cu

#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 1;}}while(0)

__global__ void writer(int* out, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = val + i;
}

// Param layout: out@0(8) val@8(4) n@12(4)
__global__ void planner(const cudaGraphDeviceNode_t* h, void* const* new_ptr,
                        const int* new_n) {
  if (threadIdx.x) return;
  // patch the pointer (8 bytes)
  void* p = *new_ptr;
  cudaGraphKernelNodeSetParam(h[0], 0, &p, sizeof(void*));
  // also patch a scalar as a control
  int n = *new_n;
  cudaGraphKernelNodeSetParam(h[0], 12, &n, sizeof(int));
  cudaGraphKernelNodeSetGridDim(h[0], dim3((n + 63) / 64, 1, 1));
}

int main() {
  const int N = 256;
  int *bufA, *bufB;
  void** d_ptr; int* d_n;
  cudaGraphDeviceNode_t* d_h;
  CK(cudaMalloc(&bufA, N * 4)); CK(cudaMalloc(&bufB, N * 4));
  CK(cudaMalloc(&d_ptr, sizeof(void*))); CK(cudaMalloc(&d_n, 4));
  CK(cudaMalloc(&d_h, sizeof(cudaGraphDeviceNode_t)));
  CK(cudaMemset(bufA, 0, N * 4)); CK(cudaMemset(bufB, 0, N * 4));

  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaGraph_t g; cudaGraphExec_t ge;

  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  planner<<<1, 32, 0, s>>>(d_h, d_ptr, d_n);
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeDeviceUpdatableKernelNode;
  attr.val.deviceUpdatableKernelNode.deviceUpdatable = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(N / 64); cfg.blockDim = dim3(64);
  cfg.stream = s; cfg.attrs = &attr; cfg.numAttrs = 1;
  // at capture time it writes bufA
  CK(cudaLaunchKernelEx(&cfg, writer, bufA, 1000, N));
  CK(cudaStreamEndCapture(s, &g));

  cudaGraphDeviceNode_t h = attr.val.deviceUpdatableKernelNode.devNode;
  CK(cudaMemcpy(d_h, &h, sizeof(h), cudaMemcpyHostToDevice));
  CK(cudaGraphInstantiate(&ge, g, 0));

  auto run = [&](void* target, int n, const char* label) -> int {
    CK(cudaMemcpy(d_ptr, &target, sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_n, &n, 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(bufA, 0, N * 4)); CK(cudaMemset(bufB, 0, N * 4));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    int a[4], b[4];
    CK(cudaMemcpy(a, bufA, 16, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b, bufB, 16, cudaMemcpyDeviceToHost));
    printf("  %-28s n=%-4d bufA[0..3]={%d,%d,%d,%d}  bufB[0..3]={%d,%d,%d,%d}\n",
           label, n, a[0],a[1],a[2],a[3], b[0],b[1],b[2],b[3]);
    return 0;
  };

  printf("At capture time it writes bufA. Now have the planner change the output pointer:\n");
  if (run(bufA, N, "points to bufA (control)")) return 1;
  if (run(bufB, N, "redirected to bufB")) return 1;
  if (run(bufB, 64, "redirected to bufB, n=64")) return 1;

  int b0[4];
  CK(cudaMemcpy(b0, bufB, 16, cudaMemcpyDeviceToHost));
  printf("\n%s\n", b0[0] == 1000
      ? "Pointer params can be patched on the device -- the arena re-layout path is feasible."
      : "The pointer was not patched; device-side re-layout is not feasible, another approach is needed.");
  return 0;
}
