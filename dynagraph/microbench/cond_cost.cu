// isolate the runtime cost of a conditional SWITCH node vs a plain kernel chain
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 1;}}while(0)
__global__ void planner(cudaGraphConditionalHandle h, const int* in) { if (threadIdx.x==0) cudaGraphSetConditional(h, in[0]); }
__global__ void noop(int* out) { if (threadIdx.x==0) out[0] += 1; }
static int *d_in, *d_out; static cudaStream_t s;
int build(int nswitch, int body_len, int nbodies, cudaGraphExec_t* ge) {
  cudaGraph_t g; CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int k = 0; k < nswitch; ++k) {
    cudaGraph_t cap; const cudaGraphNode_t* deps; size_t nd; cudaStreamCaptureStatus st;
    if (nbodies > 0) {
      CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &cap, &deps, nullptr, &nd));
      cudaGraphConditionalHandle h; CK(cudaGraphConditionalHandleCreate(&h, cap, 0, cudaGraphCondAssignDefault));
      planner<<<1, 32, 0, s>>>(h, d_in);
      CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &cap, &deps, nullptr, &nd));
      cudaGraphNodeParams p = {}; p.type = cudaGraphNodeTypeConditional; p.conditional.handle = h; p.conditional.type = cudaGraphCondTypeSwitch; p.conditional.size = nbodies;
      cudaGraphNode_t cn; CK(cudaGraphAddNode(&cn, cap, deps, nullptr, nd, &p));
      for (int v = 0; v < nbodies; ++v) { cudaGraph_t body = p.conditional.phGraph_out[v]; cudaStream_t bs; CK(cudaStreamCreate(&bs));
        CK(cudaStreamBeginCaptureToGraph(bs, body, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
        for (int j = 0; j < body_len; ++j) noop<<<1, 32, 0, bs>>>(d_out);
        CK(cudaStreamEndCapture(bs, &body)); }
      CK(cudaStreamUpdateCaptureDependencies(s, &cn, nullptr, 1, cudaStreamSetCaptureDependencies));
    } else { noop<<<1, 32, 0, s>>>(d_out); for (int j = 0; j < body_len; ++j) noop<<<1, 32, 0, s>>>(d_out); }   // plain chain of equal length
  }
  CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(ge, g, 0)); return 0;
}
float timeit(cudaGraphExec_t ge) { cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
  cudaEventRecord(e0, s); for (int i = 0; i < 500; ++i) cudaGraphLaunch(ge, s); cudaEventRecord(e1, s); cudaStreamSynchronize(s); float ms; cudaEventElapsedTime(&ms, e0, e1); return ms * 1000 / 500; }
int main() {
  CK(cudaMalloc(&d_in, 4)); CK(cudaMalloc(&d_out, 4)); CK(cudaStreamCreate(&s)); int one = 1; CK(cudaMemcpy(d_in, &one, 4, cudaMemcpyHostToDevice));
  cudaGraphExec_t ge;
  for (int body_len : {1, 8}) for (int nsw : {1, 16}) {
    build(nsw, body_len, 0, &ge); float tp = timeit(ge);
    build(nsw, body_len, 3, &ge); float tc = timeit(ge);
    printf("%2d x [planner -> %d-kernel body]: plain chain %7.1f us | with SWITCH(3) %7.1f us | per-SWITCH overhead %.1f us\n", nsw, body_len, tp, tc, (tc - tp) / nsw);
  }
  return 0;
}
