// SWITCH conditional node inserted DURING stream capture; condition chosen by an upstream "planner" kernel on the device.
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 1;}}while(0)
__global__ void planner(cudaGraphConditionalHandle h, const int* in) { if (threadIdx.x==0) { int L = in[0]; cudaGraphSetConditional(h, L <= 512 ? 0 : (L <= 2048 ? 1 : 2)); } }
__global__ void variant(int* out, int id) { if (threadIdx.x==0) out[0] = id * 1000 + gridDim.x; }
__global__ void tail(int* out) { if (threadIdx.x==0) out[1] = 77; }
int main() {
  int *d_in, *d_out; CK(cudaMalloc(&d_in, 4)); CK(cudaMalloc(&d_out, 8)); CK(cudaMemset(d_out, 0, 8));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  cudaGraph_t capGraph; const cudaGraphNode_t* deps; size_t ndeps; cudaStreamCaptureStatus st;
  CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &capGraph, &deps, nullptr, &ndeps));
  cudaGraphConditionalHandle h; CK(cudaGraphConditionalHandleCreate(&h, capGraph, 0, cudaGraphCondAssignDefault));
  planner<<<1, 32, 0, s>>>(h, d_in);
  CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &capGraph, &deps, nullptr, &ndeps));
  cudaGraphNodeParams p = {}; p.type = cudaGraphNodeTypeConditional; p.conditional.handle = h; p.conditional.type = cudaGraphCondTypeSwitch; p.conditional.size = 3;
  cudaGraphNode_t cnode; CK(cudaGraphAddNode(&cnode, capGraph, deps, nullptr, ndeps, &p));
  for (int v = 0; v < 3; ++v) {   // each body: a different kernel variant with a different launch geometry
    cudaGraph_t body = p.conditional.phGraph_out[v];
    cudaStream_t bs; CK(cudaStreamCreate(&bs));
    CK(cudaStreamBeginCaptureToGraph(bs, body, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
    variant<<<(v + 1) * 4, 64, 0, bs>>>(d_out, v);
    CK(cudaStreamEndCapture(bs, &body));
  }
  CK(cudaStreamUpdateCaptureDependencies(s, &cnode, nullptr, 1, cudaStreamSetCaptureDependencies));
  tail<<<1, 32, 0, s>>>(d_out);
  CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  int Ls[] = {100, 1000, 9000, 512, 2049};
  for (int L : Ls) {
    CK(cudaMemcpy(d_in, &L, 4, cudaMemcpyHostToDevice)); CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    int out[2]; CK(cudaMemcpy(out, d_out, 8, cudaMemcpyDeviceToHost));
    int ev = L <= 512 ? 0 : (L <= 2048 ? 1 : 2);
    printf("L=%5d -> variant %d ran with grid %d (expect variant %d grid %d), tail=%d  %s\n", L, out[0]/1000, out[0]%1000, ev, (ev+1)*4, out[1], (out[0]==ev*1000+(ev+1)*4 && out[1]==77)?"OK":"MISMATCH");
  }
  CK(cudaStreamSynchronize(s));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0, s);
  for (int i = 0; i < 2000; ++i) CK(cudaGraphLaunch(ge, s));
  cudaEventRecord(e1, s); CK(cudaStreamSynchronize(s)); float ms; cudaEventElapsedTime(&ms, e0, e1);
  printf("GPU time per launch of [planner -> SWITCH(3) -> tail]: %.2f us\n", ms * 1000 / 2000);
  return 0;
}
