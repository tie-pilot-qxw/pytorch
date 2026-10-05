// Variant dispatch inside an immutable graph: pre-place V variant nodes and have the planner
// ENABLE exactly one (disabled nodes are NOPs), vs a SWITCH conditional node.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return -1;}}while(0)
__global__ void variant(int* out, int id) { if (threadIdx.x==0 && blockIdx.x==0) out[0] = id; }
__global__ void planner_enable(const cudaGraphDeviceNode_t* h, int V, int nop, const int* in) {
  int pick = in[0];
  for (int i = threadIdx.x; i < V*nop; i += blockDim.x) cudaGraphKernelNodeSetEnabled(h[i], (i % V) == pick);
}
__global__ void setcond(cudaGraphConditionalHandle h, const int* in) { if (threadIdx.x==0) cudaGraphSetConditional(h, in[0]); }
static int *d_in,*d_out; static cudaStream_t s;
float timeit(cudaGraphExec_t ge){ cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaGraphLaunch(ge,s); cudaStreamSynchronize(s); cudaEventRecord(a,s);
  for(int i=0;i<500;i++) cudaGraphLaunch(ge,s); cudaEventRecord(b,s); cudaStreamSynchronize(s);
  float ms; cudaEventElapsedTime(&ms,a,b); return ms*1000/500; }
int main(){
  CK(cudaMalloc(&d_in,4)); CK(cudaMalloc(&d_out,4)); CK(cudaStreamCreate(&s));
  int pick=1; CK(cudaMemcpy(d_in,&pick,4,cudaMemcpyHostToDevice));
  const int NOP = 40;                        // 40 dispatch points, like 40 GEMMs in a layer stack
  for (int V : {1, 3, 5, 8}) {
    // --- A: pre-placed variants + device SetEnabled
    cudaGraph_t g; cudaGraphExec_t ge; std::vector<cudaGraphDeviceNode_t> hs;
    cudaGraphDeviceNode_t* d_h; CK(cudaMalloc(&d_h, sizeof(cudaGraphDeviceNode_t)*V*NOP));
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    if (V > 1) planner_enable<<<1,256,0,s>>>(d_h, V, NOP, d_in);
    for (int p=0;p<NOP;p++) for (int v=0;v<V;v++) {
      cudaLaunchAttribute at{}; at.id=cudaLaunchAttributeDeviceUpdatableKernelNode; at.val.deviceUpdatableKernelNode.deviceUpdatable=1;
      cudaLaunchConfig_t cf{}; cf.gridDim=dim3(1); cf.blockDim=dim3(128); cf.stream=s; cf.attrs=&at; cf.numAttrs=(V>1)?1:0;
      CK(cudaLaunchKernelEx(&cf, variant, d_out, v));
      if (V>1) hs.push_back(at.val.deviceUpdatableKernelNode.devNode);
    }
    CK(cudaStreamEndCapture(s,&g));
    if (V>1) CK(cudaMemcpy(d_h, hs.data(), sizeof(cudaGraphDeviceNode_t)*hs.size(), cudaMemcpyHostToDevice));
    CK(cudaGraphInstantiate(&ge,g,0)); CK(cudaGraphUpload(ge,s));
    float tA = timeit(ge);
    int got; CK(cudaMemcpy(&got,d_out,4,cudaMemcpyDeviceToHost));
    // --- B: SWITCH conditional nodes
    float tB = -1;
    if (V > 1) {
      cudaGraph_t g2; cudaGraphExec_t ge2;
      CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
      for (int p=0;p<NOP;p++){
        cudaGraph_t cap; const cudaGraphNode_t* dp; size_t nd; cudaStreamCaptureStatus st;
        CK(cudaStreamGetCaptureInfo(s,&st,nullptr,&cap,&dp,nullptr,&nd));
        cudaGraphConditionalHandle ch; CK(cudaGraphConditionalHandleCreate(&ch,cap,0,cudaGraphCondAssignDefault));
        setcond<<<1,32,0,s>>>(ch,d_in);
        CK(cudaStreamGetCaptureInfo(s,&st,nullptr,&cap,&dp,nullptr,&nd));
        cudaGraphNodeParams np={}; np.type=cudaGraphNodeTypeConditional; np.conditional.handle=ch; np.conditional.type=cudaGraphCondTypeSwitch; np.conditional.size=V;
        cudaGraphNode_t cn; CK(cudaGraphAddNode(&cn,cap,dp,nullptr,nd,&np));
        for (int v=0;v<V;v++){ cudaGraph_t bd=np.conditional.phGraph_out[v]; cudaStream_t bs; CK(cudaStreamCreate(&bs));
          CK(cudaStreamBeginCaptureToGraph(bs,bd,nullptr,nullptr,0,cudaStreamCaptureModeGlobal));
          variant<<<1,128,0,bs>>>(d_out,v); CK(cudaStreamEndCapture(bs,&bd)); }
        CK(cudaStreamUpdateCaptureDependencies(s,&cn,nullptr,1,cudaStreamSetCaptureDependencies));
      }
      CK(cudaStreamEndCapture(s,&g2)); CK(cudaGraphInstantiate(&ge2,g2,0)); tB = timeit(ge2);
    }
    printf("V=%d variants x %d dispatch points: SetEnabled %7.1f us (%5.2f us/point, picked variant %d) | SWITCH %7.1f us (%5.2f us/point)\n",
           V, NOP, tA, tA/NOP, got, tB, tB<0?0:tB/NOP);
  }
  return 0;
}
