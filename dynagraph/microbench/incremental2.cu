#include <cstdio>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("  ERR %s: %s\n",#x,cudaGetErrorString(e)); return -1;}}while(0)
__global__ void kA(int* o){ if(!threadIdx.x&&!blockIdx.x) o[0]=1; }
__global__ void kB(int* o){ if(!threadIdx.x&&!blockIdx.x) o[0]=2; }
__global__ void planner(const cudaGraphDeviceNode_t* h, const int* in){ if(!threadIdx.x) cudaGraphKernelNodeSetGridDim(h[0], dim3(in[0],1,1)); }
__global__ void probe(int* o){ if(!threadIdx.x&&!blockIdx.x) o[1]=gridDim.x; }
static int* d_o; static cudaStream_t s;
double us(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b){ return std::chrono::duration<double,std::micro>(b-a).count(); }
int main(){
  CK(cudaMalloc(&d_o,12)); CK(cudaMemset(d_o,0,12)); CK(cudaStreamCreate(&s));

  printf("=== A. Confirm that swapping the kernel function on the host really takes effect (the previous version had a wrong check)\n");
  { cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal)); for(int i=0;i<500;i++) kA<<<1,64,0,s>>>(d_o);
    CK(cudaStreamEndCapture(s,&g));
    size_t n=0; CK(cudaGraphGetNodes(g,nullptr,&n)); std::vector<cudaGraphNode_t> nd(n); CK(cudaGraphGetNodes(g,nd.data(),&n));
    CK(cudaGraphInstantiate(&ge,g,0)); CK(cudaGraphLaunch(ge,s)); CK(cudaStreamSynchronize(s));
    int v1; CK(cudaMemcpy(&v1,d_o,4,cudaMemcpyDeviceToHost));
    cudaKernelNodeParams np{}; void* ar[]={&d_o}; np.func=(void*)kB; np.gridDim=dim3(1); np.blockDim=dim3(64); np.kernelParams=ar;
    auto t0=std::chrono::steady_clock::now();
    for(size_t i=0;i<n;i++) CK(cudaGraphExecKernelNodeSetParams(ge,nd[i],&np));
    auto t1=std::chrono::steady_clock::now();
    CK(cudaGraphLaunch(ge,s)); CK(cudaStreamSynchronize(s));
    int v2; CK(cudaMemcpy(&v2,d_o,4,cudaMemcpyDeviceToHost));
    printf("   500 nodes kA -> kB: result %d -> %d  %s   took %.1f us (%.2f us/node), no re-instantiate\n",
           v1,v2,(v1==1&&v2==2)?"[took effect]":"[no effect]", us(t0,t1), us(t0,t1)/n);
    // then use ExecUpdate to swap the whole graph back to kA
    cudaGraph_t g2; CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal)); for(int i=0;i<500;i++) kA<<<1,64,0,s>>>(d_o);
    CK(cudaStreamEndCapture(s,&g2));
    cudaGraphExecUpdateResultInfo info{}; auto t2=std::chrono::steady_clock::now();
    cudaError_t e=cudaGraphExecUpdate(ge,g2,&info); auto t3=std::chrono::steady_clock::now();
    CK(cudaGraphLaunch(ge,s)); CK(cudaStreamSynchronize(s)); int v3; CK(cudaMemcpy(&v3,d_o,4,cudaMemcpyDeviceToHost));
    printf("   ExecUpdate whole graph back to kA: %s, result -> %d  %s   took %.1f us (%.2f us/node)\n",
           e==cudaSuccess?"succeeded":"failed", v3, v3==1?"[took effect]":"[no effect]", us(t2,t3), us(t2,t3)/500);
  }

  printf("\n=== B. Fatal limitation: can a graph with device-updatable nodes be re-instantiated\n");
  { cudaGraph_t g; cudaGraphExec_t ge1, ge2;
    cudaGraphDeviceNode_t* d_h; CK(cudaMalloc(&d_h,sizeof(cudaGraphDeviceNode_t)));   // must be outside the capture
    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    planner<<<1,32,0,s>>>(d_h,d_o);
    cudaLaunchAttribute at{}; at.id=cudaLaunchAttributeDeviceUpdatableKernelNode; at.val.deviceUpdatableKernelNode.deviceUpdatable=1;
    cudaLaunchConfig_t cf{}; cf.gridDim=dim3(1); cf.blockDim=dim3(64); cf.stream=s; cf.attrs=&at; cf.numAttrs=1;
    CK(cudaLaunchKernelEx(&cf, probe, d_o));
    CK(cudaStreamEndCapture(s,&g));
    cudaError_t e1 = cudaGraphInstantiate(&ge1,g,0);
    cudaError_t e2 = cudaGraphInstantiate(&ge2,g,0);
    printf("   first instantiate: %s\n   second instantiate: %s\n",
           e1==cudaSuccess?"succeeded":cudaGetErrorString(e1), e2==cudaSuccess?"succeeded":cudaGetErrorString(e2));
    printf("   => %s\n", e2==cudaSuccess?"can be instantiated multiple times" : "a graph with device-updatable nodes can only be instantiated once; changing the topology requires a full recapture");
  }

  printf("\n=== C. Total cost of a full recapture + instantiate (the only way when the topology really changes)\n");
  for(int N : {500, 3000}){
    auto t0=std::chrono::steady_clock::now();
    cudaGraph_t g; CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    for(int i=0;i<N;i++) kA<<<1,64,0,s>>>(d_o);
    CK(cudaStreamEndCapture(s,&g));
    auto t1=std::chrono::steady_clock::now();
    cudaGraphExec_t ge; CK(cudaGraphInstantiate(&ge,g,0));
    auto t2=std::chrono::steady_clock::now();
    CK(cudaGraphUpload(ge,s)); CK(cudaStreamSynchronize(s));
    auto t3=std::chrono::steady_clock::now();
    printf("   N=%-5d  capture %7.1f us + instantiate %8.1f us + upload %7.1f us = %8.2f ms total\n",
           N, us(t0,t1), us(t1,t2), us(t2,t3), us(t0,t3)/1000);
  }

  printf("\n=== D. Cost of empty SWITCH bodies (measured repeatedly, min taken)\n");
  for(int V : {2,4,8,16,32}){
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    cudaGraph_t cap; const cudaGraphNode_t* dp; size_t nd; cudaStreamCaptureStatus st;
    CK(cudaStreamGetCaptureInfo(s,&st,nullptr,&cap,&dp,nullptr,&nd));
    cudaGraphConditionalHandle ch; CK(cudaGraphConditionalHandleCreate(&ch,cap,0,cudaGraphCondAssignDefault));
    kA<<<1,32,0,s>>>(d_o);
    CK(cudaStreamGetCaptureInfo(s,&st,nullptr,&cap,&dp,nullptr,&nd));
    cudaGraphNodeParams np{}; np.type=cudaGraphNodeTypeConditional; np.conditional.handle=ch;
    np.conditional.type=cudaGraphCondTypeSwitch; np.conditional.size=V;
    cudaGraphNode_t cn; CK(cudaGraphAddNode(&cn,cap,dp,nullptr,nd,&np));
    for(int v=0;v<2;v++){ cudaStream_t bs; CK(cudaStreamCreate(&bs));
      CK(cudaStreamBeginCaptureToGraph(bs,np.conditional.phGraph_out[v],nullptr,nullptr,0,cudaStreamCaptureModeGlobal));
      kA<<<1,64,0,bs>>>(d_o); cudaGraph_t tmp; CK(cudaStreamEndCapture(bs,&tmp)); }
    CK(cudaStreamUpdateCaptureDependencies(s,&cn,nullptr,1,cudaStreamSetCaptureDependencies));
    CK(cudaStreamEndCapture(s,&g)); CK(cudaGraphInstantiate(&ge,g,0));
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float best=1e9;
    for(int r=0;r<5;r++){ cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
      cudaEventRecord(a,s); for(int i=0;i<300;i++) cudaGraphLaunch(ge,s); cudaEventRecord(b,s); cudaStreamSynchronize(s);
      float ms; cudaEventElapsedTime(&ms,a,b); if(ms*1000/300<best) best=ms*1000/300; }
    printf("   SWITCH size=%-3d (only 2 bodies filled, rest empty): %6.2f us / launch\n", V, best);
  }
  return 0;
}
