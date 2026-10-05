// Q1: how does a "zero work" kernel behave in a template graph?
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));}}while(0)

__global__ void mark(int* out){ if(threadIdx.x==0) atomicAdd(out,1); }   // counts BLOCKS
__global__ void setgrid(const cudaGraphDeviceNode_t* h,const int* gx){
  if(threadIdx.x==0) cudaGraphKernelNodeSetGridDim(h[0],dim3(gx[0],1,1));
}
__global__ void setenab(const cudaGraphDeviceNode_t* h,const int* en){
  if(threadIdx.x==0) cudaGraphKernelNodeSetEnabled(h[0],en[0]);
}

int main(){
  // --- A: is grid==0 even a legal launch (outside any graph)? ---
  { int* d; CK(cudaMalloc(&d,4)); CK(cudaMemset(d,0,4));
    mark<<<0,32>>>(d);
    printf("A) plain grid-0 launch -> %s\n", cudaGetErrorString(cudaGetLastError()));
    CK(cudaFree(d)); }
  cudaGetLastError();

  int *d; CK(cudaMalloc(&d,4));
  cudaGraphDeviceNode_t* dh; CK(cudaMalloc(&dh,sizeof(cudaGraphDeviceNode_t)));
  int* dv; CK(cudaMalloc(&dv,4));
  cudaStream_t s; CK(cudaStreamCreate(&s));

  // --- B: device grid patch, incl. 0 and above capture-time grid ---
  { cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeRelaxed));
    setgrid<<<1,32,0,s>>>(dh,dv);
    cudaLaunchAttribute a{}; a.id=cudaLaunchAttributeDeviceUpdatableKernelNode;
    a.val.deviceUpdatableKernelNode.deviceUpdatable=1;
    cudaLaunchConfig_t cfg{}; cfg.gridDim=dim3(4); cfg.blockDim=dim3(32); cfg.stream=s; cfg.attrs=&a; cfg.numAttrs=1;
    CK(cudaLaunchKernelEx(&cfg,mark,d));
    CK(cudaStreamEndCapture(s,&g));
    cudaGraphDeviceNode_t h=a.val.deviceUpdatableKernelNode.devNode;
    CK(cudaMemcpy(dh,&h,sizeof(h),cudaMemcpyHostToDevice));
    CK(cudaGraphInstantiate(&ge,g,0));
    printf("\nB) SetGridDim  (captured at grid=4)\n");
    int gxs[]={4,0,3,0,64,1,4096};
    for(int i=0;i<7;++i){
      CK(cudaMemcpy(dv,&gxs[i],4,cudaMemcpyHostToDevice)); CK(cudaMemset(d,0,4));
      CK(cudaGraphLaunch(ge,s)); cudaError_t r=cudaStreamSynchronize(s);
      int c; CK(cudaMemcpy(&c,d,4,cudaMemcpyDeviceToHost));
      printf("   set grid=%4d -> %4d blocks ran   %-14s [%s]\n",gxs[i],c,
        (c==gxs[i])?"as requested":"IGNORED/other",cudaGetErrorString(r));
      if(r!=cudaSuccess) break; } }

  // --- C: SetEnabled as the real "zero work" switch ---
  { cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeRelaxed));
    setenab<<<1,32,0,s>>>(dh,dv);
    cudaLaunchAttribute a{}; a.id=cudaLaunchAttributeDeviceUpdatableKernelNode;
    a.val.deviceUpdatableKernelNode.deviceUpdatable=1;
    cudaLaunchConfig_t cfg{}; cfg.gridDim=dim3(4); cfg.blockDim=dim3(32); cfg.stream=s; cfg.attrs=&a; cfg.numAttrs=1;
    CK(cudaLaunchKernelEx(&cfg,mark,d));
    CK(cudaStreamEndCapture(s,&g));
    cudaGraphDeviceNode_t h=a.val.deviceUpdatableKernelNode.devNode;
    CK(cudaMemcpy(dh,&h,sizeof(h),cudaMemcpyHostToDevice));
    CK(cudaGraphInstantiate(&ge,g,0));
    printf("\nC) SetEnabled  (captured at grid=4)\n");
    int en[]={1,0,0,1};
    for(int i=0;i<4;++i){
      CK(cudaMemcpy(dv,&en[i],4,cudaMemcpyHostToDevice)); CK(cudaMemset(d,0,4));
      CK(cudaGraphLaunch(ge,s)); CK(cudaStreamSynchronize(s));
      int c; CK(cudaMemcpy(&c,d,4,cudaMemcpyDeviceToHost));
      printf("   enabled=%d -> %d blocks ran   %s\n",en[i],c,(c==(en[i]?4:0))?"OK":"** MISMATCH **"); } }

  CK(cudaFree(d)); CK(cudaFree(dh)); CK(cudaFree(dv));
  return 0;
}
