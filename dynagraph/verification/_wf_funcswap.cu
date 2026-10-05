// Can cuGraphExecKernelNodeSetParams swap the *function* of a kernel node?
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s: %s\n",#x,cudaGetErrorString(e)); return 1;}}while(0)
__global__ void kA(int* out, int v){ if(!threadIdx.x&&!blockIdx.x){ out[0]=1; out[1]=v; } }
__global__ void kB(int* out, int v){ if(!threadIdx.x&&!blockIdx.x){ out[0]=2; out[1]=v*10; } }
int main(){
  int* d; CK(cudaMalloc(&d,8*sizeof(int))); int h[8];
  cudaStream_t s; CK(cudaStreamCreate(&s));
  CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeRelaxed));
  kA<<<1,1,0,s>>>(d,7);
  cudaGraph_t g; CK(cudaStreamEndCapture(s,&g));
  cudaGraphExec_t ex; CK(cudaGraphInstantiate(&ex,g,0));
  CK(cudaMemset(d,0,8*sizeof(int))); CK(cudaGraphLaunch(ex,s)); CK(cudaStreamSynchronize(s));
  CK(cudaMemcpy(h,d,8*sizeof(int),cudaMemcpyDeviceToHost));
  printf("baseline: marker=%d val=%d  (kA -> 1,7)\n",h[0],h[1]);

  size_t n=0; cuGraphGetNodes((CUgraph)g,nullptr,&n);
  CUgraphNode nd[4]; cuGraphGetNodes((CUgraph)g,nd,&n);
  CUDA_KERNEL_NODE_PARAMS p{}; cuGraphKernelNodeGetParams(nd[0],&p);
  // Build params naming kB instead
  CUfunction fB=nullptr;
  // get the CUfunction for kB by capturing a throwaway launch
  cudaStream_t s2; CK(cudaStreamCreate(&s2));
  CK(cudaStreamBeginCapture(s2,cudaStreamCaptureModeRelaxed));
  kB<<<1,1,0,s2>>>(d,7);
  cudaGraph_t g2; CK(cudaStreamEndCapture(s2,&g2));
  size_t n2=0; cuGraphGetNodes((CUgraph)g2,nullptr,&n2); CUgraphNode nd2[4];
  cuGraphGetNodes((CUgraph)g2,nd2,&n2);
  CUDA_KERNEL_NODE_PARAMS p2{}; cuGraphKernelNodeGetParams(nd2[0],&p2);
  fB = p2.func;
  printf("funcA=%p funcB=%p (distinct=%d)\n",(void*)p.func,(void*)fB,p.func!=fB);

  int v=7; void* args[2]={&d,&v};
  p.func=fB; p.kernelParams=args; p.extra=nullptr;
  CUresult r=cuGraphExecKernelNodeSetParams((CUgraphExec)ex,nd[0],&p);
  const char* es=nullptr; cuGetErrorString(r,&es);
  printf("SetParams with func=kB -> %d (%s)\n",(int)r,es?es:"?");
  if(r==CUDA_SUCCESS){
    CK(cudaMemset(d,0,8*sizeof(int)));
    CK(cudaGraphLaunch(ex,s)); CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(h,d,8*sizeof(int),cudaMemcpyDeviceToHost));
    printf("after func swap: marker=%d val=%d   -> %s\n",h[0],h[1],
       h[0]==2? "*** kB ACTUALLY RAN: func IS swappable ***" :
      (h[0]==1? "kA still ran: func change SILENTLY IGNORED" : "neither?!"));
  }
  return 0;
}
