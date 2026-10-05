// Does exec-graph func swap survive a large jump in launch resources
// (threads, dynamic smem, register pressure) vs what was reserved at instantiate?
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s: %s\n",#x,cudaGetErrorString(e)); return 1;}}while(0)
__global__ void tiny(int* o){ if(!threadIdx.x) o[0]=1; }          // 1 thread, 0 smem, few regs
extern __shared__ float sh[];
__global__ __launch_bounds__(1024) void fat(int* o, int n){       // 1024 thr, big smem, many regs
  float acc[32];
  #pragma unroll
  for(int i=0;i<32;i++) acc[i]=threadIdx.x*i*0.5f;
  sh[threadIdx.x]=acc[threadIdx.x%32];
  __syncthreads();
  float s=0; for(int i=0;i<32;i++) s+=acc[i]+sh[(threadIdx.x+i)%blockDim.x];
  if(threadIdx.x==0 && blockIdx.x==0){ o[0]=2; o[1]=(int)s; o[2]=blockDim.x;
    unsigned sm; asm volatile("mov.u32 %0, %%dynamic_smem_size;":"=r"(sm)); o[3]=(int)sm; }
}
static CUfunction funcof(void(*k)(int*,int), int* d){
  cudaStream_t s; cudaStreamCreate(&s);
  cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
  k<<<1,1024,40960,s>>>(d,0);
  cudaGraph_t g; cudaStreamEndCapture(s,&g);
  size_t n=0; cuGraphGetNodes((CUgraph)g,nullptr,&n); CUgraphNode nd[4];
  cuGraphGetNodes((CUgraph)g,nd,&n);
  CUDA_KERNEL_NODE_PARAMS p{}; cuGraphKernelNodeGetParams(nd[0],&p); return p.func;
}
int main(){
  int* d; CK(cudaMalloc(&d,8*sizeof(int))); int h[8];
  CUfunction ffat = funcof(fat, d);
  cudaStream_t s; CK(cudaStreamCreate(&s));
  CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeRelaxed));
  tiny<<<1,1,0,s>>>(d);                       // reserve almost nothing
  cudaGraph_t g; CK(cudaStreamEndCapture(s,&g));
  cudaGraphExec_t ex; CK(cudaGraphInstantiate(&ex,g,0));
  CK(cudaMemset(d,0,8*sizeof(int))); CK(cudaGraphLaunch(ex,s)); CK(cudaStreamSynchronize(s));
  CK(cudaMemcpy(h,d,8*sizeof(int),cudaMemcpyDeviceToHost));
  printf("baseline (tiny, 1 thread, 0 smem): marker=%d\n",h[0]);
  size_t n=0; cuGraphGetNodes((CUgraph)g,nullptr,&n); CUgraphNode nd[4];
  cuGraphGetNodes((CUgraph)g,nd,&n);
  CUDA_KERNEL_NODE_PARAMS p{}; cuGraphKernelNodeGetParams(nd[0],&p);
  int nn=0; void* args[2]={&d,&nn};
  p.func=ffat; p.gridDimX=4; p.blockDimX=1024; p.sharedMemBytes=40960;
  p.kernelParams=args; p.extra=nullptr;
  CUresult r=cuGraphExecKernelNodeSetParams((CUgraphExec)ex,nd[0],&p);
  const char* es=nullptr; cuGetErrorString(r,&es);
  printf("swap tiny->fat (1024 thr, 40KB dyn smem, 4 blocks) -> %d (%s)\n",(int)r,es?es:"?");
  if(r==CUDA_SUCCESS){
    CK(cudaMemset(d,0,8*sizeof(int)));
    cudaError_t le = cudaGraphLaunch(ex,s);
    cudaError_t se = cudaStreamSynchronize(s);
    printf("  launch=%s sync=%s\n",cudaGetErrorString(le),cudaGetErrorString(se));
    if(!le&&!se){ CK(cudaMemcpy(h,d,8*sizeof(int),cudaMemcpyDeviceToHost));
      printf("  marker=%d blockDim=%d dynsmem=%d -> %s\n",h[0],h[2],h[3],
        (h[0]==2&&h[2]==1024&&h[3]==40960)?"*** RESOURCE JUMP SURVIVED ***":"mismatch"); }
  }
  return 0;
}
