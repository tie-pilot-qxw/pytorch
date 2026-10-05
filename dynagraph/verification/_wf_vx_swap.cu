// INDEPENDENT re-test of "cuGraphExecKernelNodeSetParams can change .func".
// Harsher than the original: different ARITY (2 -> 6 params), different block,
// different dynamic smem, and a value only kB can produce. Plus swap-back.
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s: %s\n",#x,cudaGetErrorString(e)); return 1;}}while(0)
#define DR(x) do{CUresult r=(x); if(r){const char*s=0;cuGetErrorString(r,&s);printf("DRV ERR %s: %s\n",#x,s?s:"?"); }}while(0)

// 2 params, tiny launch
__global__ void kA(int* out, int v){
  if(!threadIdx.x && !blockIdx.x){ out[0]=1; out[1]=v; out[2]=blockDim.x; out[3]=gridDim.x; out[4]=0; }
}
// 6 params, needs dynamic smem, uses a register array
__global__ void kB(int* out, int a, int b, int c, long long d, float e){
  extern __shared__ int sh[];
  float acc[32];
  for(int i=0;i<32;i++) acc[i]=e*(i+1);
  sh[threadIdx.x]=a;
  __syncthreads();
  unsigned dyn; asm volatile("mov.u32 %0, %%dynamic_smem_size;":"=r"(dyn));
  if(!threadIdx.x && !blockIdx.x){
    out[0]=2; out[1]=sh[0]+b*1000+c*1000000; out[2]=blockDim.x; out[3]=gridDim.x;
    out[4]=(int)dyn; out[5]=(int)d; out[6]=(int)acc[31];
  }
}
int main(){
  int* d; CK(cudaMalloc(&d,16*sizeof(int))); int h[16];
  cudaStream_t s; CK(cudaStreamCreate(&s));
  CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeRelaxed));
  kA<<<1,1,0,s>>>(d,7);
  cudaGraph_t g; CK(cudaStreamEndCapture(s,&g));
  cudaGraphExec_t ex; CK(cudaGraphInstantiate(&ex,g,0));
  CK(cudaMemset(d,0,16*sizeof(int))); CK(cudaGraphLaunch(ex,s)); CK(cudaStreamSynchronize(s));
  CK(cudaMemcpy(h,d,16*sizeof(int),cudaMemcpyDeviceToHost));
  printf("[baseline kA] marker=%d val=%d block=%d grid=%d  (expect 1 7 1 1)\n",h[0],h[1],h[2],h[3]);

  size_t n=0; DR(cuGraphGetNodes((CUgraph)g,nullptr,&n));
  printf("[graph] %zu node(s)\n",n);
  CUgraphNode nd[8]; DR(cuGraphGetNodes((CUgraph)g,nd,&n));

  // stable identity: names, not pointers
  CUDA_KERNEL_NODE_PARAMS p{}; DR(cuGraphKernelNodeGetParams(nd[0],&p));
  const char* nmA=0; DR(cuFuncGetName(&nmA,p.func));
  CUfunction fB=nullptr; CK(cudaGetFuncBySymbol((cudaFunction_t*)&fB,(const void*)kB));
  const char* nmB=0; DR(cuFuncGetName(&nmB,fB));
  size_t na=0,nb=0,o,z;
  while(cuFuncGetParamInfo(p.func,na,&o,&z)==CUDA_SUCCESS) na++;
  while(cuFuncGetParamInfo(fB,nb,&o,&z)==CUDA_SUCCESS) nb++;
  printf("[identity] node func name = '%s' (%zu params); target = '%s' (%zu params)\n",nmA,na,nmB,nb);

  int a=42,b=3,c=5; long long dd=123456789; float e=2.0f;
  void* args[6]={&d,&a,&b,&c,&dd,&e};
  CUDA_KERNEL_NODE_PARAMS q = p;
  q.func=fB; q.kern=nullptr; q.kernelParams=args; q.extra=nullptr;
  q.gridDimX=4; q.gridDimY=1; q.gridDimZ=1;
  q.blockDimX=256; q.blockDimY=1; q.blockDimZ=1;
  q.sharedMemBytes=8192;
  CUresult r=cuGraphExecKernelNodeSetParams((CUgraphExec)ex,nd[0],&q);
  const char* es=0; cuGetErrorString(r,&es);
  printf("[swap A->B] SetParams -> %d (%s)\n",(int)r,es?es:"?");
  if(r==CUDA_SUCCESS){
    CK(cudaMemset(d,0,16*sizeof(int))); CK(cudaGraphLaunch(ex,s));
    cudaError_t le=cudaStreamSynchronize(s);
    printf("[swap A->B] launch/sync -> %s\n",cudaGetErrorString(le));
    CK(cudaMemcpy(h,d,16*sizeof(int),cudaMemcpyDeviceToHost));
    int want1=42+3*1000+5*1000000;
    printf("[after swap] marker=%d val=%d(want %d) block=%d(want 256) grid=%d(want 4) dynsmem=%d(want 8192) d=%d(want %d) acc31=%d(want 64)\n",
      h[0],h[1],want1,h[2],h[3],h[4],h[5],(int)dd,h[6]);
    printf("[VERDICT swap] %s\n", (h[0]==2&&h[1]==want1&&h[2]==256&&h[3]==4&&h[4]==8192&&h[6]==64)
      ? "kB REALLY RAN with new arity/block/smem" : (h[0]==1?"kA still ran (swap ignored)":"kB ran but values WRONG"));
  }
  // swap back
  int v2=99; void* argsA[2]={&d,&v2};
  CUDA_KERNEL_NODE_PARAMS p2 = p; p2.kern=nullptr; p2.kernelParams=argsA; p2.extra=nullptr;
  r=cuGraphExecKernelNodeSetParams((CUgraphExec)ex,nd[0],&p2);
  cuGetErrorString(r,&es); printf("[swap B->A] SetParams -> %d (%s)\n",(int)r,es?es:"?");
  if(r==CUDA_SUCCESS){
    CK(cudaMemset(d,0,16*sizeof(int))); CK(cudaGraphLaunch(ex,s)); CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(h,d,16*sizeof(int),cudaMemcpyDeviceToHost));
    printf("[after swap-back] marker=%d val=%d (want 1 99) -> %s\n",h[0],h[1],(h[0]==1&&h[1]==99)?"REVERSIBLE":"NOT reversible");
  }
  // source graph untouched?
  CUDA_KERNEL_NODE_PARAMS p3{}; DR(cuGraphKernelNodeGetParams(nd[0],&p3));
  const char* nm3=0; DR(cuFuncGetName(&nm3,p3.func));
  printf("[source graph] func='%s' grid=%u block=%u smem=%u (want kA 1 1 0)\n",nm3,p3.gridDimX,p3.blockDimX,p3.sharedMemBytes);
  return 0;
}
