// If host-side patching can fully overlap with GPU execution, then the "O(#nodes) host work" is not on the critical path,
// and our quantitative advantage over the PyTorch route shrinks substantially. Measure it.
#include <cstdio>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("  ERR %s: %s\n",#x,cudaGetErrorString(e)); return -1;}}while(0)
__global__ void spin(int* o, long long cycles){ long long t0=clock64(); while(clock64()-t0<cycles); if(!threadIdx.x&&!blockIdx.x) o[0]=1; }
__global__ void spinB(int* o, long long cycles){ long long t0=clock64(); while(clock64()-t0<cycles); if(!threadIdx.x&&!blockIdx.x) o[0]=2; }
static int* d_o; static cudaStream_t s;
double us(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b){
  return std::chrono::duration<double,std::micro>(b-a).count(); }
int main(){
  CK(cudaMalloc(&d_o,8)); CK(cudaMemset(d_o,0,8)); CK(cudaStreamCreate(&s));
  const int N=3000; const long long CYC=20000;      // each kernel ~20k cycles, so the whole graph has enough GPU time

  cudaGraph_t g; CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
  for(int i=0;i<N;i++) spin<<<1,64,0,s>>>(d_o,CYC);
  CK(cudaStreamEndCapture(s,&g));
  size_t n=0; CK(cudaGraphGetNodes(g,nullptr,&n)); std::vector<cudaGraphNode_t> nd(n); CK(cudaGraphGetNodes(g,nd.data(),&n));
  cudaGraphExec_t ge; CK(cudaGraphInstantiate(&ge,g,0));
  CK(cudaGraphUpload(ge,s)); CK(cudaStreamSynchronize(s));

  // 1. GPU time of the graph itself
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaGraphLaunch(ge,s); CK(cudaStreamSynchronize(s));
  cudaEventRecord(a,s); for(int i=0;i<20;i++) cudaGraphLaunch(ge,s); cudaEventRecord(b,s); CK(cudaStreamSynchronize(s));
  float ms; cudaEventElapsedTime(&ms,a,b); double gpu_us = ms*1000/20;
  printf("graph: %d nodes, GPU execution %.2f ms / launch\n", N, gpu_us/1000);

  long long cyc=CYC; void* argsB[]={&d_o,&cyc};
  cudaKernelNodeParams npB{}; npB.func=(void*)spinB; npB.gridDim=dim3(1); npB.blockDim=dim3(64); npB.kernelParams=argsB;
  cudaKernelNodeParams npA=npB; npA.func=(void*)spin;

  // 2. patch all nodes while the GPU is idle
  CK(cudaStreamSynchronize(s));
  auto t0=std::chrono::steady_clock::now();
  for(size_t i=0;i<n;i++) CK(cudaGraphExecKernelNodeSetParams(ge,nd[i],&npB));
  auto t1=std::chrono::steady_clock::now();
  double idle_patch = us(t0,t1);

  // 3. patch the graph while the GPU is running it -- is that legal? does it block?
  CK(cudaGraphLaunch(ge,s));                       // issued asynchronously; the GPU is now running
  auto t2=std::chrono::steady_clock::now();
  cudaError_t patch_err=cudaSuccess;
  for(size_t i=0;i<n;i++){ cudaError_t e=cudaGraphExecKernelNodeSetParams(ge,nd[i],&npA); if(e!=cudaSuccess){patch_err=e;break;} }
  auto t3=std::chrono::steady_clock::now();
  double inflight_patch = us(t2,t3);
  CK(cudaStreamSynchronize(s));
  auto t4=std::chrono::steady_clock::now();
  int v_inflight; CK(cudaMemcpy(&v_inflight,d_o,4,cudaMemcpyDeviceToHost));

  printf("\npatching all %d nodes:\n", N);
  printf("  GPU idle            %8.1f us  (%.2f us/node)\n", idle_patch, idle_patch/n);
  printf("  while graph running %8.1f us  (%.2f us/node)   %s\n", inflight_patch, inflight_patch/n,
         patch_err==cudaSuccess?"all legal":cudaGetErrorString(patch_err));
  printf("  after the patch returned, waited another %.1f us for the GPU before sync completed\n", us(t3,t4));
  // when that launch was issued all nodes were spinB, so o[0]==2 means "not affected by the in-flight patch"
  printf("  the in-flight launch executed spin%c  %s\n", v_inflight==1?'A':'B',
         v_inflight==2 ? "[not affected, matches the docs]" : "[affected -- dangerous]");

  printf("\nResult: host patch %.1f us vs graph GPU time %.1f us -- %s\n",
         inflight_patch, gpu_us,
         inflight_patch < gpu_us ? "the patch fits inside the GPU execution window and can fully overlap" : "the patch takes longer than GPU execution and cannot be hidden");

  // 4. control: can we double-buffer (two execs) when there are device-updatable nodes
  printf("\nControl: can an ordinary graph be instantiated twice for double buffering\n");
  cudaGraphExec_t ge2; cudaError_t e2 = cudaGraphInstantiate(&ge2,g,0);
  printf("  second instantiate of an ordinary graph: %s  => %s\n", e2==cudaSuccess?"succeeded":cudaGetErrorString(e2),
         e2==cudaSuccess?"the host-patch route can double-buffer and hide the patch completely behind the GPU":"cannot");
  return 0;
}
