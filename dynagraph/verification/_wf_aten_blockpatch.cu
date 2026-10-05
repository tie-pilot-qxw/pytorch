// Decisive test: does cudaGraphExecKernelNodeSetParams honor changes to
// blockDim and sharedMemBytes (and gridDim) on an already-instantiated graph?
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 1;}}while(0)
#define DR(x) do{CUresult r=(x); if(r){const char*s;cuGetErrorString(r,&s);printf("DRV ERR %s @%d: %s\n",#x,__LINE__,s); return 1;}}while(0)

extern __shared__ char dyn[];
__global__ void probe(int* out, int arg) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = blockDim.x;
    out[1] = gridDim.x;
    out[2] = arg;
    unsigned int smem;                 // dynamic smem size, sm_90 special reg
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(smem));
    out[3] = (int)smem;
  }
}

int main() {
  int* d; CK(cudaMalloc(&d, 16*sizeof(int))); CK(cudaMemset(d,0,16*sizeof(int)));
  int h[16];

  cudaStream_t s; CK(cudaStreamCreate(&s));
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
  probe<<<3, 32, 0, s>>>(d, 111);
  cudaGraph_t g; CK(cudaStreamEndCapture(s, &g));
  cudaGraphExec_t ex; CK(cudaGraphInstantiate(&ex, g, 0));

  CK(cudaGraphLaunch(ex, s)); CK(cudaStreamSynchronize(s));
  CK(cudaMemcpy(h,d,16*sizeof(int),cudaMemcpyDeviceToHost));
  printf("baseline      : block=%d grid=%d arg=%d dynsmem=%d   (expect 32 3 111 0)\n",h[0],h[1],h[2],h[3]);

  // Fetch the node and patch grid/block/smem/args
  size_t n=0; DR(cuGraphGetNodes((CUgraph)g, nullptr, &n));
  CUgraphNode nodes[8]; DR(cuGraphGetNodes((CUgraph)g, nodes, &n));
  printf("nodes in graph: %zu\n", n);
  CUDA_KERNEL_NODE_PARAMS p{};
  DR(cuGraphKernelNodeGetParams(nodes[0], &p));
  printf("recorded      : grid=%u block=%u smem=%u\n", p.gridDimX, p.blockDimX, p.sharedMemBytes);

  int newarg = 222;
  void* args[2] = { &d, &newarg };
  p.gridDimX = 5; p.blockDimX = 64; p.sharedMemBytes = 1024;
  p.kernelParams = args; p.extra = nullptr;
  CUresult r = cuGraphExecKernelNodeSetParams((CUgraphExec)ex, nodes[0], &p);
  const char* es=nullptr; cuGetErrorString(r,&es);
  printf("SetParams(grid5 block64 smem1024 arg222) -> %d (%s)\n", (int)r, es?es:"?");

  if (r == CUDA_SUCCESS) {
    CK(cudaMemset(d,0,16*sizeof(int)));
    CK(cudaGraphLaunch(ex, s)); CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(h,d,16*sizeof(int),cudaMemcpyDeviceToHost));
    printf("after patch   : block=%d grid=%d arg=%d dynsmem=%d   (want 64 5 222 1024)\n",h[0],h[1],h[2],h[3]);
    printf("  gridDim  patch: %s\n", h[1]==5   ? "APPLIED" : "IGNORED");
    printf("  blockDim patch: %s\n", h[0]==64  ? "APPLIED" : "IGNORED");
    printf("  args     patch: %s\n", h[2]==222 ? "APPLIED" : "IGNORED");
    printf("  smem     patch: %s\n", h[3]==1024? "APPLIED" : "IGNORED");
  }

  // Also: what does cuGraphKernelNodeGetParams on the *original graph* say now?
  CUDA_KERNEL_NODE_PARAMS q{}; DR(cuGraphKernelNodeGetParams(nodes[0], &q));
  printf("graph node after exec-patch: grid=%u block=%u smem=%u (exec patch should NOT touch the graph)\n",
         q.gridDimX, q.blockDimX, q.sharedMemBytes);
  return 0;
}
