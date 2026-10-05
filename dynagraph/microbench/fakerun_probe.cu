// Does a "fake run" really need to patch the CUDA driver API?
// A cleaner approach: capture into a graph, then read the nodes out directly. Measure how much can be read.
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s: %s\n",#x,cudaGetErrorString(e)); return -1;}}while(0)
#define CB(x) do{cublasStatus_t e=(x); if(e!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS ERR %s: %d\n",#x,(int)e); return -1;}}while(0)

#include <string>
struct NodeInfo { const void* func; dim3 grid, block; unsigned smem; std::string name; };

static const char* fname(CUfunction f) {
  static char buf[256]; const char* n = nullptr;
#if CUDA_VERSION >= 12030
  if (cuFuncGetName(&n, f) == CUDA_SUCCESS && n) { snprintf(buf,sizeof buf,"%s",n); return buf; }
#endif
  return "(name unavailable)";
}

int capture_gemm(cublasHandle_t h, cudaStream_t s, int M, int K, int N,
                 __half* A, __half* B, __half* C, std::vector<NodeInfo>& out) {
  cudaGraph_t g;
  float alpha=1.f, beta=0.f;
  CB(cublasSetStream(h, s));
  // warm-up, so lazy init does not get recorded too
  CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
                  &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  CK(cudaStreamSynchronize(s));
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
                  &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  CK(cudaStreamEndCapture(s, &g));
  size_t n=0; CK(cudaGraphGetNodes(g, nullptr, &n));
  std::vector<cudaGraphNode_t> nd(n); CK(cudaGraphGetNodes(g, nd.data(), &n));
  for (size_t i=0;i<n;i++) {
    cudaGraphNodeType t; CK(cudaGraphNodeGetType(nd[i], &t));
    if (t != cudaGraphNodeTypeKernel) { printf("    node %zu: non-kernel node (type=%d)\n", i, (int)t); continue; }
    cudaKernelNodeParams p{};
    cudaError_t e = cudaGraphKernelNodeGetParams(nd[i], &p);
    if (e != cudaSuccess) { printf("    node %zu: GetParams failed %s\n", i, cudaGetErrorString(e)); continue; }
    NodeInfo ni; ni.func=p.func; ni.grid=p.gridDim; ni.block=p.blockDim; ni.smem=p.sharedMemBytes;
    ni.name = fname((CUfunction)p.func);
    out.push_back(ni);
    printf("    node %zu: grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u  kernelParams=%s extra=%s\n",
           i, p.gridDim.x,p.gridDim.y,p.gridDim.z, p.blockDim.x,p.blockDim.y,p.blockDim.z,
           p.sharedMemBytes, p.kernelParams? "readable":"NULL", p.extra? "present":"NULL");
    printf("           kernel = %s\n", ni.name.c_str());
  }
  return 0;
}

int main(){
  cuInit(0);
  cublasHandle_t h; CB(cublasCreate(&h));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  int K=4096,N=4096;
  __half *A,*B,*C; CK(cudaMalloc(&A,(size_t)8192*K*2)); CK(cudaMalloc(&B,(size_t)K*N*2)); CK(cudaMalloc(&C,(size_t)8192*N*2));
  printf("Question: without patching the driver, can grid / kernel / params be read directly from the nodes of a captured graph?\n");
  printf("Method: the same cuBLAS GEMM at two different M, capture each once, read the nodes, compare.\n\n");
  std::vector<NodeInfo> a,b;
  printf("  M=1024:\n"); capture_gemm(h,s,1024,K,N,A,B,C,a);
  printf("  M=4096:\n"); capture_gemm(h,s,4096,K,N,A,B,C,b);
  printf("\nComparison:\n");
  printf("  node count   %zu vs %zu  %s\n", a.size(), b.size(), a.size()==b.size()?"(same, topology unchanged)":"(**topology changed**)");
  if (!a.empty() && !b.empty()) {
    printf("  kernel func  %s\n", a[0].func==b[0].func ? "same" : "**different -- changing M changed the kernel**");
    printf("               M=1024: %s\n               M=4096: %s\n", a[0].name.c_str(), b[0].name.c_str());
    printf("  gridDim      (%u,%u,%u) vs (%u,%u,%u)  %s\n", a[0].grid.x,a[0].grid.y,a[0].grid.z,
           b[0].grid.x,b[0].grid.y,b[0].grid.z, (a[0].grid.x==b[0].grid.x&&a[0].grid.y==b[0].grid.y)?"same":"**different**");
    printf("  blockDim     (%u,%u,%u) vs (%u,%u,%u)\n", a[0].block.x,a[0].block.y,a[0].block.z, b[0].block.x,b[0].block.y,b[0].block.z);
    printf("  smem         %u vs %u  %s\n", a[0].smem, b[0].smem, a[0].smem==b[0].smem?"same":"**different (a graph cannot change smem either)**");
  }
  return 0;
}
