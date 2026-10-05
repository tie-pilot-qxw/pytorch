// How much can a fake run read? Read graph nodes via the driver API, and try cuFuncGetParamInfo to enumerate the param layout.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s: %s\n",#x,cudaGetErrorString(e)); return -1;}}while(0)
#define CB(x) do{cublasStatus_t e=(x); if(e!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS ERR %s: %d\n",#x,(int)e); return -1;}}while(0)
static const char* NT[]={"Kernel","Memcpy","Memset","Host","ChildGraph","Empty","WaitEvent","EventRecord",
                         "SemSignal","SemWait","MemAlloc","MemFree","BatchMemOp","Conditional"};

struct Rec { CUfunction f; std::string name; unsigned gx,gy,gz,bx,by,bz,smem; std::vector<unsigned char> blob; std::vector<std::pair<size_t,size_t>> params; };

int inspect(CUgraph g, Rec& r, const char* tag) {
  size_t n=0; cuGraphGetNodes(g,nullptr,&n);
  std::vector<CUgraphNode> nd(n); cuGraphGetNodes(g,nd.data(),&n);
  printf("  %s: %zu nodes -> ", tag, n);
  for (size_t i=0;i<n;i++){ CUgraphNodeType t; cuGraphNodeGetType(nd[i],&t); printf("%s ", (int)t<14?NT[(int)t]:"?"); }
  printf("\n");
  for (size_t i=0;i<n;i++){
    CUgraphNodeType t; cuGraphNodeGetType(nd[i],&t);
    if (t != CU_GRAPH_NODE_TYPE_KERNEL) continue;
    CUDA_KERNEL_NODE_PARAMS p{};
    CUresult e = cuGraphKernelNodeGetParams(nd[i], &p);
    if (e != CUDA_SUCCESS){ const char* m; cuGetErrorString(e,&m); printf("    cuGraphKernelNodeGetParams failed: %s\n", m); continue; }
    r.f=p.func; r.gx=p.gridDimX; r.gy=p.gridDimY; r.gz=p.gridDimZ;
    r.bx=p.blockDimX; r.by=p.blockDimY; r.bz=p.blockDimZ; r.smem=p.sharedMemBytes;
    const char* nm=nullptr; if (cuFuncGetName(&nm,p.func)==CUDA_SUCCESS && nm) r.name=nm; else r.name="(unnamed)";
    printf("    kernel: %s\n", r.name.c_str());
    printf("      grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u  kernelParams=%s extra=%s\n",
           p.gridDimX,p.gridDimY,p.gridDimZ,p.blockDimX,p.blockDimY,p.blockDimZ,p.sharedMemBytes,
           p.kernelParams?"non-NULL":"NULL", p.extra?"non-NULL":"NULL");
    // the key question: can the param layout be enumerated
    size_t off,sz,total=0; int k=0;
    for (;;k++){ if (cuFuncGetParamInfo(p.func,k,&off,&sz)!=CUDA_SUCCESS) break; r.params.push_back({off,sz}); total=off+sz; }
    printf("      cuFuncGetParamInfo: %s", k? "" : "**unavailable**\n");
    if (k){ printf("enumerated %d params, blob is %zu bytes total\n      first 12 (offset,size): ", k, total);
            for (int j=0;j<k && j<12;j++) printf("(%zu,%zu) ", r.params[j].first, r.params[j].second); printf("\n"); }
    if (p.kernelParams && k){   // grab the param values
      r.blob.resize(total,0);
      for (int j=0;j<k;j++) memcpy(r.blob.data()+r.params[j].first, p.kernelParams[j], r.params[j].second);
      printf("      captured %zu bytes of param data\n", r.blob.size());
    }
  }
  return 0;
}

int cap(cublasHandle_t h, cudaStream_t s, int M,int K,int N,__half*A,__half*B,__half*C, CUgraph* g){
  float al=1.f,be=0.f; CB(cublasSetStream(h,s));
  CB(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,B,CUDA_R_16F,N,A,CUDA_R_16F,K,&be,C,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  CK(cudaStreamSynchronize(s));
  CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
  CB(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,B,CUDA_R_16F,N,A,CUDA_R_16F,K,&be,C,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  cudaGraph_t gg; CK(cudaStreamEndCapture(s,&gg)); *g=(CUgraph)gg; return 0;
}

int main(){
  cuInit(0); cublasHandle_t h; CB(cublasCreate(&h)); cudaStream_t s; CK(cudaStreamCreate(&s));
  int K=4096,N=4096; __half *A,*B,*C;
  CK(cudaMalloc(&A,(size_t)8192*K*2)); CK(cudaMalloc(&B,(size_t)K*N*2)); CK(cudaMalloc(&C,(size_t)8192*N*2));
  printf("What can a fake run actually read? (no driver patching, read the nodes directly after capture)\n\n");
  Rec r1,r2,r3;
  CUgraph g1,g2,g3;
  cap(h,s,1024,K,N,A,B,C,&g1); inspect(g1,r1,"M=1024");
  cap(h,s,4096,K,N,A,B,C,&g2); inspect(g2,r2,"M=4096");
  cap(h,s,1088,K,N,A,B,C,&g3); inspect(g3,r3,"M=1088");
  printf("\n== Comparison ==\n");
  printf("  M=1024 vs 1088: kernel %s, grid (%u,%u)->(%u,%u)\n",
         r1.f==r3.f?"same":"**different**", r1.gx,r1.gy, r3.gx,r3.gy);
  if (r1.f==r3.f && r1.blob.size()==r3.blob.size() && !r1.blob.empty()){
    printf("  param blob field-by-field diff (same kernel, only M changes):\n");
    int changed=0;
    for (size_t j=0;j<r1.params.size();j++){
      size_t o=r1.params[j].first, z=r1.params[j].second;
      if (memcmp(r1.blob.data()+o, r3.blob.data()+o, z)){
        changed++;
        unsigned long long v1=0,v3=0; memcpy(&v1,r1.blob.data()+o,z>8?8:z); memcpy(&v3,r3.blob.data()+o,z>8?8:z);
        if (changed<=10) printf("    param %2zu (offset %3zu, %zu bytes): %llu -> %llu %s\n", j,o,z,v1,v3,
               (v1>0x700000000000ULL)?"[looks like a pointer]":"[looks like a scalar]");
      }
    }
    printf("    %d/%zu params changed in total\n", changed, r1.params.size());
  }
  printf("  M=1024 vs 4096: kernel %s\n", r1.f==r2.f?"same":"**different**");
  printf("    %s\n    %s\n", r1.name.c_str(), r2.name.c_str());
  return 0;
}
