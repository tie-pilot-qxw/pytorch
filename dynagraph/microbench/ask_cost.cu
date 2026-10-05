// What the "ask" path costs per call: cublasLtMatmulAlgoGetHeuristic tells you what it would pick without launching anything.
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
#include <cublasLt.h>
#define CL(x) do{cublasStatus_t e=(x); if(e!=CUBLAS_STATUS_SUCCESS){printf("LT ERR %d\n",(int)e); return -1;}}while(0)
int main(){
  cublasLtHandle_t lt; CL(cublasLtCreate(&lt));
  int K=4096,N=4096; size_t WS=64ull<<20;
  cublasLtMatmulDesc_t op; CL(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t tn=CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&tn,sizeof(tn)));
  CL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&tn,sizeof(tn)));
  cublasLtMatmulPreference_t pref; CL(cublasLtMatmulPreferenceCreate(&pref));
  CL(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&WS,sizeof(WS)));
  auto ask=[&](int M)->double{
    cublasLtMatrixLayout_t la,lb,lc;
    cublasLtMatrixLayoutCreate(&lb,CUDA_R_16F,N,K,N);
    cublasLtMatrixLayoutCreate(&la,CUDA_R_16F,K,M,K);
    cublasLtMatrixLayoutCreate(&lc,CUDA_R_16F,N,M,N);
    cublasLtMatmulHeuristicResult_t r[1]; int got=0;
    auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<100;i++) cublasLtMatmulAlgoGetHeuristic(lt,op,lb,la,lc,lc,pref,1,r,&got);
    auto t1=std::chrono::steady_clock::now();
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    return std::chrono::duration<double,std::micro>(t1-t0).count()/100; };
  ask(1024);   // warm-up
  printf("cublasLtMatmulAlgoGetHeuristic -- no launch, only asks 'which one would you pick':\n");
  for (int M : {512,1024,2048,4096,8192}) printf("  M=%-5d  %6.2f us/call\n", M, ask(M));
  printf("\nReference: one graph launch is ~4 us; if a training step has 100 GEMMs,\n");
  printf("      asking for all of them every step = 100 x the number above. Whether that is acceptable depends on whether it overlaps with the GPU.\n");
  return 0;
}
