// Observation is not sound. So don't observe -- "pin" cuBLAS's choice so it has nothing to choose.
// cublasLt allows specifying the algo explicitly. Test: once an algo is pinned, does the kernel still change? What does it cost?
#include <cstdio>
#include <vector>
#include <cuda.h>
#include <set>
#include <string>
#include <cuda_runtime.h>
#include <cublasLt.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s: %s\n",#x,cudaGetErrorString(e)); return -1;}}while(0)
#define CL(x) do{cublasStatus_t e=(x); if(e!=CUBLAS_STATUS_SUCCESS){printf("LT ERR %s: %d\n",#x,(int)e); return -1;}}while(0)

static cublasLtHandle_t lt;
static int K=4096, N=4096;
static __half *A,*B,*C;
static void *ws; static size_t WS = 64ull<<20;

// Run once with the given algo (algo=nullptr means the default heuristic); return the captured kernel name + grid
int probe(int M, const cublasLtMatmulAlgo_t* algo, char* name, int* gx, int* gy, int* nnodes, float* ms) {
  cublasLtMatmulDesc_t op; CL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t tn = CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tn, sizeof(tn)));
  CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof(tn)));
  cublasLtMatrixLayout_t la,lb,lc;
  CL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, N, K, N));
  CL(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, K, M, K));
  CL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, N, M, N));
  float al=1.f, be=0.f;
  cublasLtMatmulAlgo_t chosen;
  if (!algo) {
    cublasLtMatmulPreference_t pref; CL(cublasLtMatmulPreferenceCreate(&pref));
    CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof(WS)));
    cublasLtMatmulHeuristicResult_t r[1]; int got=0;
    CL(cublasLtMatmulAlgoGetHeuristic(lt, op, lb, la, lc, lc, pref, 1, r, &got));
    if (!got) { printf("    M=%d no usable algo\n", M); return -1; }
    chosen = r[0].algo; cublasLtMatmulPreferenceDestroy(pref);
  } else chosen = *algo;

  cudaStream_t s; CK(cudaStreamCreate(&s));
  CL(cublasLtMatmul(lt, op, &al, B, lb, A, la, &be, C, lc, C, lc, &chosen, ws, WS, s));   // warm-up
  CK(cudaStreamSynchronize(s));
  cudaGraph_t g;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  CL(cublasLtMatmul(lt, op, &al, B, lb, A, la, &be, C, lc, C, lc, &chosen, ws, WS, s));
  CK(cudaStreamEndCapture(s, &g));
  size_t nn=0; cuGraphGetNodes((CUgraph)g, nullptr, &nn); *nnodes=(int)nn;
  std::vector<CUgraphNode> nd(nn); cuGraphGetNodes((CUgraph)g, nd.data(), &nn);
  name[0]=0; *gx=*gy=0;
  for (size_t i=0;i<nn;i++){ CUgraphNodeType t; cuGraphNodeGetType(nd[i],&t);
    if (t!=CU_GRAPH_NODE_TYPE_KERNEL) continue;
    CUDA_KERNEL_NODE_PARAMS p{}; if (cuGraphKernelNodeGetParams(nd[i],&p)!=CUDA_SUCCESS) continue;
    const char* nm=nullptr; if (cuFuncGetName(&nm,p.func)==CUDA_SUCCESS && nm) snprintf(name,120,"%s",nm);
    *gx=p.gridDimX; *gy=p.gridDimY; }
  // timing
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for(int i=0;i<5;i++) CL(cublasLtMatmul(lt,op,&al,B,lb,A,la,&be,C,lc,C,lc,&chosen,ws,WS,s));
  CK(cudaStreamSynchronize(s)); cudaEventRecord(e0,s);
  for(int i=0;i<50;i++) CL(cublasLtMatmul(lt,op,&al,B,lb,A,la,&be,C,lc,C,lc,&chosen,ws,WS,s));
  cudaEventRecord(e1,s); CK(cudaStreamSynchronize(s)); cudaEventElapsedTime(ms,e0,e1); *ms/=50;
  cublasLtMatmulDescDestroy(op); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
  cudaStreamDestroy(s);
  return 0;
}

int main(){
  cuInit(0); CL(cublasLtCreate(&lt));
  CK(cudaMalloc(&A,(size_t)8192*K*2)); CK(cudaMalloc(&B,(size_t)K*N*2)); CK(cudaMalloc(&C,(size_t)8192*N*2)); CK(cudaMalloc(&ws,WS));
  int Ms[] = {512, 1024, 1088, 2048, 3000, 4096, 6144, 8192};
  char nm[128]; int gx,gy,nn; float ms;

  printf("=== A. Default heuristic (today's behavior) ===\n");
  printf("  %6s %10s %8s  %s\n","M","grid","nodes","kernel");
  std::vector<std::string> defk;
  for (int M : Ms){ if(probe(M,nullptr,nm,&gx,&gy,&nn,&ms)) continue;
    printf("  %6d %5dx%-4d %8d  %.60s\n",M,gx,gy,nn,nm); defk.push_back(nm); }
  printf("  => %zu M values used %zu distinct kernels\n", defk.size(), std::set<std::string>(defk.begin(),defk.end()).size());

  printf("\n=== B. Pick an algo at M=4096, then pin it for all M ===\n");
  // take the top algo for M=4096
  cublasLtMatmulDesc_t op; CL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t tn=CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&tn,sizeof(tn)));
  CL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&tn,sizeof(tn)));
  cublasLtMatrixLayout_t la,lb,lc;
  CL(cublasLtMatrixLayoutCreate(&lb,CUDA_R_16F,N,K,N));
  CL(cublasLtMatrixLayoutCreate(&la,CUDA_R_16F,K,4096,K));
  CL(cublasLtMatrixLayoutCreate(&lc,CUDA_R_16F,N,4096,N));
  cublasLtMatmulPreference_t pref; CL(cublasLtMatmulPreferenceCreate(&pref));
  CL(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&WS,sizeof(WS)));
  cublasLtMatmulHeuristicResult_t r[1]; int got=0;
  CL(cublasLtMatmulAlgoGetHeuristic(lt,op,lb,la,lc,lc,pref,1,r,&got));
  if(!got){ printf("  could not get an algo\n"); return 1; }
  cublasLtMatmulAlgo_t pinned = r[0].algo;
  printf("  %6s %10s %8s %10s  %s\n","M","grid","nodes","vs default","kernel");
  std::vector<std::string> pink; int fail=0;
  size_t i=0;
  for (int M : Ms){
    float ms_def=0; char nd2[128]; int a,b,c2; probe(M,nullptr,nd2,&a,&b,&c2,&ms_def);
    if (probe(M,&pinned,nm,&gx,&gy,&nn,&ms)) { printf("  %6d  **this algo does not support this M**\n",M); fail++; continue; }
    printf("  %6d %5dx%-4d %8d %9.0f%%  %.55s\n",M,gx,gy,nn, 100.0*ms_def/ms, nm); pink.push_back(nm);
  }
  printf("  => %zu M values used %zu kernels, %d M values do not support this algo\n",
         pink.size(), std::set<std::string>(pink.begin(),pink.end()).size(), fail);
  return 0;
}
// (Addendum: timing of the ask cost is in ask_cost.cu)
