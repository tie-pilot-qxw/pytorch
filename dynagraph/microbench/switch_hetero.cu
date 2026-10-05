// Two things:
// 1) Can different bodies of a SWITCH have different node counts (a load-bearing assumption)
// 2) Can cuBLAS be put inside a SWITCH body -- this decides whether "cuBLAS topology changes" is a dead end or can be handled
#include <cstdio>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("  ERR %s: %s\n",#x,cudaGetErrorString(e)); return -1;}}while(0)
#define CL(x) do{cublasStatus_t e=(x); if(e!=CUBLAS_STATUS_SUCCESS){printf("  LT ERR %s: %d\n",#x,(int)e); return -1;}}while(0)
static const char* NT[]={"Kernel","Memcpy","Memset","Host","ChildGraph","Empty","WaitEvent","EventRecord",
                         "SemSignal","SemWait","MemAlloc","MemFree","BatchMemOp","Conditional"};
__global__ void planner(cudaGraphConditionalHandle h, const int* in){ if(!threadIdx.x) cudaGraphSetConditional(h, in[0]); }
__global__ void mark(int* o, int v){ if(!threadIdx.x&&!blockIdx.x) o[0]=v; }
__global__ void bump(int* o){ if(!threadIdx.x&&!blockIdx.x) o[1]+=1; }

static void dump(CUgraph g, const char* tag){
  size_t n=0; cuGraphGetNodes(g,nullptr,&n);
  std::vector<CUgraphNode> nd(n); cuGraphGetNodes(g,nd.data(),&n);
  printf("      %s: %zu nodes [", tag, n);
  for(size_t i=0;i<n;i++){ CUgraphNodeType t; cuGraphNodeGetType(nd[i],&t); printf("%s%s",(int)t<14?NT[(int)t]:"?", i+1<n?" ":""); }
  printf("]\n");
}

int main(){
  cuInit(0);
  int *d_in,*d_o; CK(cudaMalloc(&d_in,4)); CK(cudaMalloc(&d_o,8)); CK(cudaMemset(d_o,0,8));
  cudaStream_t s; CK(cudaStreamCreate(&s));

  printf("=== Test 1: three SWITCH bodies with different node counts (1 / 2 incl. memset / 3) ===\n");
  { cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    cudaGraph_t cap; const cudaGraphNode_t* dp; size_t nd; cudaStreamCaptureStatus st;
    CK(cudaStreamGetCaptureInfo(s,&st,nullptr,&cap,&dp,nullptr,&nd));
    cudaGraphConditionalHandle h; CK(cudaGraphConditionalHandleCreate(&h,cap,0,cudaGraphCondAssignDefault));
    planner<<<1,32,0,s>>>(h,d_in);
    CK(cudaStreamGetCaptureInfo(s,&st,nullptr,&cap,&dp,nullptr,&nd));
    cudaGraphNodeParams p={}; p.type=cudaGraphNodeTypeConditional; p.conditional.handle=h;
    p.conditional.type=cudaGraphCondTypeSwitch; p.conditional.size=3;
    cudaGraphNode_t cn; CK(cudaGraphAddNode(&cn,cap,dp,nullptr,nd,&p));
    for(int v=0;v<3;v++){
      cudaGraph_t body=p.conditional.phGraph_out[v]; cudaStream_t bs; CK(cudaStreamCreate(&bs));
      CK(cudaStreamBeginCaptureToGraph(bs,body,nullptr,nullptr,0,cudaStreamCaptureModeGlobal));
      if(v==0){ mark<<<1,32,0,bs>>>(d_o,100); }
      else if(v==1){ CK(cudaMemsetAsync(d_o+1,0,4,bs)); mark<<<1,32,0,bs>>>(d_o,200); }   // memset + kernel
      else { mark<<<1,32,0,bs>>>(d_o,300); bump<<<1,32,0,bs>>>(d_o); bump<<<1,32,0,bs>>>(d_o); }
      cudaGraph_t tmp; CK(cudaStreamEndCapture(bs,&tmp));
      dump((CUgraph)body, (std::string("body ")+std::to_string(v)).c_str());
    }
    CK(cudaStreamUpdateCaptureDependencies(s,&cn,nullptr,1,cudaStreamSetCaptureDependencies));
    CK(cudaStreamEndCapture(s,&g)); CK(cudaGraphInstantiate(&ge,g,0));
    bool ok=true;
    for(int v=0;v<3;v++){ CK(cudaMemcpy(d_in,&v,4,cudaMemcpyHostToDevice)); CK(cudaMemset(d_o,0,8));
      CK(cudaGraphLaunch(ge,s)); CK(cudaStreamSynchronize(s));
      int out[2]; CK(cudaMemcpy(out,d_o,8,cudaMemcpyDeviceToHost));
      int exp_mark=(v+1)*100, exp_bump=(v==2)?2:0;
      printf("      select body %d -> mark=%d(expect %d) bump=%d(expect %d) %s\n",v,out[0],exp_mark,out[1],exp_bump,
             (out[0]==exp_mark&&out[1]==exp_bump)?"OK":"**WRONG**");
      ok &= (out[0]==exp_mark && out[1]==exp_bump); }
    printf("   => bodies can have different node counts: %s\n", ok?"**yes**":"no");
  }

  printf("\n=== Test 2: can cuBLAS go into a SWITCH body (two M values, topology may differ) ===\n");
  { cublasLtHandle_t lt; CL(cublasLtCreate(&lt));
    int K=1024,N=1024; size_t WS=32ull<<20; void* ws; CK(cudaMalloc(&ws,WS));
    __half *A,*B,*C; CK(cudaMalloc(&A,(size_t)8192*K*2)); CK(cudaMalloc(&B,(size_t)K*N*2)); CK(cudaMalloc(&C,(size_t)8192*N*2));
    float al=1.f,be=0.f;
    auto run=[&](int M, cudaStream_t st)->int{
      cublasLtMatmulDesc_t op; CL(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
      cublasOperation_t tn=CUBLAS_OP_N;
      CL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&tn,sizeof(tn)));
      CL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&tn,sizeof(tn)));
      cublasLtMatrixLayout_t la,lb,lc;
      CL(cublasLtMatrixLayoutCreate(&lb,CUDA_R_16F,N,K,N));
      CL(cublasLtMatrixLayoutCreate(&la,CUDA_R_16F,K,M,K));
      CL(cublasLtMatrixLayoutCreate(&lc,CUDA_R_16F,N,M,N));
      cublasLtMatmulPreference_t pf; CL(cublasLtMatmulPreferenceCreate(&pf));
      CL(cublasLtMatmulPreferenceSetAttribute(pf,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&WS,sizeof(WS)));
      cublasLtMatmulHeuristicResult_t r[1]; int got=0;
      CL(cublasLtMatmulAlgoGetHeuristic(lt,op,lb,la,lc,lc,pf,1,r,&got));
      if(!got) return -1;

      CL(cublasLtMatmul(lt,op,&al,B,lb,A,la,&be,C,lc,C,lc,&r[0].algo,ws,WS,st));
      return 0; };
    // warm-up, so lazy init is not recorded into the body
    run(1024,s); run(4096,s); CK(cudaStreamSynchronize(s));

    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    cudaGraph_t cap; const cudaGraphNode_t* dp; size_t nd; cudaStreamCaptureStatus st2;
    CK(cudaStreamGetCaptureInfo(s,&st2,nullptr,&cap,&dp,nullptr,&nd));
    cudaGraphConditionalHandle h; CK(cudaGraphConditionalHandleCreate(&h,cap,0,cudaGraphCondAssignDefault));
    planner<<<1,32,0,s>>>(h,d_in);
    CK(cudaStreamGetCaptureInfo(s,&st2,nullptr,&cap,&dp,nullptr,&nd));
    cudaGraphNodeParams p={}; p.type=cudaGraphNodeTypeConditional; p.conditional.handle=h;
    p.conditional.type=cudaGraphCondTypeSwitch; p.conditional.size=2;
    cudaGraphNode_t cn; CK(cudaGraphAddNode(&cn,cap,dp,nullptr,nd,&p));
    int Ms[2]={1024,4096}; bool cap_ok=true;
    for(int v=0;v<2;v++){
      cudaGraph_t body=p.conditional.phGraph_out[v]; cudaStream_t bs; CK(cudaStreamCreate(&bs));
      cudaError_t e1=cudaStreamBeginCaptureToGraph(bs,body,nullptr,nullptr,0,cudaStreamCaptureModeGlobal);
      if(e1!=cudaSuccess){ printf("      body %d BeginCapture failed: %s\n",v,cudaGetErrorString(e1)); cap_ok=false; break; }
      int rc=run(Ms[v],bs);
      cudaGraph_t tmp; cudaError_t e2=cudaStreamEndCapture(bs,&tmp);
      if(rc||e2!=cudaSuccess){ printf("      body %d (M=%d) capture failed: rc=%d %s\n",v,Ms[v],rc,cudaGetErrorString(e2)); cap_ok=false; break; }
      dump((CUgraph)body,(std::string("M=")+std::to_string(Ms[v])).c_str());
    }
    if(cap_ok){
      CK(cudaStreamUpdateCaptureDependencies(s,&cn,nullptr,1,cudaStreamSetCaptureDependencies));
      CK(cudaStreamEndCapture(s,&g));
      cudaError_t ei=cudaGraphInstantiate(&ge,g,0);
      printf("   instantiate: %s\n", ei==cudaSuccess?"succeeded":cudaGetErrorString(ei));
      if(ei==cudaSuccess){
        for(int v=0;v<2;v++){ CK(cudaMemcpy(d_in,&v,4,cudaMemcpyHostToDevice));
          cudaError_t el=cudaGraphLaunch(ge,s); cudaError_t es=cudaStreamSynchronize(s);
          printf("      select body %d (M=%d) launch: %s / sync: %s\n",v,Ms[v],
                 cudaGetErrorString(el),cudaGetErrorString(es)); }
        printf("   => cuBLAS can go into a SWITCH body: **yes**\n");
      }
    } else {
      CK(cudaStreamEndCapture(s,&g));
      printf("   => cuBLAS in a SWITCH body: no (see above)\n");
    }
  }
  return 0;
}
