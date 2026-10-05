// "Just update the graph whenever a new kernel shows up" -- which updates are actually legal, and how expensive is each?
#include <cstdio>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("  ERR %s: %s\n",#x,cudaGetErrorString(e)); return -1;}}while(0)
#define TRY(x) ({cudaError_t e_=(x); e_;})
__global__ void kA(int* o, int n){ if(!threadIdx.x&&!blockIdx.x) o[0]=100+n; }
__global__ void kB(int* o, int n){ if(!threadIdx.x&&!blockIdx.x) o[0]=200+n; }   // same signature, different implementation (= a GEMM variant with a different tile)
__global__ void kExtra(int* o, int n){ if(!threadIdx.x&&!blockIdx.x) o[1]=999; }
__global__ void nop(int* o){ if(!threadIdx.x) o[2]++; }
static int* d_o; static cudaStream_t s;
double us(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b){
  return std::chrono::duration<double,std::micro>(b-a).count(); }

int build(int N, bool extra, cudaGraph_t* g, std::vector<cudaGraphNode_t>* nodes){
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for(int i=0;i<N;i++) kA<<<1,64,0,s>>>(d_o,i);
  if(extra) kExtra<<<1,64,0,s>>>(d_o,0);
  CK(cudaStreamEndCapture(s,g));
  if(nodes){ size_t n=0; CK(cudaGraphGetNodes(*g,nullptr,&n)); nodes->resize(n); CK(cudaGraphGetNodes(*g,nodes->data(),&n)); }
  return 0;
}

int main(){
  CK(cudaMalloc(&d_o,12)); CK(cudaMemset(d_o,0,12)); CK(cudaStreamCreate(&s));

  printf("=== 1. Swap the kernel function: host-side cudaGraphExecKernelNodeSetParams, no re-instantiate\n");
  for(int N : {40, 500, 3000}){
    cudaGraph_t g; std::vector<cudaGraphNode_t> nodes; cudaGraphExec_t ge;
    build(N,false,&g,&nodes); CK(cudaGraphInstantiate(&ge,g,0));
    CK(cudaGraphLaunch(ge,s)); CK(cudaStreamSynchronize(s));
    int before; CK(cudaMemcpy(&before,d_o,4,cudaMemcpyDeviceToHost));
    // swap the last node's function from kA to kB
    cudaKernelNodeParams np{}; int n_arg=N-1; void* args[]={&d_o,&n_arg};
    np.func=(void*)kB; np.gridDim=dim3(1); np.blockDim=dim3(64); np.sharedMemBytes=0; np.kernelParams=args;
    auto t0=std::chrono::steady_clock::now();
    cudaError_t e = TRY(cudaGraphExecKernelNodeSetParams(ge, nodes[N-1], &np));
    auto t1=std::chrono::steady_clock::now();
    if(e!=cudaSuccess){ printf("   N=%-5d FAILED: %s\n",N,cudaGetErrorString(e)); continue; }
    CK(cudaGraphLaunch(ge,s)); CK(cudaStreamSynchronize(s));
    int after; CK(cudaMemcpy(&after,d_o,4,cudaMemcpyDeviceToHost));
    // then measure how long swapping all N nodes takes
    auto t2=std::chrono::steady_clock::now();
    for(int i=0;i<N;i++){ int a=i; void* ar[]={&d_o,&a}; np.kernelParams=ar; cudaGraphExecKernelNodeSetParams(ge,nodes[i],&np); }
    auto t3=std::chrono::steady_clock::now();
    printf("   N=%-5d swap func on one node %6.2f us  (%d -> %d, %s)   swap all %d %8.1f us (%.2f us/node)\n",
           N, us(t0,t1), before, after, (before/100==1&&after/100==2)?"took effect":"no effect", N, us(t2,t3), us(t2,t3)/N);
  }

  printf("\n=== 2. cudaGraphExecUpdate: swap function vs add a node\n");
  {
    cudaGraph_t g1,g2,g3; cudaGraphExec_t ge; std::vector<cudaGraphNode_t> nd;
    build(500,false,&g1,&nd); CK(cudaGraphInstantiate(&ge,g1,0));
    // g2: same topology, but all nodes are kB
    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    for(int i=0;i<500;i++) kB<<<1,64,0,s>>>(d_o,i);
    CK(cudaStreamEndCapture(s,&g2));
    cudaGraphExecUpdateResultInfo info{};
    auto t0=std::chrono::steady_clock::now();
    cudaError_t e = TRY(cudaGraphExecUpdate(ge,g2,&info));
    auto t1=std::chrono::steady_clock::now();
    printf("   same topology, all 500 nodes swap function : %s  (%.1f us)\n", e==cudaSuccess?"succeeded":cudaGetErrorString(e), us(t0,t1));
    if(e==cudaSuccess){ CK(cudaGraphLaunch(ge,s)); CK(cudaStreamSynchronize(s)); int v; CK(cudaMemcpy(&v,d_o,4,cudaMemcpyDeviceToHost)); printf("     -> actually executed k%c\n", v/100==2?'B':'A'); }
    // g3: one extra node
    build(500,true,&g3,nullptr);
    t0=std::chrono::steady_clock::now();
    e = TRY(cudaGraphExecUpdate(ge,g3,&info));
    t1=std::chrono::steady_clock::now();
    printf("   one extra kernel node (topology changed)   : %s  (%.1f us)  result=%d\n",
           e==cudaSuccess?"succeeded":cudaGetErrorString(e), us(t0,t1), (int)info.result);
  }

  printf("\n=== 3. When the topology really has to change: how expensive is re-instantiation\n");
  for(int N : {500, 3000}){
    cudaGraph_t g; cudaGraphExec_t ge; build(N,false,&g,nullptr);
    auto t0=std::chrono::steady_clock::now();
    CK(cudaGraphInstantiate(&ge,g,0));
    auto t1=std::chrono::steady_clock::now();
    CK(cudaGraphUpload(ge,s)); CK(cudaStreamSynchronize(s));
    auto t2=std::chrono::steady_clock::now();
    printf("   N=%-5d instantiate %8.1f us (%.2f us/node)   first upload afterwards %7.1f us\n", N, us(t0,t1), us(t0,t1)/N, us(t1,t2));
  }

  printf("\n=== 4. SWITCH: do empty bodies cost anything? Can branches be added afterwards?\n");
  for(int V : {2, 8, 16}){
    int POP = 2;   // only fill 2 bodies, leave the rest empty
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    cudaGraph_t cap; const cudaGraphNode_t* dp; size_t nd; cudaStreamCaptureStatus st;
    CK(cudaStreamGetCaptureInfo(s,&st,nullptr,&cap,&dp,nullptr,&nd));
    cudaGraphConditionalHandle ch; CK(cudaGraphConditionalHandleCreate(&ch,cap,0,cudaGraphCondAssignDefault));
    nop<<<1,32,0,s>>>(d_o);
    CK(cudaStreamGetCaptureInfo(s,&st,nullptr,&cap,&dp,nullptr,&nd));
    cudaGraphNodeParams np{}; np.type=cudaGraphNodeTypeConditional; np.conditional.handle=ch;
    np.conditional.type=cudaGraphCondTypeSwitch; np.conditional.size=V;
    cudaGraphNode_t cn; CK(cudaGraphAddNode(&cn,cap,dp,nullptr,nd,&np));
    std::vector<cudaGraph_t> bodies(V);
    for(int v=0;v<V;v++){ bodies[v]=np.conditional.phGraph_out[v];
      if(v<POP){ cudaStream_t bs; CK(cudaStreamCreate(&bs));
        CK(cudaStreamBeginCaptureToGraph(bs,bodies[v],nullptr,nullptr,0,cudaStreamCaptureModeGlobal));
        kA<<<1,64,0,bs>>>(d_o,v); cudaGraph_t tmp; CK(cudaStreamEndCapture(bs,&tmp)); } }
    CK(cudaStreamUpdateCaptureDependencies(s,&cn,nullptr,1,cudaStreamSetCaptureDependencies));
    CK(cudaStreamEndCapture(s,&g)); CK(cudaGraphInstantiate(&ge,g,0));
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
    cudaEventRecord(a,s); for(int i=0;i<300;i++) cudaGraphLaunch(ge,s); cudaEventRecord(b,s); cudaStreamSynchronize(s);
    float ms; cudaEventElapsedTime(&ms,a,b);
    // afterwards add a kernel to an empty body, then ExecUpdate
    cudaStream_t bs; CK(cudaStreamCreate(&bs));
    cudaError_t e1 = TRY(cudaStreamBeginCaptureToGraph(bs,bodies[POP],nullptr,nullptr,0,cudaStreamCaptureModeGlobal));
    const char* addres = "N/A";
    if(e1==cudaSuccess){ kB<<<1,64,0,bs>>>(d_o,POP); cudaGraph_t tmp; cudaStreamEndCapture(bs,&tmp);
      cudaGraphExecUpdateResultInfo info{};
      cudaError_t e2 = TRY(cudaGraphExecUpdate(ge,g,&info));
      addres = (e2==cudaSuccess) ? "ExecUpdate succeeded" : cudaGetErrorString(e2); }
    else addres = cudaGetErrorString(e1);
    printf("   SWITCH size=%-3d (only %d bodies filled): %6.2f us per launch   filling body %d afterwards -> %s\n",
           V, POP, ms*1000/300, POP, addres);
  }
  return 0;
}
