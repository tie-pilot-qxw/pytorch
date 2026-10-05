// Which node types can be enabled/disabled? This decides how far "linear preset + SetEnabled" can reach.
// The critical one for DynaGraph is ChildGraph: extern sites are child-graph nodes, and if child-graph nodes
// cannot be disabled, selecting an extern variant can only be done with SWITCH.
#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("  ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return -1;}}while(0)
static const char* NT[]={"Kernel","Memcpy","Memset","Host","ChildGraph","Empty","WaitEvent","EventRecord",
                         "SemSignal","SemWait","MemAlloc","MemFree","BatchMemOp","Conditional"};
__global__ void mark(int* o, int v){ if(!threadIdx.x&&!blockIdx.x) o[0]=v; }
__global__ void devdis(cudaGraphDeviceNode_t n){ if(!threadIdx.x) cudaGraphKernelNodeSetEnabled(n,0); }

int main(){
  cuInit(0);
  int *d_o, *d_src; CK(cudaMalloc(&d_o,4)); CK(cudaMalloc(&d_src,4));
  int v=42; CK(cudaMemcpy(d_src,&v,4,cudaMemcpyHostToDevice));
  cudaStream_t s; CK(cudaStreamCreate(&s));

  // Put four node types in one graph: Kernel / Memset / Memcpy(D2D) / ChildGraph (one kernel inside)
  cudaGraph_t child;
  CK(cudaGraphCreate(&child,0));
  { cudaKernelNodeParams kp={}; void* args[]={&d_o,nullptr};
    static int cv=555; void* a2[]={&d_o,&cv};
    kp.func=(void*)mark; kp.gridDim=dim3(1); kp.blockDim=dim3(32); kp.kernelParams=a2;
    cudaGraphNode_t cn; CK(cudaGraphAddKernelNode(&cn,child,nullptr,0,&kp)); (void)args; }

  cudaGraph_t g; CK(cudaGraphCreate(&g,0));
  std::vector<cudaGraphNode_t> nds;
  { cudaKernelNodeParams kp={}; static int kv=111; void* a[]={&d_o,&kv};
    kp.func=(void*)mark; kp.gridDim=dim3(1); kp.blockDim=dim3(32); kp.kernelParams=a;
    cudaGraphNode_t n; CK(cudaGraphAddKernelNode(&n,g,nullptr,0,&kp)); nds.push_back(n); }
  { cudaMemsetParams mp={}; mp.dst=d_o; mp.value=0; mp.elementSize=4; mp.width=1; mp.height=1;
    cudaGraphNode_t n; CK(cudaGraphAddMemsetNode(&n,g,&nds.back(),1,&mp)); nds.push_back(n); }
  { cudaGraphNode_t n; CK(cudaGraphAddMemcpyNode1D(&n,g,&nds.back(),1,d_o,d_src,4,cudaMemcpyDeviceToDevice)); nds.push_back(n); }
  { cudaGraphNode_t n; CK(cudaGraphAddChildGraphNode(&n,g,&nds.back(),1,child)); nds.push_back(n); }

  cudaGraphExec_t ge; CK(cudaGraphInstantiate(&ge,g,0));
  printf("=== Host-side cudaGraphNodeSetEnabled on each node type ===\n");
  for(auto n : nds){
    cudaGraphNodeType t; CK(cudaGraphNodeGetType(n,&t));
    cudaError_t e0 = cudaGraphNodeSetEnabled(ge,n,0);
    cudaError_t e1 = (e0==cudaSuccess) ? cudaGraphNodeSetEnabled(ge,n,1) : cudaSuccess;
    printf("  %-11s disable: %-42s %s\n", NT[(int)t],
           e0==cudaSuccess?"allowed":cudaGetErrorString(e0),
           e0==cudaSuccess?(e1==cudaSuccess?"(can be re-enabled)":"**cannot be re-enabled**"):"");
    cudaGetLastError();
  }

  printf("\n=== With ChildGraph disabled, does it really not run (sentinel) ===\n");
  { // order: kernel(111) -> memset(0) -> memcpy(42) -> child(555)
    int out;
    CK(cudaGraphLaunch(ge,s)); CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(&out,d_o,4,cudaMemcpyDeviceToHost));
    printf("  all enabled         -> %d (expected 555, child writes last)\n", out);
    cudaError_t e = cudaGraphNodeSetEnabled(ge,nds[3],0);
    if(e==cudaSuccess){
      CK(cudaGraphLaunch(ge,s)); CK(cudaStreamSynchronize(s));
      CK(cudaMemcpy(&out,d_o,4,cudaMemcpyDeviceToHost));
      printf("  ChildGraph disabled -> %d %s\n", out, out==42?"(child did not run, memcpy's 42 remains)":"**child still runs**");
    } else printf("  ChildGraph disabled -> not allowed: %s\n", cudaGetErrorString(e));
    cudaGetLastError();
  }

  printf("\n=== Device-side cudaGraphKernelNodeSetEnabled only accepts kernel nodes ===\n");
  printf("  The API name itself says KernelNode, and the handle type cudaGraphDeviceNode_t can only be\n"
         "  obtained from cudaLaunchAttributeDeviceUpdatableKernelNode -- non-kernel nodes\n"
         "  can never get such a handle, so they cannot be disabled from the device.\n");
  return 0;
}
