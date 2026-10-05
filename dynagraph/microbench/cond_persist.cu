// Two points raised in the astra review, both things where "code that is already running" may already be wrong.
//
// 1) Does a conditional's value persist across replays? The planner comment in dynagraph.py says
//    "Set only when the shape changed, so a skipped planner leaves the last
//    selection standing". The CUDA docs say that without cudaGraphCondAssignDefault the value
//    is undefined at the start of each execution, and with that flag it is reset to the default. Neither supports "it stays".
//
// 2) SetEnabled only applies to kernel nodes. If a branch contains a memset/copy,
//    disabling its kernel does not disable that memset -- it still runs and wipes out someone else's result.
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("  ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return -1;}}while(0)

// One kernel sets all handles -- this is exactly what DynaGraph's planner does;
// used to separate "how expensive the conditional node itself is" from "how expensive one extra kernel per site is".
struct Hs { cudaGraphConditionalHandle h[8]; };
__global__ void setcond_all(Hs hs, int n, const int* sel) {
  if (threadIdx.x == 0)
    for (int k = 0; k < n; k++) cudaGraphSetConditional(hs.h[k], (unsigned)*sel);
}
__global__ void setcond(cudaGraphConditionalHandle h, const int* sel, const int* go) {
  // go==0 simulates "the planner exits early and does not set the condition this round"
  if (threadIdx.x == 0 && *go) cudaGraphSetConditional(h, (unsigned)*sel);
}
__global__ void mark(int* o, int v) { if (!threadIdx.x && !blockIdx.x) o[0] = v; }

static int test_persist(unsigned dflt, const char* tag) {
  int *d_sel, *d_go, *d_o;
  CK(cudaMalloc(&d_sel, 4)); CK(cudaMalloc(&d_go, 4)); CK(cudaMalloc(&d_o, 4));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  cudaGraph_t cap; const cudaGraphNode_t* dp; size_t nd; cudaStreamCaptureStatus st;
  CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &cap, &dp, nullptr, &nd));
  cudaGraphConditionalHandle h;
  CK(cudaGraphConditionalHandleCreate(&h, cap, dflt, cudaGraphCondAssignDefault));
  setcond<<<1, 32, 0, s>>>(h, d_sel, d_go);
  CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &cap, &dp, nullptr, &nd));
  cudaGraphNodeParams p = {}; p.type = cudaGraphNodeTypeConditional;
  p.conditional.handle = h; p.conditional.type = cudaGraphCondTypeSwitch; p.conditional.size = 3;
  cudaGraphNode_t cn; CK(cudaGraphAddNode(&cn, cap, dp, nullptr, nd, &p));
  for (int v = 0; v < 3; v++) {
    cudaGraph_t body = p.conditional.phGraph_out[v]; cudaStream_t bs; CK(cudaStreamCreate(&bs));
    CK(cudaStreamBeginCaptureToGraph(bs, body, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
    mark<<<1, 32, 0, bs>>>(d_o, 100 * (v + 1));
    cudaGraph_t tmp; CK(cudaStreamEndCapture(bs, &tmp));
  }
  CK(cudaStreamUpdateCaptureDependencies(s, &cn, nullptr, 1, cudaStreamSetCaptureDependencies));
  CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));

  int one = 1, zero = 0, two = 2, out;
  printf("  [%s] default branch = %u\n", tag, dflt);
  // round 1: set it to body 2
  CK(cudaMemcpy(d_sel, &two, 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_go, &one, 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(d_o, 0, 4));
  CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
  CK(cudaMemcpy(&out, d_o, 4, cudaMemcpyDeviceToHost));
  printf("    set body 2 and run         -> %d (expected 300)\n", out);
  // rounds 2 and 3: the planner exits early, the condition is not set
  for (int r = 0; r < 2; r++) {
    CK(cudaMemcpy(d_go, &zero, 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_o, 0, 4));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(&out, d_o, 4, cudaMemcpyDeviceToHost));
    printf("    condition not set, replay %d -> %d  %s\n", r + 1, out,
           out == 300 ? "(last selection persisted)"
                      : (out == (int)(100 * (dflt + 1)) ? "**back to the default branch**" : "**something else**"));
  }
three:
  printf("=== sites: %s ===\n", getenv("NOP") ? getenv("NOP") : "8");
  {
    // The same work (one mark kernel), three wrappings:
    //   bare            -- placed directly in the main graph
    //   SWITCH          -- placed in a body of a 3-body conditional
    //   preset+disable  -- three sibling nodes, only one enabled
    // The selection never changes, so the measured difference is "the standing per-replay cost of carrying this mechanism".
    int *d_sel, *d_go, *d_o;
    CK(cudaMalloc(&d_sel, 4)); CK(cudaMalloc(&d_go, 4)); CK(cudaMalloc(&d_o, 4));
    int one = 1, zero = 0;
    CK(cudaMemcpy(d_sel, &zero, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_go, &one, 4, cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    const int NOP = getenv("NOP") ? atoi(getenv("NOP")) : 8;
    const int REPS = 300;

    auto timeit = [&](cudaGraphExec_t ge) {
      for (int i = 0; i < 20; i++) cudaGraphLaunch(ge, s);
      cudaStreamSynchronize(s);
      float best = 1e9f;
      for (int t = 0; t < 5; t++) {
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a, s);
        for (int i = 0; i < REPS; i++) cudaGraphLaunch(ge, s);
        cudaEventRecord(b, s); cudaStreamSynchronize(s);
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (ms * 1000 / REPS < best) best = ms * 1000 / REPS;
      }
      return best; };

    cudaGraph_t g1; cudaGraphExec_t e1;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < NOP; k++) mark<<<1, 32, 0, s>>>(d_o, k);
    CK(cudaStreamEndCapture(s, &g1)); CK(cudaGraphInstantiate(&e1, g1, 0));
    float t_plain = timeit(e1);

    cudaGraph_t g2; cudaGraphExec_t e2;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < NOP; k++) {
      cudaGraph_t cap; const cudaGraphNode_t* dp; size_t nd; cudaStreamCaptureStatus st;
      CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &cap, &dp, nullptr, &nd));
      cudaGraphConditionalHandle h;
      CK(cudaGraphConditionalHandleCreate(&h, cap, 0, cudaGraphCondAssignDefault));
      setcond<<<1, 32, 0, s>>>(h, d_sel, d_go);
      CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &cap, &dp, nullptr, &nd));
      cudaGraphNodeParams p = {}; p.type = cudaGraphNodeTypeConditional;
      p.conditional.handle = h; p.conditional.type = cudaGraphCondTypeSwitch; p.conditional.size = 3;
      cudaGraphNode_t cn; CK(cudaGraphAddNode(&cn, cap, dp, nullptr, nd, &p));
      for (int v = 0; v < 3; v++) {
        cudaGraph_t body = p.conditional.phGraph_out[v]; cudaStream_t bs; CK(cudaStreamCreate(&bs));
        CK(cudaStreamBeginCaptureToGraph(bs, body, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
        mark<<<1, 32, 0, bs>>>(d_o, k);
        cudaGraph_t tmp; CK(cudaStreamEndCapture(bs, &tmp));
      }
      CK(cudaStreamUpdateCaptureDependencies(s, &cn, nullptr, 1, cudaStreamSetCaptureDependencies));
    }
    CK(cudaStreamEndCapture(s, &g2)); CK(cudaGraphInstantiate(&e2, g2, 0));
    float t_switch = timeit(e2);

    cudaGraph_t g3; cudaGraphExec_t e3;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < NOP; k++) for (int v = 0; v < 3; v++) mark<<<1, 32, 0, s>>>(d_o, k);
    CK(cudaStreamEndCapture(s, &g3)); CK(cudaGraphInstantiate(&e3, g3, 0));
    { size_t n = 0; CK(cudaGraphGetNodes(g3, nullptr, &n));
      cudaGraphNode_t nds[64]; CK(cudaGraphGetNodes(g3, nds, &n));
      for (size_t i = 0; i < n; i++) CK(cudaGraphNodeSetEnabled(e3, nds[i], (i % 3) == 0)); }
    float t_pre = timeit(e3);

    printf("  %d dispatch sites, selection fixed, replay only:\n", NOP);
    printf("    bare (no mechanism)       %7.2f us/replay\n", t_plain);
    printf("    SWITCH conditional        %7.2f us/replay  -> +%.2f us per site\n",
           t_switch, (t_switch - t_plain) / NOP);
    printf("    preset 3, enable 1        %7.2f us/replay  -> +%.2f us per site (2 disabled nodes)\n",
           t_pre, (t_pre - t_plain) / NOP);

    // SWITCH, but only one planner kernel is launched (the form DynaGraph uses)
    cudaGraph_t g4; cudaGraphExec_t e4;
    Hs hs{};
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    { cudaGraph_t cap; const cudaGraphNode_t* dp; size_t nd; cudaStreamCaptureStatus st;
      CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &cap, &dp, nullptr, &nd));
      for (int k = 0; k < NOP; k++)
        CK(cudaGraphConditionalHandleCreate(&hs.h[k], cap, 0, cudaGraphCondAssignDefault));
      setcond_all<<<1, 32, 0, s>>>(hs, NOP, d_sel);
      for (int k = 0; k < NOP; k++) {
        CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &cap, &dp, nullptr, &nd));
        cudaGraphNodeParams p = {}; p.type = cudaGraphNodeTypeConditional;
        p.conditional.handle = hs.h[k]; p.conditional.type = cudaGraphCondTypeSwitch; p.conditional.size = 3;
        cudaGraphNode_t cn; CK(cudaGraphAddNode(&cn, cap, dp, nullptr, nd, &p));
        for (int v = 0; v < 3; v++) {
          cudaGraph_t body = p.conditional.phGraph_out[v]; cudaStream_t bs; CK(cudaStreamCreate(&bs));
          CK(cudaStreamBeginCaptureToGraph(bs, body, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
          mark<<<1, 32, 0, bs>>>(d_o, k);
          cudaGraph_t tmp; CK(cudaStreamEndCapture(bs, &tmp));
        }
        CK(cudaStreamUpdateCaptureDependencies(s, &cn, nullptr, 1, cudaStreamSetCaptureDependencies));
      } }
    CK(cudaStreamEndCapture(s, &g4)); CK(cudaGraphInstantiate(&e4, g4, 0));
    float t_sw1 = timeit(e4);
    printf("    SWITCH + single planner   %7.2f us/replay  -> +%.2f us per site (the per-site kernel removed)\n",
           t_sw1, (t_sw1 - t_plain) / NOP);
  }
  return 0;
}

int main() {
  cuInit(0);
  printf("=== 1. Does a conditional's value persist across replays ===\n");
  if (test_persist(0, "default0")) return 1;
  if (test_persist(1, "default1")) return 1;

  printf("\n=== 2. With the kernel node disabled, does the memset in the same branch still run ===\n");
  {
    int *d_o; CK(cudaMalloc(&d_o, 8));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    // variant A: mark(o, 777)      -- just one kernel
    // variant B: memset(o, 0) + mark(o,999) -- one memset and one kernel, B comes after A
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    mark<<<1, 32, 0, s>>>(d_o, 777);
    CK(cudaMemsetAsync(d_o, 0, 4, s));
    mark<<<1, 32, 0, s>>>(d_o, 999);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    size_t n = 0; CK(cudaGraphGetNodes(g, nullptr, &n));
    cudaGraphNode_t nds[8]; CK(cudaGraphGetNodes(g, nds, &n));
    printf("  graph has %zu nodes: ", n);
    for (size_t i = 0; i < n; i++) { cudaGraphNodeType t; cudaGraphNodeGetType(nds[i], &t); printf("%d ", (int)t); }
    printf(" (0=Kernel 2=Memset)\n");
    // select variant A: disable B's **kernel**, leave the memset alone (exactly the kernel-only approach)
    for (size_t i = 0; i < n; i++) {
      cudaGraphNodeType t; cudaGraphNodeGetType(nds[i], &t);
      if (t == cudaGraphNodeTypeKernel && i == n - 1) CK(cudaGraphNodeSetEnabled(ge, nds[i], 0));
    }
    int out;
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(&out, d_o, 4, cudaMemcpyDeviceToHost));
    printf("    only B's kernel disabled   -> %d  %s\n", out,
           out == 777 ? "(memset did not run either?)" : "**memset still runs, A's result wiped to 0**");
    // can the memset be disabled too
    for (size_t i = 0; i < n; i++) {
      cudaGraphNodeType t; cudaGraphNodeGetType(nds[i], &t);
      if (t == cudaGraphNodeTypeMemset) {
        cudaError_t e = cudaGraphNodeSetEnabled(ge, nds[i], 0);
        printf("    host-side disable Memset   -> %s\n", e == cudaSuccess ? "allowed" : cudaGetErrorString(e));
      }
    }
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(&out, d_o, 4, cudaMemcpyDeviceToHost));
    printf("    kernel+memset both disabled -> %d %s\n", out, out == 777 ? "(A's result preserved)" : "");
  }
  return 0;
}
