// Does batching the device-side node updates actually cost less?
//
// DynaGraph's generated planner calls SetEnabled + SetGridDim + SetParam per
// node, one call each. Profiling a real 48-node graph put `dynagraph_planner`
// at 43.7 us per replay -- 14.3% of the whole replay's device time -- i.e.
// 0.91 us/node. This project's own devupdate_scale.cu, which batches two
// updates per node through cudaGraphKernelNodeUpdatesApply, measured 3000 nodes
// at +148 us, i.e. 0.049 us/node. That is an 18x gap, but the two differ in
// more than batching (thread count, updates per node), so it is a lead and not
// a result. This measures the one variable.
//
//   nvcc -O2 -arch=sm_90a -rdc=true planner_batched.cu -o planner_batched -lcudadevrt
//
// Timing is with CUDA events, so it is device-side and unaffected by host load.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void worker(int* out, int n, int id) {
  if (threadIdx.x == 0 && blockIdx.x == 0) out[id] = n + gridDim.x;
}

// What DynaGraph generates today: separate calls, one per field.
__global__ void planner_separate(const cudaGraphDeviceNode_t* handles, int N,
                                 const long long* ctx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  long long L = ctx[0];
  int gx = (int)((L + 255) / 256);
  if (gx <= 0) { cudaGraphKernelNodeSetEnabled(handles[i], 0); return; }
  cudaGraphKernelNodeSetEnabled(handles[i], 1);
  cudaGraphKernelNodeSetGridDim(handles[i], dim3((unsigned)gx, 1, 1));
  int nn = (int)L * 3 + i;
  cudaGraphKernelNodeSetParam(handles[i], 8, &nn, sizeof(int));
}

// Same three updates, one batched call.
__global__ void planner_batched(const cudaGraphDeviceNode_t* handles, int N,
                                const long long* ctx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  long long L = ctx[0];
  int gx = (int)((L + 255) / 256);
  int nn = (int)L * 3 + i;
  __align__(16) char buf[3 * sizeof(cudaGraphKernelNodeUpdate)];
  cudaGraphKernelNodeUpdate* u = (cudaGraphKernelNodeUpdate*)buf;
  u[0].node = handles[i];
  u[0].field = cudaGraphKernelNodeFieldEnabled;
  u[0].updateData.isEnabled = gx > 0 ? 1u : 0u;
  if (gx <= 0) { cudaGraphKernelNodeUpdatesApply(u, 1); return; }
  u[1].node = handles[i];
  u[1].field = cudaGraphKernelNodeFieldGridDim;
  u[1].updateData.gridDim = dim3((unsigned)gx, 1, 1);
  u[2].node = handles[i];
  u[2].field = cudaGraphKernelNodeFieldParam;
  u[2].updateData.param.offset = 8;
  u[2].updateData.param.pValue = &nn;
  u[2].updateData.param.size = sizeof(int);
  cudaGraphKernelNodeUpdatesApply(u, 3);
}


// Does the same thing as mode 1 with the same number of update calls; the only difference is that the grid formula goes
// through a per-node switch -- exactly the shape of the planner DynaGraph generates. The 32 threads of a warp
// land in 32 different branches, so the cost is the sum of the branch bodies rather than the max.
__global__ void planner_switch(const cudaGraphDeviceNode_t* handles, int N,
                               const long long* ctx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  long long L = ctx[0];
  int gx = 1;
  switch (i % 32) {
    case 0: gx = (int)((L + 256) / 256); break;
    case 1: gx = (int)((L + 257) / 257); break;
    case 2: gx = (int)((L + 258) / 258); break;
    case 3: gx = (int)((L + 259) / 259); break;
    case 4: gx = (int)((L + 260) / 260); break;
    case 5: gx = (int)((L + 261) / 261); break;
    case 6: gx = (int)((L + 262) / 262); break;
    case 7: gx = (int)((L + 263) / 263); break;
    case 8: gx = (int)((L + 264) / 264); break;
    case 9: gx = (int)((L + 265) / 265); break;
    case 10: gx = (int)((L + 266) / 266); break;
    case 11: gx = (int)((L + 267) / 267); break;
    case 12: gx = (int)((L + 268) / 268); break;
    case 13: gx = (int)((L + 269) / 269); break;
    case 14: gx = (int)((L + 270) / 270); break;
    case 15: gx = (int)((L + 271) / 271); break;
    case 16: gx = (int)((L + 272) / 272); break;
    case 17: gx = (int)((L + 273) / 273); break;
    case 18: gx = (int)((L + 274) / 274); break;
    case 19: gx = (int)((L + 275) / 275); break;
    case 20: gx = (int)((L + 276) / 276); break;
    case 21: gx = (int)((L + 277) / 277); break;
    case 22: gx = (int)((L + 278) / 278); break;
    case 23: gx = (int)((L + 279) / 279); break;
    case 24: gx = (int)((L + 280) / 280); break;
    case 25: gx = (int)((L + 281) / 281); break;
    case 26: gx = (int)((L + 282) / 282); break;
    case 27: gx = (int)((L + 283) / 283); break;
    case 28: gx = (int)((L + 284) / 284); break;
    case 29: gx = (int)((L + 285) / 285); break;
    case 30: gx = (int)((L + 286) / 286); break;
    case 31: gx = (int)((L + 287) / 287); break;
    default: break;
  }
  if (gx <= 0) { cudaGraphKernelNodeSetEnabled(handles[i], 0); return; }
  cudaGraphKernelNodeSetEnabled(handles[i], 1);
  cudaGraphKernelNodeSetGridDim(handles[i], dim3((unsigned)gx, 1, 1));
  int nn = (int)L * 3 + i;
  cudaGraphKernelNodeSetParam(handles[i], 8, &nn, sizeof(int));
}


// Replicates the real shape of the generated planner: an outer per-node switch, where each case in turn calls
// dg_eval several times -- and dg_eval is itself a switch over all expressions.
// The divergence of the two levels multiplies.
__device__ __forceinline__ long long dg_eval2(int k, const long long* __restrict__ ctx) {
  switch (k) {
    case 0: return ctx[0] * 1 + 0;
    case 1: return ctx[0] * 2 + 1;
    case 2: return ctx[0] * 3 + 2;
    case 3: return ctx[0] * 4 + 3;
    case 4: return ctx[0] * 5 + 4;
    case 5: return ctx[0] * 6 + 5;
    case 6: return ctx[0] * 7 + 6;
    case 7: return ctx[0] * 8 + 7;
    case 8: return ctx[0] * 9 + 8;
    case 9: return ctx[0] * 10 + 9;
    case 10: return ctx[0] * 11 + 10;
    case 11: return ctx[0] * 12 + 11;
    case 12: return ctx[0] * 13 + 12;
    case 13: return ctx[0] * 14 + 13;
    case 14: return ctx[0] * 15 + 14;
    case 15: return ctx[0] * 16 + 15;
    case 16: return ctx[0] * 17 + 16;
    case 17: return ctx[0] * 18 + 17;
    case 18: return ctx[0] * 19 + 18;
    case 19: return ctx[0] * 20 + 19;
    case 20: return ctx[0] * 21 + 20;
    case 21: return ctx[0] * 22 + 21;
    case 22: return ctx[0] * 23 + 22;
    case 23: return ctx[0] * 24 + 23;
    case 24: return ctx[0] * 25 + 24;
    case 25: return ctx[0] * 26 + 25;
    case 26: return ctx[0] * 27 + 26;
    case 27: return ctx[0] * 28 + 27;
    case 28: return ctx[0] * 29 + 28;
    case 29: return ctx[0] * 30 + 29;
    case 30: return ctx[0] * 31 + 30;
    case 31: return ctx[0] * 32 + 31;
    case 32: return ctx[0] * 33 + 32;
    case 33: return ctx[0] * 34 + 33;
    case 34: return ctx[0] * 35 + 34;
    case 35: return ctx[0] * 36 + 35;
    case 36: return ctx[0] * 37 + 36;
    case 37: return ctx[0] * 38 + 37;
    case 38: return ctx[0] * 39 + 38;
    case 39: return ctx[0] * 40 + 39;
    case 40: return ctx[0] * 41 + 40;
    case 41: return ctx[0] * 42 + 41;
    case 42: return ctx[0] * 43 + 42;
    case 43: return ctx[0] * 44 + 43;
    case 44: return ctx[0] * 45 + 44;
    case 45: return ctx[0] * 46 + 45;
    case 46: return ctx[0] * 47 + 46;
    case 47: return ctx[0] * 48 + 47;
    case 48: return ctx[0] * 49 + 48;
    case 49: return ctx[0] * 50 + 49;
    case 50: return ctx[0] * 51 + 50;
    case 51: return ctx[0] * 52 + 51;
    case 52: return ctx[0] * 53 + 52;
    case 53: return ctx[0] * 54 + 53;
    case 54: return ctx[0] * 55 + 54;
    case 55: return ctx[0] * 56 + 55;
    case 56: return ctx[0] * 57 + 56;
    case 57: return ctx[0] * 58 + 57;
    case 58: return ctx[0] * 59 + 58;
    case 59: return ctx[0] * 60 + 59;
    case 60: return ctx[0] * 61 + 60;
    case 61: return ctx[0] * 62 + 61;
    case 62: return ctx[0] * 63 + 62;
    case 63: return ctx[0] * 64 + 63;
    default: return 0;
  }
}

__global__ void planner_nested(const cudaGraphDeviceNode_t* handles, int N,
                               const long long* ctx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  long long L = ctx[0];
  int gx = 1;
  long long a = 0, b = 0, c = 0, d = 0, e = 0;
  switch (i % 32) {
    case 0: gx = (int)((dg_eval2(0, ctx) + 255) / 256); a = dg_eval2(0, ctx); b = dg_eval2(0, ctx); c = dg_eval2(0, ctx); d = dg_eval2(0, ctx); e = dg_eval2(0, ctx); break;
    case 1: gx = (int)((dg_eval2(1, ctx) + 255) / 256); a = dg_eval2(7, ctx); b = dg_eval2(13, ctx); c = dg_eval2(3, ctx); d = dg_eval2(5, ctx); e = dg_eval2(11, ctx); break;
    case 2: gx = (int)((dg_eval2(2, ctx) + 255) / 256); a = dg_eval2(14, ctx); b = dg_eval2(26, ctx); c = dg_eval2(6, ctx); d = dg_eval2(10, ctx); e = dg_eval2(22, ctx); break;
    case 3: gx = (int)((dg_eval2(3, ctx) + 255) / 256); a = dg_eval2(21, ctx); b = dg_eval2(39, ctx); c = dg_eval2(9, ctx); d = dg_eval2(15, ctx); e = dg_eval2(33, ctx); break;
    case 4: gx = (int)((dg_eval2(4, ctx) + 255) / 256); a = dg_eval2(28, ctx); b = dg_eval2(52, ctx); c = dg_eval2(12, ctx); d = dg_eval2(20, ctx); e = dg_eval2(44, ctx); break;
    case 5: gx = (int)((dg_eval2(5, ctx) + 255) / 256); a = dg_eval2(35, ctx); b = dg_eval2(1, ctx); c = dg_eval2(15, ctx); d = dg_eval2(25, ctx); e = dg_eval2(55, ctx); break;
    case 6: gx = (int)((dg_eval2(6, ctx) + 255) / 256); a = dg_eval2(42, ctx); b = dg_eval2(14, ctx); c = dg_eval2(18, ctx); d = dg_eval2(30, ctx); e = dg_eval2(2, ctx); break;
    case 7: gx = (int)((dg_eval2(7, ctx) + 255) / 256); a = dg_eval2(49, ctx); b = dg_eval2(27, ctx); c = dg_eval2(21, ctx); d = dg_eval2(35, ctx); e = dg_eval2(13, ctx); break;
    case 8: gx = (int)((dg_eval2(8, ctx) + 255) / 256); a = dg_eval2(56, ctx); b = dg_eval2(40, ctx); c = dg_eval2(24, ctx); d = dg_eval2(40, ctx); e = dg_eval2(24, ctx); break;
    case 9: gx = (int)((dg_eval2(9, ctx) + 255) / 256); a = dg_eval2(63, ctx); b = dg_eval2(53, ctx); c = dg_eval2(27, ctx); d = dg_eval2(45, ctx); e = dg_eval2(35, ctx); break;
    case 10: gx = (int)((dg_eval2(10, ctx) + 255) / 256); a = dg_eval2(6, ctx); b = dg_eval2(2, ctx); c = dg_eval2(30, ctx); d = dg_eval2(50, ctx); e = dg_eval2(46, ctx); break;
    case 11: gx = (int)((dg_eval2(11, ctx) + 255) / 256); a = dg_eval2(13, ctx); b = dg_eval2(15, ctx); c = dg_eval2(33, ctx); d = dg_eval2(55, ctx); e = dg_eval2(57, ctx); break;
    case 12: gx = (int)((dg_eval2(12, ctx) + 255) / 256); a = dg_eval2(20, ctx); b = dg_eval2(28, ctx); c = dg_eval2(36, ctx); d = dg_eval2(60, ctx); e = dg_eval2(4, ctx); break;
    case 13: gx = (int)((dg_eval2(13, ctx) + 255) / 256); a = dg_eval2(27, ctx); b = dg_eval2(41, ctx); c = dg_eval2(39, ctx); d = dg_eval2(1, ctx); e = dg_eval2(15, ctx); break;
    case 14: gx = (int)((dg_eval2(14, ctx) + 255) / 256); a = dg_eval2(34, ctx); b = dg_eval2(54, ctx); c = dg_eval2(42, ctx); d = dg_eval2(6, ctx); e = dg_eval2(26, ctx); break;
    case 15: gx = (int)((dg_eval2(15, ctx) + 255) / 256); a = dg_eval2(41, ctx); b = dg_eval2(3, ctx); c = dg_eval2(45, ctx); d = dg_eval2(11, ctx); e = dg_eval2(37, ctx); break;
    case 16: gx = (int)((dg_eval2(16, ctx) + 255) / 256); a = dg_eval2(48, ctx); b = dg_eval2(16, ctx); c = dg_eval2(48, ctx); d = dg_eval2(16, ctx); e = dg_eval2(48, ctx); break;
    case 17: gx = (int)((dg_eval2(17, ctx) + 255) / 256); a = dg_eval2(55, ctx); b = dg_eval2(29, ctx); c = dg_eval2(51, ctx); d = dg_eval2(21, ctx); e = dg_eval2(59, ctx); break;
    case 18: gx = (int)((dg_eval2(18, ctx) + 255) / 256); a = dg_eval2(62, ctx); b = dg_eval2(42, ctx); c = dg_eval2(54, ctx); d = dg_eval2(26, ctx); e = dg_eval2(6, ctx); break;
    case 19: gx = (int)((dg_eval2(19, ctx) + 255) / 256); a = dg_eval2(5, ctx); b = dg_eval2(55, ctx); c = dg_eval2(57, ctx); d = dg_eval2(31, ctx); e = dg_eval2(17, ctx); break;
    case 20: gx = (int)((dg_eval2(20, ctx) + 255) / 256); a = dg_eval2(12, ctx); b = dg_eval2(4, ctx); c = dg_eval2(60, ctx); d = dg_eval2(36, ctx); e = dg_eval2(28, ctx); break;
    case 21: gx = (int)((dg_eval2(21, ctx) + 255) / 256); a = dg_eval2(19, ctx); b = dg_eval2(17, ctx); c = dg_eval2(63, ctx); d = dg_eval2(41, ctx); e = dg_eval2(39, ctx); break;
    case 22: gx = (int)((dg_eval2(22, ctx) + 255) / 256); a = dg_eval2(26, ctx); b = dg_eval2(30, ctx); c = dg_eval2(2, ctx); d = dg_eval2(46, ctx); e = dg_eval2(50, ctx); break;
    case 23: gx = (int)((dg_eval2(23, ctx) + 255) / 256); a = dg_eval2(33, ctx); b = dg_eval2(43, ctx); c = dg_eval2(5, ctx); d = dg_eval2(51, ctx); e = dg_eval2(61, ctx); break;
    case 24: gx = (int)((dg_eval2(24, ctx) + 255) / 256); a = dg_eval2(40, ctx); b = dg_eval2(56, ctx); c = dg_eval2(8, ctx); d = dg_eval2(56, ctx); e = dg_eval2(8, ctx); break;
    case 25: gx = (int)((dg_eval2(25, ctx) + 255) / 256); a = dg_eval2(47, ctx); b = dg_eval2(5, ctx); c = dg_eval2(11, ctx); d = dg_eval2(61, ctx); e = dg_eval2(19, ctx); break;
    case 26: gx = (int)((dg_eval2(26, ctx) + 255) / 256); a = dg_eval2(54, ctx); b = dg_eval2(18, ctx); c = dg_eval2(14, ctx); d = dg_eval2(2, ctx); e = dg_eval2(30, ctx); break;
    case 27: gx = (int)((dg_eval2(27, ctx) + 255) / 256); a = dg_eval2(61, ctx); b = dg_eval2(31, ctx); c = dg_eval2(17, ctx); d = dg_eval2(7, ctx); e = dg_eval2(41, ctx); break;
    case 28: gx = (int)((dg_eval2(28, ctx) + 255) / 256); a = dg_eval2(4, ctx); b = dg_eval2(44, ctx); c = dg_eval2(20, ctx); d = dg_eval2(12, ctx); e = dg_eval2(52, ctx); break;
    case 29: gx = (int)((dg_eval2(29, ctx) + 255) / 256); a = dg_eval2(11, ctx); b = dg_eval2(57, ctx); c = dg_eval2(23, ctx); d = dg_eval2(17, ctx); e = dg_eval2(63, ctx); break;
    case 30: gx = (int)((dg_eval2(30, ctx) + 255) / 256); a = dg_eval2(18, ctx); b = dg_eval2(6, ctx); c = dg_eval2(26, ctx); d = dg_eval2(22, ctx); e = dg_eval2(10, ctx); break;
    case 31: gx = (int)((dg_eval2(31, ctx) + 255) / 256); a = dg_eval2(25, ctx); b = dg_eval2(19, ctx); c = dg_eval2(29, ctx); d = dg_eval2(27, ctx); e = dg_eval2(21, ctx); break;
    default: break;
  }
  if (gx <= 0) { cudaGraphKernelNodeSetEnabled(handles[i], 0); return; }
  cudaGraphKernelNodeSetEnabled(handles[i], 1);
  cudaGraphKernelNodeSetGridDim(handles[i], dim3((unsigned)gx, 1, 1));
  int nn = (int)(L * 3 + i + a + b + c + d + e);
  cudaGraphKernelNodeSetParam(handles[i], 8, &nn, sizeof(int));
}

// mode: 0 = no planner at all (the floor), 1 = separate calls, 2 = batched
static int run(int N, int mode, float* us_out) {
  int *d_in, *d_out;
  cudaGraphDeviceNode_t* d_handles;
  CK(cudaMalloc(&d_in, sizeof(long long)));
  CK(cudaMalloc(&d_out, 4 * N));
  CK(cudaMalloc(&d_handles, sizeof(cudaGraphDeviceNode_t) * N));
  long long L = 4096;
  CK(cudaMemcpy(d_in, &L, sizeof(long long), cudaMemcpyHostToDevice));

  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  std::vector<cudaGraphDeviceNode_t> hs(N);

  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  if (mode == 1) planner_separate<<<(N + 255) / 256, 256, 0, s>>>(d_handles, N, (const long long*)d_in);
  if (mode == 2) planner_batched<<<(N + 255) / 256, 256, 0, s>>>(d_handles, N, (const long long*)d_in);
  if (mode == 3) planner_switch<<<(N + 255) / 256, 256, 0, s>>>(d_handles, N, (const long long*)d_in);
  if (mode == 4) planner_nested<<<(N + 255) / 256, 256, 0, s>>>(d_handles, N, (const long long*)d_in);
  for (int i = 0; i < N; ++i) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeDeviceUpdatableKernelNode;
    attr.val.deviceUpdatableKernelNode.deviceUpdatable = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(32);
    cfg.stream = s;
    cfg.attrs = &attr;
    cfg.numAttrs = mode ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, worker, d_out, 0, i));
    if (mode) hs[i] = attr.val.deviceUpdatableKernelNode.devNode;
  }
  CK(cudaStreamEndCapture(s, &g));
  if (mode) CK(cudaMemcpy(d_handles, hs.data(), sizeof(cudaGraphDeviceNode_t) * N,
                          cudaMemcpyHostToDevice));
  CK(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));

  for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(ge, s));
  CK(cudaStreamSynchronize(s));

  // Median of many single-replay event timings: one slow replay should not
  // move the number, and the mean would hide a bimodal distribution.
  std::vector<float> ms;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int i = 0; i < 50; ++i) {
    CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(e0, s));
    CK(cudaGraphLaunch(ge, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaStreamSynchronize(s));
    float t;
    CK(cudaEventElapsedTime(&t, e0, e1));
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  *us_out = ms[ms.size() / 2] * 1000.0f;

  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroy(g));
  CK(cudaStreamDestroy(s));
  CK(cudaFree(d_in));
  CK(cudaFree(d_out));
  CK(cudaFree(d_handles));
  return 0;
}

int main() {
  const char* names[5] = {"no planner (floor)", "uniform, separate calls",
                          "uniform, batched", "per-node switch",
                          "nested switch (real shape)"};
  printf("%6s  %-24s %10s %12s\n", "nodes", "variant", "us/replay", "us/node");
  for (int N : {48, 128}) {
    float base = 0;
    for (int mode = 0; mode < 5; ++mode) {
      float us = 0;
      if (run(N, mode, &us)) return 1;
      if (mode == 0) base = us;
      float per = mode ? (us - base) / N : 0.0f;
      printf("%6d  %-24s %10.1f %12s\n", N, names[mode], us,
             mode ? ([&]{ static char b[32]; snprintf(b, sizeof b, "%.3f", per); return b; }()) : "-");
    }
    printf("\n");
  }
  return 0;
}
