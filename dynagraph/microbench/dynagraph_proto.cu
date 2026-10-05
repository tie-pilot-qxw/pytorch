// DynaGraph prototype: one capture covers a whole shape range, zero re-recording at runtime.
//
// Earlier probes verified each primitive on its own (devupdate_test: SetGridDim/SetParam take effect in the same launch;
// devupdate_scale: UpdatesApply at scale; grid_zero: grid=0 is illegal, SetEnabled must be used;
// switch_*: conditional nodes). This is the first time they are **wired into one complete loop** and run against real baselines.
//
// Four modes, same shape sequence, same kernels:
//   A eager      N launches per step, exact shape                -- lower bound without graphs
//   B recapture  record a new graph for every new shape          -- what PyTorch does today
//   C pad2max    one static graph, always runs at n_max          -- dynamic=False + pad to max
//   D dynagraph  record once, planner patches node grids on device -- this project
//
// Criterion (set in FINDINGS): net headroom = min(pad waste, cudagraph speedup),
// and it must be computed against the **best** of {A, B, C}, not by picking the worst one.
//
// Build (no GPU needed):
//   nvcc -O3 -arch=sm_90a -o dynagraph_proto dynagraph_proto.cu
// Run (needs a card to itself):
//   ./dynagraph_proto --nodes 200 --steps 300 --min 4096 --max 262144

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static const int BLOCK = 256;

// Chained elementwise: out[i] = in[i]*1.0009f + 0.0001f, grid varies with n.
// Param layout (the planner patches n at this offset):
//   0: out  (8B)   8: in (8B)   16: n (4B)
__global__ void chain_k(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 1.0009f + 0.0001f;
}

// planner: reads this step's n from the InputContext on the device and rewrites each node's gridDim and n param.
// This is the "planner kernel" from the design doc, except the expressions are hand-written;
// in the real system it is generated mechanically from Inductor's ShapeEnv sympy expressions.
__global__ void planner_k(const cudaGraphDeviceNode_t* handles, int n_nodes,
                          const int* ctx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_nodes) return;
  int n = ctx[0];
  int g = (n + BLOCK - 1) / BLOCK;
  if (g <= 0) {
    // grid=0 is cudaErrorInvalidArgument (verified by grid_zero.cu), so the node must be disabled instead
    cudaGraphKernelNodeSetEnabled(handles[i], 0);
    return;
  }
  cudaGraphKernelNodeSetEnabled(handles[i], 1);
  cudaGraphKernelNodeSetGridDim(handles[i], dim3(g, 1, 1));
  cudaGraphKernelNodeSetParam(handles[i], 16, &n, sizeof(int));
}

struct Bufs { float *a, *b; int *ctx; cudaGraphDeviceNode_t* handles; };

static void build_chain(cudaStream_t s, Bufs& B, int n_nodes, int n,
                        std::vector<cudaGraphDeviceNode_t>* out_handles) {
  int grid = (n + BLOCK - 1) / BLOCK;
  for (int i = 0; i < n_nodes; ++i) {
    float* dst = (i & 1) ? B.b : B.a;
    float* src = (i & 1) ? B.a : B.b;
    if (out_handles) {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeDeviceUpdatableKernelNode;
      attr.val.deviceUpdatableKernelNode.deviceUpdatable = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(grid); cfg.blockDim = dim3(BLOCK);
      cfg.stream = s; cfg.attrs = &attr; cfg.numAttrs = 1;
      CK(cudaLaunchKernelEx(&cfg, chain_k, dst, src, n));
      out_handles->push_back(attr.val.deviceUpdatableKernelNode.devNode);
    } else {
      chain_k<<<grid, BLOCK, 0, s>>>(dst, src, n);
    }
  }
}

static double ms(std::chrono::steady_clock::time_point a,
                 std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
  int n_nodes = 200, steps = 300, nmin = 4096, nmax = 262144, seed = 0;
  for (int i = 1; i < argc - 1; ++i) {
    if (!strcmp(argv[i], "--nodes")) n_nodes = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps")) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--min")) nmin = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--max")) nmax = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed")) seed = atoi(argv[++i]);
  }
  printf("nodes=%d steps=%d n in [%d, %d]  BLOCK=%d\n", n_nodes, steps, nmin, nmax, BLOCK);

  // shape sequence: log-uniform, simulating a heavy-tailed workload (protein lengths and active voxel counts look like this)
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::vector<int> seq(steps);
  double lo = log((double)nmin), hi = log((double)nmax);
  for (int i = 0; i < steps; ++i) seq[i] = (int)exp(lo + U(rng) * (hi - lo));
  std::vector<int> uniq(seq); std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  double mean = 0; for (int v : seq) mean += v; mean /= steps;
  printf("distinct shapes = %zu / %d steps   pad2max waste = %.2fx (max/mean)\n\n",
         uniq.size(), steps, (double)nmax / mean);

  Bufs B{};
  CK(cudaMalloc(&B.a, (size_t)nmax * 4));
  CK(cudaMalloc(&B.b, (size_t)nmax * 4));
  CK(cudaMalloc(&B.ctx, 4));
  CK(cudaMalloc(&B.handles, sizeof(cudaGraphDeviceNode_t) * n_nodes));
  CK(cudaMemset(B.a, 0, (size_t)nmax * 4));
  cudaStream_t s; CK(cudaStreamCreate(&s));

  // after each chain runs, whether the result lands in a or b depends on the parity of the node count
  auto result_buf = [&]() { return (n_nodes & 1) ? B.b : B.a; };
  // Reset the input to a known pattern that does not depend on n. **Must reset before every comparison** --
  // all four modes share the same buffer pair, and the chained multiply means the next mode does not start where the previous one did;
  // without a reset you are comparing two different inputs.
  auto reset_input = [&](int n) {
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) h[i] = (float)((i % 97) * 0.01);
    CK(cudaMemcpy(B.a, h.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(B.b, 0, (size_t)nmax * 4));
  };
  auto read_out = [&](int n) {
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), result_buf(), (size_t)n * 4, cudaMemcpyDeviceToHost));
    return h;
  };

  // ---------------- A: eager ----------------
  for (int r = 0; r < 3; ++r) build_chain(s, B, n_nodes, seq[0], nullptr);
  CK(cudaStreamSynchronize(s));
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < steps; ++i) build_chain(s, B, n_nodes, seq[i], nullptr);
  CK(cudaStreamSynchronize(s));
  double t_eager = ms(t0, std::chrono::steady_clock::now());

  // ---------------- B: re-record for every new shape ----------------
  {
    std::vector<int> keys; std::vector<cudaGraphExec_t> execs;
    int recaptures = 0;
    CK(cudaStreamSynchronize(s));
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) {
      int n = seq[i];
      int idx = -1;
      for (size_t k = 0; k < keys.size(); ++k) if (keys[k] == n) { idx = (int)k; break; }
      if (idx < 0) {                                   // cache miss -> record a new graph
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        build_chain(s, B, n_nodes, n, nullptr);
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        keys.push_back(n); execs.push_back(ge); idx = (int)keys.size() - 1;
        ++recaptures;
      }
      CK(cudaGraphLaunch(execs[idx], s));
    }
    CK(cudaStreamSynchronize(s));
    double t_re = ms(t0, std::chrono::steady_clock::now());
    printf("B recapture   %8.2f ms   recorded %d graphs\n", t_re, recaptures);
    // save it for the final comparison
    setenv("_T_RE", std::to_string(t_re).c_str(), 1);
  }

  // ---------------- C: static graph padded to max ----------------
  double t_pad;
  {
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    build_chain(s, B, n_nodes, nmax, nullptr);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int r = 0; r < 3; ++r) CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    t_pad = ms(t0, std::chrono::steady_clock::now());
  }

  // ---------------- D: DynaGraph, record once ----------------
  double t_dyn;
  cudaGraphExec_t dyn_exec;            // kept for the correctness check later
  {
    std::vector<cudaGraphDeviceNode_t> handles;
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    planner_k<<<(n_nodes + 127) / 128, 128, 0, s>>>(B.handles, n_nodes, B.ctx);
    build_chain(s, B, n_nodes, seq[0], &handles);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaMemcpy(B.handles, handles.data(),
                  sizeof(cudaGraphDeviceNode_t) * n_nodes, cudaMemcpyHostToDevice));
    // a device-updatable graph can only be instantiated once (README:207/294)
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int r = 0; r < 3; ++r) {
      CK(cudaMemcpyAsync(B.ctx, &seq[0], 4, cudaMemcpyHostToDevice, s));
      CK(cudaGraphLaunch(ge, s));
    }
    CK(cudaStreamSynchronize(s));
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) {
      CK(cudaMemcpyAsync(B.ctx, &seq[i], 4, cudaMemcpyHostToDevice, s));
      CK(cudaGraphLaunch(ge, s));
    }
    CK(cudaStreamSynchronize(s));
    t_dyn = ms(t0, std::chrono::steady_clock::now());
    dyn_exec = ge;
  }

  double t_re = atof(getenv("_T_RE") ? getenv("_T_RE") : "0");
  printf("A eager       %8.2f ms\n", t_eager);
  printf("C pad2max     %8.2f ms\n", t_pad);
  printf("D dynagraph   %8.2f ms\n", t_dyn);
  double best = std::min(std::min(t_eager, t_re), t_pad);
  const char* bn = (best == t_eager) ? "eager" : (best == t_re ? "recapture" : "pad2max");
  printf("\nbest baseline is %s (%.2f ms)\n", bn, best);
  printf("=> DynaGraph vs best baseline: %.2fx\n", best / t_dyn);
  // ---------------- correctness: element-wise cross-check ----------------
  // This is what the prototype really has to prove: **one capture covers the whole range, and results match eager element by element**.
  printf("\ncorrectness cross-check (input reset for every n, eager vs dynagraph element by element)\n");
  int bad = 0;
  for (int n : {nmin, (nmin + nmax) / 2, nmax, nmin + 1, nmax - 7}) {
    if (n <= 0 || n > nmax) continue;
    reset_input(n);
    build_chain(s, B, n_nodes, n, nullptr);
    CK(cudaStreamSynchronize(s));
    auto ref = read_out(n);

    reset_input(n);
    CK(cudaMemcpyAsync(B.ctx, &n, 4, cudaMemcpyHostToDevice, s));
    CK(cudaGraphLaunch(dyn_exec, s));
    CK(cudaStreamSynchronize(s));
    auto got = read_out(n);

    double worst = 0; int at = -1;
    for (int i = 0; i < n; ++i) {
      double d = fabs((double)ref[i] - (double)got[i]);
      double rel = d / (fabs((double)ref[i]) + 1e-9);
      if (rel > worst) { worst = rel; at = i; }
    }
    bool ok = worst < 1e-5;
    if (!ok) ++bad;
    printf("  n=%-8d max rel err %.3e (at index %d)  %s\n", n, worst, at,
           ok ? "match" : "**MISMATCH**");
  }
  printf(bad ? "\nCorrectness FAILED: the mechanism is broken, ignore the timings for now.\n"
             : "\nAll correctness checks passed: one capture covered the whole range.\n");
  return bad ? 1 : 0;
}
