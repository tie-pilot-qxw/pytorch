// Can the nodes INSIDE a SWITCH body still be patched after instantiation?
//
// The layered design puts topology variation into a SWITCH (one body per
// topology) and shape variation into per-node patches. That only works if a
// node living inside a conditional body can be patched like a top-level one.
// Two routes, both needed:
//   (a) device side: the body's kernel launched device-updatable, and the
//       planner patches its grid from ctx in the same launch (what DynaGraph's
//       planner does today for top-level nodes);
//   (b) host side: cudaGraphExecKernelNodeSetParams on a body node of the
//       instantiated exec (what the extern route would use for cuBLAS/cuDNN
//       kernel-variant swaps).
//
//   nvcc -O2 -arch=sm_90a -rdc=true switch_patch.cu -o switch_patch -lcudadevrt
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

// ctx[0] = L (drives which body), ctx[1] = wanted grid.x for that body's node
__global__ void planner(cudaGraphConditionalHandle h, const cudaGraphDeviceNode_t* handles,
                        const int* ctx) {
  if (threadIdx.x != 0) return;
  int L = ctx[0];
  int v = L <= 512 ? 0 : (L <= 2048 ? 1 : 2);
  cudaGraphSetConditional(h, v);
  // (a): patch the selected body's node from the device
  cudaGraphKernelNodeSetGridDim(handles[v], dim3((unsigned)ctx[1], 1, 1));
}
__global__ void variant(int* out, int id) { if (threadIdx.x == 0) out[0] = id * 1000 + gridDim.x; }
__global__ void tail(int* out) { if (threadIdx.x == 0) out[1] = 77; }

int main() {
  int *d_ctx, *d_out;
  cudaGraphDeviceNode_t* d_handles;
  CK(cudaMalloc(&d_ctx, 8)); CK(cudaMalloc(&d_out, 8)); CK(cudaMemset(d_out, 0, 8));
  CK(cudaMalloc(&d_handles, 3 * sizeof(cudaGraphDeviceNode_t)));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaGraph_t g; cudaGraphExec_t ge;

  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  cudaGraph_t capGraph; const cudaGraphNode_t* deps; size_t ndeps; cudaStreamCaptureStatus st;
  CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &capGraph, &deps, nullptr, &ndeps));
  cudaGraphConditionalHandle h;
  CK(cudaGraphConditionalHandleCreate(&h, capGraph, 0, cudaGraphCondAssignDefault));
  planner<<<1, 32, 0, s>>>(h, d_handles, d_ctx);
  CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &capGraph, &deps, nullptr, &ndeps));
  cudaGraphNodeParams p = {};
  p.type = cudaGraphNodeTypeConditional; p.conditional.handle = h;
  p.conditional.type = cudaGraphCondTypeSwitch; p.conditional.size = 3;
  cudaGraphNode_t cnode; CK(cudaGraphAddNode(&cnode, capGraph, deps, nullptr, ndeps, &p));

  cudaGraphDeviceNode_t hs[3];
  cudaGraph_t bodies[3];
  for (int v = 0; v < 3; ++v) {
    cudaGraph_t body = p.conditional.phGraph_out[v];
    bodies[v] = body;
    cudaStream_t bs; CK(cudaStreamCreate(&bs));
    CK(cudaStreamBeginCaptureToGraph(bs, body, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
    // launched device-updatable so the planner can reach it
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeDeviceUpdatableKernelNode;
    attr.val.deviceUpdatableKernelNode.deviceUpdatable = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((v + 1) * 4); cfg.blockDim = dim3(64); cfg.stream = bs;
    cfg.attrs = &attr; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, variant, d_out, v));
    hs[v] = attr.val.deviceUpdatableKernelNode.devNode;
    CK(cudaStreamEndCapture(bs, &body));
  }
  CK(cudaStreamUpdateCaptureDependencies(s, &cnode, nullptr, 1, cudaStreamSetCaptureDependencies));
  tail<<<1, 32, 0, s>>>(d_out);
  CK(cudaStreamEndCapture(s, &g));
  // Handles are read at replay, not at capture, so this can wait until the
  // capture is closed -- a synchronous memcpy is illegal while it is open.
  CK(cudaMemcpy(d_handles, hs, sizeof hs, cudaMemcpyHostToDevice));
  CK(cudaGraphInstantiate(&ge, g, 0));

  int bad = 0;
  // (a) device-side: planner patches the selected body's grid to ctx[1]
  printf("== (a) device-side patch of a node inside a SWITCH body ==\n");
  int cases[][2] = {{100, 7}, {1000, 9}, {9000, 11}, {100, 13}, {2049, 5}};
  for (auto& c : cases) {
    CK(cudaMemcpy(d_ctx, c, 8, cudaMemcpyHostToDevice));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    int out[2]; CK(cudaMemcpy(out, d_out, 8, cudaMemcpyDeviceToHost));
    int ev = c[0] <= 512 ? 0 : (c[0] <= 2048 ? 1 : 2);
    bool ok = out[0] == ev * 1000 + c[1] && out[1] == 77;
    bad += !ok;
    printf("  L=%5d want grid %2d -> body %d ran grid %d  %s\n", c[0], c[1],
           out[0] / 1000, out[0] % 1000, ok ? "OK" : "MISMATCH");
  }

  // (b) host-side: cudaGraphExecKernelNodeSetParams on a body node
  printf("== (b) host-side cudaGraphExecKernelNodeSetParams on a body node ==\n");
  for (int v = 0; v < 3; ++v) {
    size_t n = 0; CK(cudaGraphGetNodes(bodies[v], nullptr, &n));
    cudaGraphNode_t nodes[8]; CK(cudaGraphGetNodes(bodies[v], nodes, &n));
    cudaKernelNodeParams np{};
    CK(cudaGraphKernelNodeGetParams(nodes[0], &np));
    unsigned want = 20 + v;
    np.gridDim = dim3(want, 1, 1);
    cudaError_t e = cudaGraphExecKernelNodeSetParams(ge, nodes[0], &np);
    if (e != cudaSuccess) {
      printf("  body %d: ExecKernelNodeSetParams -> %s\n", v, cudaGetErrorString(e));
      bad++;
      continue;
    }
    // choose this body but tell the planner to set the SAME grid so (a) does not mask (b)
    int c[2] = {v == 0 ? 100 : (v == 1 ? 1000 : 9000), (int)want};
    CK(cudaMemcpy(d_ctx, c, 8, cudaMemcpyHostToDevice));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    int out[2]; CK(cudaMemcpy(out, d_out, 8, cudaMemcpyDeviceToHost));
    bool ok = out[0] == v * 1000 + (int)want;
    bad += !ok;
    printf("  body %d: set grid %u via exec -> ran grid %d  %s\n", v, want, out[0] % 1000,
           ok ? "OK" : "MISMATCH");
  }
  // (b') host-side with the planner NOT overriding: set ctx[1] to something else
  // and confirm the device patch wins (it runs later in the same launch).
  printf("== (b') order: host set 30, device planner sets 31 in the same launch ==\n");
  {
    size_t n = 0; CK(cudaGraphGetNodes(bodies[0], nullptr, &n));
    cudaGraphNode_t nodes[8]; CK(cudaGraphGetNodes(bodies[0], nodes, &n));
    cudaKernelNodeParams np{}; CK(cudaGraphKernelNodeGetParams(nodes[0], &np));
    np.gridDim = dim3(30, 1, 1);
    CK(cudaGraphExecKernelNodeSetParams(ge, nodes[0], &np));
    int c[2] = {100, 31};
    CK(cudaMemcpy(d_ctx, c, 8, cudaMemcpyHostToDevice));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    int out[2]; CK(cudaMemcpy(out, d_out, 8, cudaMemcpyDeviceToHost));
    printf("  body 0 ran grid %d (device patch should win -> 31)  %s\n", out[0] % 1000,
           out[0] % 1000 == 31 ? "OK" : "MISMATCH");
    bad += out[0] % 1000 != 31;
  }
  printf("\n%s\n", bad ? "SOME MISMATCH" : "ALL OK");
  return bad ? 1 : 0;
}
