// Graph introspection probe: dump every node of a cudaGraph_t, with
// CUfunction identity (mangled name), grid/block, and raw parameter bytes
// decoded via cuFuncGetParamInfo.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <cstring>
#include <cstdio>

#define DRV(x) do { CUresult _r = (x); if (_r != CUDA_SUCCESS) { \
    const char* _s=nullptr; cuGetErrorString(_r,&_s); \
    throw std::runtime_error(std::string(#x)+" failed: "+(_s?_s:"?")); } } while(0)

static const char* node_type_name(CUgraphNodeType t) {
  switch (t) {
    case CU_GRAPH_NODE_TYPE_KERNEL: return "KERNEL";
    case CU_GRAPH_NODE_TYPE_MEMCPY: return "MEMCPY";
    case CU_GRAPH_NODE_TYPE_MEMSET: return "MEMSET";
    case CU_GRAPH_NODE_TYPE_HOST: return "HOST";
    case CU_GRAPH_NODE_TYPE_GRAPH: return "CHILD_GRAPH";
    case CU_GRAPH_NODE_TYPE_EMPTY: return "EMPTY";
    case CU_GRAPH_NODE_TYPE_MEM_ALLOC: return "MEM_ALLOC";
    case CU_GRAPH_NODE_TYPE_MEM_FREE: return "MEM_FREE";
    default: return "OTHER";
  }
}

// Return all nodes of the graph, in a deterministic topological order.
static std::vector<CUgraphNode> topo_nodes(CUgraph g) {
  size_t n = 0;
  DRV(cuGraphGetNodes(g, nullptr, &n));
  std::vector<CUgraphNode> all(n);
  if (n) DRV(cuGraphGetNodes(g, all.data(), &n));

  // build indegree
  std::map<CUgraphNode, int> indeg;
  for (auto nd : all) indeg[nd] = 0;
  for (auto nd : all) {
    size_t nd_count = 0;
    DRV(cuGraphNodeGetDependentNodes(nd, nullptr, nullptr, &nd_count));
    std::vector<CUgraphNode> deps(nd_count);
    if (nd_count) DRV(cuGraphNodeGetDependentNodes(nd, deps.data(), nullptr, &nd_count));
    for (auto d : deps) indeg[d] += 1;
  }
  // Kahn, with ties broken by the order cuGraphGetNodes returned (stable).
  std::map<CUgraphNode, int> orig;
  for (size_t i = 0; i < all.size(); ++i) orig[all[i]] = (int)i;
  std::vector<CUgraphNode> out;
  std::set<std::pair<int, CUgraphNode>> ready;
  for (auto& kv : indeg) if (kv.second == 0) ready.insert({orig[kv.first], kv.first});
  while (!ready.empty()) {
    auto it = ready.begin();
    CUgraphNode nd = it->second;
    ready.erase(it);
    out.push_back(nd);
    size_t nd_count = 0;
    DRV(cuGraphNodeGetDependentNodes(nd, nullptr, nullptr, &nd_count));
    std::vector<CUgraphNode> deps(nd_count);
    if (nd_count) DRV(cuGraphNodeGetDependentNodes(nd, deps.data(), nullptr, &nd_count));
    for (auto d : deps) if (--indeg[d] == 0) ready.insert({orig[d], d});
  }
  if (out.size() != all.size()) return all;  // cycle shouldn't happen
  return out;
}

static py::dict describe_node(CUgraphNode nd) {
  py::dict d;
  CUgraphNodeType t;
  DRV(cuGraphNodeGetType(nd, &t));
  d["type"] = node_type_name(t);
  d["node_handle"] = (uint64_t)(uintptr_t)nd;
  if (t != CU_GRAPH_NODE_TYPE_KERNEL) {
    if (t == CU_GRAPH_NODE_TYPE_MEMSET) {
      CUDA_MEMSET_NODE_PARAMS mp;
      if (cuGraphMemsetNodeGetParams(nd, &mp) == CUDA_SUCCESS) {
        d["memset_width"] = (uint64_t)mp.width;
        d["memset_height"] = (uint64_t)mp.height;
        d["memset_elemsize"] = (uint64_t)mp.elementSize;
        d["memset_value"] = (uint64_t)mp.value;
        d["memset_dst"] = (uint64_t)mp.dst;
      }
    } else if (t == CU_GRAPH_NODE_TYPE_MEMCPY) {
      CUDA_MEMCPY3D cp;
      if (cuGraphMemcpyNodeGetParams(nd, &cp) == CUDA_SUCCESS) {
        d["memcpy_bytes"] = (uint64_t)(cp.WidthInBytes * (cp.Height ? cp.Height : 1) *
                                       (cp.Depth ? cp.Depth : 1));
        d["memcpy_srckind"] = (int)cp.srcMemoryType;
        d["memcpy_dstkind"] = (int)cp.dstMemoryType;
      }
    }
    return d;
  }

  CUDA_KERNEL_NODE_PARAMS p;
  std::memset(&p, 0, sizeof(p));
  DRV(cuGraphKernelNodeGetParams(nd, &p));
  CUfunction f = p.func;
  d["func"] = (uint64_t)(uintptr_t)f;
  d["grid"] = py::make_tuple(p.gridDimX, p.gridDimY, p.gridDimZ);
  d["block"] = py::make_tuple(p.blockDimX, p.blockDimY, p.blockDimZ);
  d["smem"] = (uint64_t)p.sharedMemBytes;
  d["has_extra"] = (p.extra != nullptr);
  d["has_kernelParams"] = (p.kernelParams != nullptr);

  const char* nm = nullptr;
  if (f && cuFuncGetName(&nm, f) == CUDA_SUCCESS && nm) d["name"] = std::string(nm);
  else d["name"] = std::string("<unknown>");

  CUmodule mod = nullptr;
  if (f && cuFuncGetModule(&mod, f) == CUDA_SUCCESS) d["module"] = (uint64_t)(uintptr_t)mod;
  else d["module"] = (uint64_t)0;

  // Enumerate formal parameters
  py::list params;
  if (f) {
    for (size_t i = 0; i < 256; ++i) {
      size_t off = 0, sz = 0;
      CUresult r = cuFuncGetParamInfo(f, i, &off, &sz);
      if (r != CUDA_SUCCESS) break;
      py::dict pi;
      pi["index"] = (uint64_t)i;
      pi["offset"] = (uint64_t)off;
      pi["size"] = (uint64_t)sz;
      if (p.kernelParams && p.kernelParams[i]) {
        pi["bytes"] = py::bytes((const char*)p.kernelParams[i], sz);
      } else if (p.extra) {
        pi["bytes"] = py::none();
      } else {
        pi["bytes"] = py::none();
      }
      params.append(pi);
    }
  }
  d["params"] = params;
  return d;
}

std::vector<py::dict> dump_graph(uint64_t graph_ptr) {
  CUgraph g = (CUgraph)(uintptr_t)graph_ptr;
  auto nodes = topo_nodes(g);
  std::vector<py::dict> out;
  out.reserve(nodes.size());
  for (auto nd : nodes) out.push_back(describe_node(nd));
  return out;
}

// --- mid-capture node attribution (question A) ---
// Returns the set of node handles currently in the graph being captured
// on the current torch stream.
std::vector<uint64_t> capture_node_set(uint64_t stream_ptr) {
  cudaStream_t s = (cudaStream_t)(uintptr_t)stream_ptr;
  CUstreamCaptureStatus st;
  cuuint64_t id = 0;
  CUgraph g = nullptr;
  const CUgraphNode* deps = nullptr;
  const CUgraphEdgeData* ed = nullptr;
  size_t ndeps = 0;
  DRV(cuStreamGetCaptureInfo(s, &st, &id, &g, &deps, &ed, &ndeps));
  std::vector<uint64_t> out;
  if (st != CU_STREAM_CAPTURE_STATUS_ACTIVE || g == nullptr) return out;
  size_t n = 0;
  DRV(cuGraphGetNodes(g, nullptr, &n));
  std::vector<CUgraphNode> all(n);
  if (n) DRV(cuGraphGetNodes(g, all.data(), &n));
  for (auto nd : all) out.push_back((uint64_t)(uintptr_t)nd);
  return out;
}

// graph handle mid-capture
uint64_t capture_graph_handle(uint64_t stream_ptr) {
  cudaStream_t s = (cudaStream_t)(uintptr_t)stream_ptr;
  CUstreamCaptureStatus st; cuuint64_t id = 0; CUgraph g = nullptr;
  const CUgraphNode* deps = nullptr; const CUgraphEdgeData* ed = nullptr; size_t ndeps = 0;
  DRV(cuStreamGetCaptureInfo(s, &st, &id, &g, &deps, &ed, &ndeps));
  return (uint64_t)(uintptr_t)g;
}

py::dict describe_node_handle(uint64_t h) { return describe_node((CUgraphNode)(uintptr_t)h); }

// Patch a kernel node's params in an instantiated exec graph.
// new_params: full flat byte buffer laid out at the formal param offsets.
void set_exec_kernel_params(uint64_t exec_ptr, uint64_t node_handle,
                            py::bytes flat, unsigned gx, unsigned gy, unsigned gz,
                            int bx, int by, int bz, long long smem) {
  CUgraphExec ex = (CUgraphExec)(uintptr_t)exec_ptr;
  CUgraphNode nd = (CUgraphNode)(uintptr_t)node_handle;
  CUDA_KERNEL_NODE_PARAMS p; std::memset(&p, 0, sizeof(p));
  DRV(cuGraphKernelNodeGetParams(nd, &p));
  CUfunction f = p.func;
  std::string buf = flat;
  std::vector<void*> argp;
  for (size_t i = 0; i < 256; ++i) {
    size_t off = 0, sz = 0;
    if (cuFuncGetParamInfo(f, i, &off, &sz) != CUDA_SUCCESS) break;
    argp.push_back((void*)(buf.data() + off));
  }
  p.kernelParams = argp.data();
  p.extra = nullptr;
  if (gx) { p.gridDimX = gx; p.gridDimY = gy; p.gridDimZ = gz; }
  if (bx > 0) { p.blockDimX = bx; p.blockDimY = by; p.blockDimZ = bz; }
  if (smem >= 0) { p.sharedMemBytes = (unsigned)smem; }
  DRV(cuGraphExecKernelNodeSetParams(ex, nd, &p));
}

uint64_t param_buffer_size(uint64_t node_handle) {
  CUgraphNode nd = (CUgraphNode)(uintptr_t)node_handle;
  CUDA_KERNEL_NODE_PARAMS p; std::memset(&p, 0, sizeof(p));
  DRV(cuGraphKernelNodeGetParams(nd, &p));
  size_t total = 0;
  for (size_t i = 0; i < 256; ++i) {
    size_t off = 0, sz = 0;
    if (cuFuncGetParamInfo(p.func, i, &off, &sz) != CUDA_SUCCESS) break;
    if (off + sz > total) total = off + sz;
  }
  return (uint64_t)total;
}


// Copy node B's params onto node A inside an instantiated exec graph,
// dispatching on node type (kernel / memcpy / memset).
std::string copy_node_params(uint64_t exec_ptr, uint64_t a_h, uint64_t b_h, bool force=false) {
  CUgraphExec ex = (CUgraphExec)(uintptr_t)exec_ptr;
  CUgraphNode a = (CUgraphNode)(uintptr_t)a_h, b = (CUgraphNode)(uintptr_t)b_h;
  CUgraphNodeType ta, tb;
  if (cuGraphNodeGetType(a,&ta)||cuGraphNodeGetType(b,&tb)) return "getType failed";
  if (ta != tb) return "node type mismatch";
  CUresult r = CUDA_SUCCESS;
  const char* es = nullptr;
  if (ta == CU_GRAPH_NODE_TYPE_KERNEL) {
    CUDA_KERNEL_NODE_PARAMS pb; std::memset(&pb,0,sizeof(pb));
    DRV(cuGraphKernelNodeGetParams(b,&pb));
    CUDA_KERNEL_NODE_PARAMS pa; std::memset(&pa,0,sizeof(pa));
    DRV(cuGraphKernelNodeGetParams(a,&pa));
    if (pa.func != pb.func && !force) return "DIFFERENT_FUNC";
    // copy argument bytes out of B into our own buffer, then point at it
    std::vector<size_t> offs, szs; size_t total=0;
    for (size_t i=0;i<256;++i){ size_t o,z; if(cuFuncGetParamInfo(pb.func,i,&o,&z)!=CUDA_SUCCESS) break;
                                offs.push_back(o); szs.push_back(z); if(o+z>total) total=o+z; }
    std::vector<char> buf(total);
    std::vector<void*> argp(offs.size());
    for (size_t i=0;i<offs.size();++i){
      if (pb.kernelParams && pb.kernelParams[i]) std::memcpy(buf.data()+offs[i], pb.kernelParams[i], szs[i]);
      argp[i] = buf.data()+offs[i];
    }
    pb.kernelParams = argp.data(); pb.extra = nullptr;
    r = cuGraphExecKernelNodeSetParams(ex, a, &pb);
  } else if (ta == CU_GRAPH_NODE_TYPE_MEMCPY) {
    CUDA_MEMCPY3D cp; DRV(cuGraphMemcpyNodeGetParams(b,&cp));
    CUcontext ctx; cuCtxGetCurrent(&ctx);
    r = cuGraphExecMemcpyNodeSetParams(ex, a, &cp, ctx);
  } else if (ta == CU_GRAPH_NODE_TYPE_MEMSET) {
    CUDA_MEMSET_NODE_PARAMS ms; DRV(cuGraphMemsetNodeGetParams(b,&ms));
    CUcontext ctx; cuCtxGetCurrent(&ctx);
    r = cuGraphExecMemsetNodeSetParams(ex, a, &ms, ctx);
  } else {
    return std::string("unsupported node type ") + node_type_name(ta);
  }
  if (r != CUDA_SUCCESS) { cuGetErrorString(r,&es); return std::string("ERR: ")+(es?es:"?"); }
  return "ok";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dump_graph", &dump_graph);
  m.def("capture_node_set", &capture_node_set);
  m.def("capture_graph_handle", &capture_graph_handle);
  m.def("describe_node_handle", &describe_node_handle);
  m.def("set_exec_kernel_params", &set_exec_kernel_params,
        py::arg("exec_ptr"), py::arg("node_handle"), py::arg("flat"),
        py::arg("gx")=0u, py::arg("gy")=1u, py::arg("gz")=1u,
        py::arg("bx")=-1, py::arg("by")=1, py::arg("bz")=1, py::arg("smem")=(long long)-1);
  m.def("param_buffer_size", &param_buffer_size);
  m.def("copy_node_params", &copy_node_params, py::arg("exec_ptr"), py::arg("a_h"), py::arg("b_h"), py::arg("force")=false);
}
