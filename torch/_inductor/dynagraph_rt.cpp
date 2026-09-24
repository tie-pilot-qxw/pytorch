// DynaGraph's per-call path, in C++.
//
// The Python runner (torch/_inductor/dynagraph.py) builds a region, checks its
// first shapes against eager, and captures a graph per launch structure. What
// a call needs after that is tables and arithmetic, and it happens here:
//
//  * a shape served before (`put` by the Python path, or made here): read the
//    key out of the inputs, copy the inputs held in stores, rewrite the inline
//    nodes if the graph holds another shape, run the host patcher (which
//    launches), write mutated inputs back, make the output tensors;
//  * a new shape, once the region has a program (`set_program`): evaluate the
//    layout and every geometry with the region's generated C, ask each inline
//    site's library for its launches through the C describe ABI
//    (torch/utils/_capture_launch.py), pick the graph captured with that
//    launch structure, and do the above -- then keep the shape.
//
// Whatever is not one of these (a store too small, a structure no graph was
// captured with, a library error) returns None before any side effect, and
// the Python path serves the call as before.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

// `dg_step` of a region's host patcher (`_HOST_TEMPLATE`).
using StepFn = int (*)(void*, const int64_t*, int, char*, const char* const*,
                       const void*, const char* const*, const char* const*,
                       void*, int);
// `dg_inline_set` (`_INLINE_SRC`).
using InlineFn = int (*)(void*, int, const void*, const void*, const unsigned*,
                         const char* const*, const int*, const int*, const int*,
                         const int*, const int*, const unsigned long long*);
// The region's generated layout and geometry (`_rt_geom_source`).
using GeomFn = void (*)(const int64_t*, int64_t*, int64_t*);
// cuFuncGetParamInfo.
using ParamInfoFn = int (*)(void*, size_t, size_t*, size_t*);

// The launch-describe ABI, v1.
struct dg_operand {
  uint64_t ptr;
  int32_t dtype;
  int32_t ndim;
  int64_t sizes[8];
  int64_t strides[8];
};
struct dg_launch {
  uint64_t func;
  uint32_t grid[3], block[3];
  uint32_t smem, cluster, pdl, nargs;
};
using DescribeFn = int (*)(const dg_operand*, int, const char*, dg_launch*, int,
                           char*, int64_t, uint32_t*, int, char*, int);

struct KeyHash {
  size_t operator()(const std::vector<int64_t>& v) const {
    size_t h = v.size();
    for (int64_t x : v)
      h ^= std::hash<int64_t>()(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

enum OutKind : int {
  kNone = 0,    // None
  kArena = 1,   // a tensor over the arena: byte offset, dtype, geometry
  kInput = 2,   // input `idx` itself
  kInView = 3,  // a view of input `idx`: geometry, element offset
  kFixed = 4,   // a tensor fixed for the region (an inline site's result)
};

struct Out {
  int kind = kNone;
  int64_t off = 0;
  int dt = 0;
  int idx = 0;
  std::vector<int64_t> sizes, strides;
  at::Tensor fixed;
};

// The inline node rewrite for one shape, owned (a shape made here).
struct InlineBatch {
  std::vector<uint64_t> nodes, funcs;
  std::vector<unsigned> dims;
  std::vector<std::string> blob_store;
  std::vector<const char*> blobs;
  std::vector<int> lens, nparam, poffs, npatch, pw;
  std::vector<unsigned long long> pv;
  // The pointer arrays, once the strings sit where they will stay.
  void seal() {
    blobs.clear();
    for (const auto& b : blob_store) blobs.push_back(b.data());
    npatch.assign(nodes.size(), 0);
    if (pw.empty()) pw.push_back(0);
    if (pv.empty()) pv.push_back(0);
    if (poffs.empty()) poffs.push_back(0);
  }
};

struct Entry {
  py::object ex;       // the Python `_Exec` serving this shape
  py::object token;    // what the graph's inline nodes are set for
  py::object applied;  // what `ex.inline_applied` becomes once they are
  StepFn step = nullptr;
  void* host = nullptr;
  const int64_t* syms = nullptr;  // from Python, or null for `syms_own`
  const char* const* ext = nullptr;
  const void* child = nullptr;
  std::vector<int64_t> syms_own;
  bool has_inline = false;
  void* exec = nullptr;
  int n_inline = 0;
  std::vector<uintptr_t> inl;  // the 10 arrays after n, from Python
  InlineBatch batch;           // or owned
  bool own_batch = false;
  int64_t total = 0;
  std::vector<Out> outs;
  py::object keep;
};

// One graph a new shape may be served by (`add_exec`).
struct ExecInfo {
  py::object ex;
  void* exec = nullptr;
  void* host = nullptr;
  StepFn step = nullptr;
  const char* const* ext = nullptr;
  const void* child = nullptr;
  std::vector<std::vector<uint64_t>> nodes;  // per inline site
  std::vector<std::vector<int>> clusters;    // per inline site
  py::object keep;
};

struct Operand {
  int kind;  // 0: arena slot, 1: input
  int slot, pos;
  int64_t itemsize;
  int dt;    // proto index (arena)
  int nd;
  int g_at;  // sizes, strides, element offset in the geometry; -1: the input's own
};

struct Site {
  DescribeFn describe;
  std::string statics;
  std::vector<Operand> ops;
};

// What a new shape takes (`set_program`).
struct Program {
  GeomFn geom = nullptr;
  int n_geom = 0, n_slots = 0;
  std::vector<int> sym_pos;  // input position of every symbol, in `symbols` order
  // per output: kind, slot (arena) or input position, itemsize, proto, rank,
  // where its geometry starts
  std::vector<int> o_kind, o_slot, o_dt, o_nd, o_gat;
  std::vector<int64_t> o_item;
  std::vector<at::Tensor> o_fixed;
  std::vector<Site> sites;
};

struct Described {
  uint64_t func;
  unsigned dims[7];
  int cluster;
  std::string blob;
  std::vector<int> offs;
};

class Region {
 public:
  Region(std::vector<int> sym_pos, std::vector<int> ext_read,
         std::vector<int> inplace, uintptr_t in_ptrs, std::vector<int> mut_copy,
         std::vector<int> mut_patch, uintptr_t inline_fn, int64_t device)
      : sym_pos_(std::move(sym_pos)),
        ext_read_(std::move(ext_read)),
        in_ptrs_(reinterpret_cast<int64_t*>(in_ptrs)),
        mut_copy_(std::move(mut_copy)),
        mut_patch_(std::move(mut_patch)),
        inline_fn_(reinterpret_cast<InlineFn>(inline_fn)),
        device_(device) {
    for (int j : inplace) {
      if ((int)inplace_.size() <= j) inplace_.resize(j + 1, 0);
      inplace_[j] = 1;
    }
  }

  void set_store(int j, at::Tensor t) {
    if ((int)stores_.size() <= j) stores_.resize(j + 1);
    stores_[j] = std::move(t);
  }

  void set_static(std::vector<int> pos, std::vector<int64_t> ptr) {
    static_pos_ = std::move(pos);
    static_ptr_ = std::move(ptr);
    static_of_.clear();
    for (size_t k = 0; k < static_pos_.size(); ++k) {
      int j = static_pos_[k];
      if ((int)static_of_.size() <= j) static_of_.resize(j + 1, -1);
      static_of_[j] = (int)k;
    }
  }

  int add_proto(at::Tensor t) {
    protos_.push_back(std::move(t));
    return (int)protos_.size() - 1;
  }

  void clear() { entries_.clear(); }
  size_t size() const { return entries_.size(); }
  // (served, handed back, handed back per reason, of the served: shapes made here)
  py::tuple stats() const { return py::make_tuple(hits_, misses_, why_, made_); }

  void put(std::vector<int64_t> key, py::object ex, py::object token,
           py::object applied, uintptr_t step, uintptr_t host, uintptr_t syms,
           uintptr_t ext, uintptr_t child, py::object inline_args,
           int64_t total, std::vector<int> kinds, std::vector<int64_t> offs,
           std::vector<int> dts, std::vector<int> idxs, std::vector<int> nds,
           std::vector<int64_t> sizes, std::vector<int64_t> strides,
           py::list fixed, py::object keep) {
    Entry e;
    e.ex = std::move(ex);
    e.token = std::move(token);
    e.applied = std::move(applied);
    e.step = reinterpret_cast<StepFn>(step);
    e.host = reinterpret_cast<void*>(host);
    e.syms = reinterpret_cast<const int64_t*>(syms);
    e.ext = reinterpret_cast<const char* const*>(ext);
    e.child = reinterpret_cast<const void*>(child);
    if (!inline_args.is_none()) {
      auto t = inline_args.cast<py::tuple>();
      e.has_inline = true;
      e.exec = reinterpret_cast<void*>(t[0].cast<uintptr_t>());
      e.n_inline = t[1].cast<int>();
      for (size_t k = 2; k < t.size(); ++k) e.inl.push_back(t[k].cast<uintptr_t>());
      TORCH_CHECK(e.inl.size() == 10, "dg_inline_set takes 10 arrays");
    }
    e.total = total;
    size_t k = 0;
    for (size_t i = 0; i < kinds.size(); ++i) {
      Out o;
      o.kind = kinds[i];
      o.off = offs[i];
      o.dt = dts[i];
      o.idx = idxs[i];
      o.sizes.assign(sizes.begin() + k, sizes.begin() + k + nds[i]);
      o.strides.assign(strides.begin() + k, strides.begin() + k + nds[i]);
      k += nds[i];
      if (o.kind == kFixed) o.fixed = fixed[i].cast<at::Tensor>();
      e.outs.push_back(std::move(o));
    }
    e.keep = std::move(keep);
    insert(std::move(key), std::move(e));
  }

  // What a new shape takes; see `Program`. Graphs are added with `add_exec`.
  void set_program(uintptr_t geom, int n_geom, int n_slots,
                   std::vector<int> sym_pos, std::vector<int> o_kind,
                   std::vector<int> o_slot, std::vector<int64_t> o_item,
                   std::vector<int> o_dt, std::vector<int> o_nd,
                   std::vector<int> o_gat, py::list o_fixed,
                   std::vector<uintptr_t> describe, std::vector<py::bytes> statics,
                   std::vector<int> n_ops, std::vector<int> op_kind,
                   std::vector<int> op_slot, std::vector<int> op_pos,
                   std::vector<int64_t> op_item, std::vector<int> op_dt,
                   std::vector<int> op_nd, std::vector<int> op_gat,
                   uintptr_t param_info) {
    Program p;
    p.geom = reinterpret_cast<GeomFn>(geom);
    p.n_geom = n_geom;
    p.n_slots = n_slots;
    p.sym_pos = std::move(sym_pos);
    p.o_kind = std::move(o_kind);
    p.o_slot = std::move(o_slot);
    p.o_item = std::move(o_item);
    p.o_dt = std::move(o_dt);
    p.o_nd = std::move(o_nd);
    p.o_gat = std::move(o_gat);
    for (size_t i = 0; i < p.o_kind.size(); ++i)
      p.o_fixed.push_back(p.o_kind[i] == kFixed ? o_fixed[i].cast<at::Tensor>()
                                                : at::Tensor());
    size_t k = 0;
    for (size_t s = 0; s < describe.size(); ++s) {
      Site site;
      site.describe = reinterpret_cast<DescribeFn>(describe[s]);
      site.statics = std::string(statics[s]);
      for (int q = 0; q < n_ops[s]; ++q, ++k)
        site.ops.push_back(Operand{op_kind[k], op_slot[k], op_pos[k], op_item[k],
                                   op_dt[k], op_nd[k], op_gat[k]});
      p.sites.push_back(std::move(site));
    }
    param_info_ = reinterpret_cast<ParamInfoFn>(param_info);
    prog_ = std::move(p);
    has_prog_ = true;
    execs_.clear();
  }

  void add_exec(py::object ex, uintptr_t exec, uintptr_t host, uintptr_t step,
                uintptr_t ext, uintptr_t child,
                std::vector<std::vector<uint64_t>> nodes,
                std::vector<std::vector<int>> clusters, py::object keep) {
    ExecInfo x;
    x.ex = std::move(ex);
    x.exec = reinterpret_cast<void*>(exec);
    x.host = reinterpret_cast<void*>(host);
    x.step = reinterpret_cast<StepFn>(step);
    x.ext = reinterpret_cast<const char* const*>(ext);
    x.child = reinterpret_cast<const void*>(child);
    x.nodes = std::move(nodes);
    x.clusters = std::move(clusters);
    x.keep = std::move(keep);
    execs_.push_back(std::move(x));
  }

  void clear_execs() { execs_.clear(); }
  size_t n_execs() const { return execs_.size(); }
  bool has_program() const { return has_prog_; }

  // (outputs, the `_Exec` that served them), (what, rc) when a CUDA call
  // failed after the inputs were copied, or None to be served by the Python
  // path, with nothing done.
  py::object call(py::list inputs, int64_t lane, const at::Tensor& arena) {
    in_ = inputs.ptr();
    n_in_ = PyList_GET_SIZE(in_);
    int why = plan_inputs();
    if (why >= 0) return miss(why);

    key_.clear();
    key_.push_back(lane);
    for (int p : sym_pos_) {
      if (p >= n_in_) return miss(3);
      PyObject* o = PyList_GET_ITEM(in_, p);
      if (!PyLong_CheckExact(o)) return miss(3);
      key_.push_back(PyLong_AsLongLong(o));
    }
    // The addresses the extern sites read, as they will be on this call: a
    // harvest (and an inline site's parameters) is keyed by them.
    for (int j : ext_read_) {
      int64_t v = in_ptrs_[j];
      if (j < (int)plan_ptr_.size() && plan_ptr_[j]) v = plan_ptr_[j];
      if (j < (int)static_of_.size() && static_of_[j] >= 0)
        v = reinterpret_cast<int64_t>(tensor_at(j)->data_ptr());
      key_.push_back(v);
    }
    if (!writes_back_ok()) return miss(6);
    auto it = entries_.find(key_);
    if (it != entries_.end()) {
      Entry& e = it->second;
      if (arena.numel() * arena.element_size() < e.total) return miss(5);
      ++hits_;
      return serve(e, arena);
    }
    if (!has_prog_ || execs_.empty()) return miss(4);
    return serve_new(arena);
  }

 private:
  const at::Tensor* tensor_at(int j) const {
    if (j >= n_in_) return nullptr;
    PyObject* o = PyList_GET_ITEM(in_, j);
    if (!THPVariable_Check(o)) return nullptr;
    return &THPVariable_Unpack(o);
  }

  // What every input held in a store will be read through on this call, and
  // which static inputs moved; -1, or why the call is handed back. Nothing
  // is done yet.
  int plan_inputs() {
    plan_ptr_.assign(stores_.size(), 0);
    copy_.clear();
    for (size_t j = 0; j < stores_.size(); ++j) {
      const at::Tensor& st = stores_[j];
      if (!st.defined()) continue;
      const at::Tensor* t = tensor_at((int)j);
      if (t == nullptr) continue;
      bool inpl = j < inplace_.size() && inplace_[j];
      int64_t p = reinterpret_cast<int64_t>(t->data_ptr());
      if (inpl && p % 16 == 0) {
        plan_ptr_[j] = p;
        continue;
      }
      if (!t->is_contiguous() || t->scalar_type() != st.scalar_type() ||
          t->numel() > st.numel())
        return 0;
      copy_.push_back((int)j);
      if (inpl) plan_ptr_[j] = reinterpret_cast<int64_t>(st.data_ptr());
    }
    // A static input that moved (a backward's are the forward's outputs,
    // laid out per shape) is read at its new address, as `_rebind_static`
    // does: the host patcher repoints every node that reads it. One that is
    // no longer 16-byte aligned cannot be read in place at all.
    moved_.clear();
    for (size_t k = 0; k < static_pos_.size(); ++k) {
      const at::Tensor* t = tensor_at(static_pos_[k]);
      if (t == nullptr) return 1;
      int64_t p = reinterpret_cast<int64_t>(t->data_ptr());
      if (p != static_ptr_[k]) {
        if (p % 16) return 2;
        moved_.push_back((int)k);
      }
    }
    return -1;
  }

  // Write-backs come after the launch; they must not be able to fail.
  bool writes_back_ok() const {
    auto ok = [&](int j) {
      const at::Tensor* t = tensor_at(j);
      return t != nullptr && j < (int)stores_.size() && stores_[j].defined() &&
             t->is_contiguous() && t->scalar_type() == stores_[j].scalar_type();
    };
    for (int j : mut_copy_)
      if (!ok(j)) return false;
    for (int j : mut_patch_)
      if (!ok(j)) return false;
    return true;
  }

  // The address input `j` is read at on this call (after `plan_inputs`):
  // its own, or the store it is copied into.
  int64_t input_addr(int j) const {
    if (j < (int)plan_ptr_.size() && plan_ptr_[j]) return plan_ptr_[j];
    if (j < (int)stores_.size() && stores_[j].defined())
      return reinterpret_cast<int64_t>(stores_[j].data_ptr());
    return reinterpret_cast<int64_t>(tensor_at(j)->data_ptr());
  }

  void apply_inputs(cudaStream_t stream) {
    for (int j : copy_) {
      const at::Tensor& t = *tensor_at(j);
      size_t bytes = t.numel() * t.element_size();
      if (bytes)
        cudaMemcpyAsync(stores_[j].data_ptr(), t.data_ptr(), bytes,
                        cudaMemcpyDeviceToDevice, stream);
    }
    for (size_t j = 0; j < plan_ptr_.size(); ++j)
      if (plan_ptr_[j]) in_ptrs_[j] = plan_ptr_[j];
    for (int k : moved_) {
      int pos = static_pos_[k];
      int64_t p = reinterpret_cast<int64_t>(tensor_at(pos)->data_ptr());
      static_ptr_[k] = p;
      in_ptrs_[pos] = p;
    }
  }

  int set_inline(Entry& e) {
    if (e.own_batch) {
      const InlineBatch& b = e.batch;
      return inline_fn_(e.exec, (int)b.nodes.size(), b.nodes.data(), b.funcs.data(),
                        b.dims.data(), b.blobs.data(), b.lens.data(), b.nparam.data(),
                        b.poffs.data(), b.npatch.data(), b.pw.data(), b.pv.data());
    }
    const auto& a = e.inl;
    return inline_fn_(
        e.exec, e.n_inline, reinterpret_cast<const void*>(a[0]),
        reinterpret_cast<const void*>(a[1]), reinterpret_cast<const unsigned*>(a[2]),
        reinterpret_cast<const char* const*>(a[3]), reinterpret_cast<const int*>(a[4]),
        reinterpret_cast<const int*>(a[5]), reinterpret_cast<const int*>(a[6]),
        reinterpret_cast<const int*>(a[7]), reinterpret_cast<const int*>(a[8]),
        reinterpret_cast<const unsigned long long*>(a[9]));
  }

  py::object serve(Entry& e, const at::Tensor& arena) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_).stream();
    apply_inputs(stream);
    if (e.has_inline) {
      py::object state = e.ex.attr("inline_state");
      if (!(state.is(e.token) || state.equal(e.token))) {
        int rc = set_inline(e);
        if (rc != 0) {
          e.ex.attr("inline_state") = py::none();
          return py::make_tuple("inline", rc);
        }
        e.ex.attr("inline_state") = e.token;
        e.ex.attr("inline_applied") = py::dict(e.applied);
      }
    }
    PyObject* d = PyObject_GetAttrString(e.ex.ptr(), "ptr_dirty");
    if (d == nullptr) throw py::error_already_set();
    int dirty = PyObject_IsTrue(d) == 1;
    Py_DECREF(d);
    const int64_t* syms = e.syms ? e.syms : e.syms_own.data();
    int rc = e.step(e.host, syms, dirty, static_cast<char*>(arena.data_ptr()), e.ext,
                    e.child, reinterpret_cast<const char* const*>(in_ptrs_), nullptr,
                    stream, 1);
    if (rc != 0) return py::make_tuple("step", rc);
    e.ex.attr("ptr_dirty") = py::bool_(false);
    e.ex.attr("applied") = e.token;
    e.ex.attr("child_applied") = e.token;

    auto write_back = [&](int j) {
      const at::Tensor& t = *tensor_at(j);
      size_t bytes = t.numel() * t.element_size();
      if (bytes)
        cudaMemcpyAsync(t.data_ptr(), stores_[j].data_ptr(), bytes,
                        cudaMemcpyDeviceToDevice, stream);
    };
    for (int j : mut_copy_) write_back(j);
    for (int j : mut_patch_)
      for (int c : copy_)
        if (c == j) write_back(j);

    py::list out(e.outs.size());
    c10::Storage storage = arena.storage();
    for (size_t i = 0; i < e.outs.size(); ++i) {
      const Out& o = e.outs[i];
      switch (o.kind) {
        case kArena: {
          at::Tensor t = at::empty({0}, protos_[o.dt].options());
          t.set_(storage, o.off / t.element_size(), o.sizes, o.strides);
          out[i] = py::cast(std::move(t));
          break;
        }
        case kInput:
          out[i] = py::reinterpret_borrow<py::object>(PyList_GET_ITEM(in_, o.idx));
          break;
        case kInView: {
          const at::Tensor& b = *tensor_at(o.idx);
          at::Tensor t = at::empty({0}, b.options());
          t.set_(b.storage(), b.storage_offset() + o.off, o.sizes, o.strides);
          out[i] = py::cast(std::move(t));
          break;
        }
        case kFixed:
          out[i] = py::cast(o.fixed);
          break;
        default:
          out[i] = py::none();
      }
    }
    return py::make_tuple(out, e.ex);
  }

  // (offset, size) of every parameter of a kernel, from the driver.
  const std::vector<std::pair<int, int>>& params_of(uint64_t func) {
    auto it = params_.find(func);
    if (it != params_.end()) return it->second;
    std::vector<std::pair<int, int>> v;
    for (size_t j = 0;; ++j) {
      size_t off, sz;
      if (param_info_(reinterpret_cast<void*>(func), j, &off, &sz) != 0) break;
      v.emplace_back((int)off, (int)sz);
    }
    return params_[func] = std::move(v);
  }

  // A shape no call has been served at: made here, then kept.
  py::object serve_new(const at::Tensor& arena) {
    const Program& P = prog_;
    Entry e;
    e.syms_own.assign(std::max<size_t>(P.sym_pos.size(), 1), 0);
    for (size_t k = 0; k < P.sym_pos.size(); ++k) {
      int p = P.sym_pos[k];
      if (p >= n_in_) return miss(3);
      PyObject* o = PyList_GET_ITEM(in_, p);
      if (!PyLong_CheckExact(o)) return miss(3);
      e.syms_own[k] = PyLong_AsLongLong(o);
    }
    slot_off_.assign(P.n_slots + 1, 0);
    geom_.assign(std::max(P.n_geom, 1), 0);
    P.geom(e.syms_own.data(), slot_off_.data(), geom_.data());
    e.total = slot_off_[P.n_slots];
    if (arena.numel() * arena.element_size() < e.total) return miss(5);
    const int64_t base = reinterpret_cast<int64_t>(arena.data_ptr());

    // Every site's launches, from its library.
    described_.clear();
    per_site_.assign(P.sites.size(), 0);
    for (size_t s = 0; s < P.sites.size(); ++s) {
      const Site& site = P.sites[s];
      ops_.resize(site.ops.size());
      for (size_t q = 0; q < site.ops.size(); ++q) {
        const Operand& op = site.ops[q];
        dg_operand& d = ops_[q];
        std::memset(&d, 0, sizeof(d));
        const int64_t* g = op.g_at >= 0 ? &geom_[op.g_at] : nullptr;
        if (op.kind == 0) {
          d.ptr = base + slot_off_[op.slot] + g[2 * op.nd] * op.itemsize;
          d.dtype = (int32_t)protos_[op.dt].scalar_type();
          d.ndim = op.nd;
          for (int x = 0; x < op.nd; ++x) {
            d.sizes[x] = g[x];
            d.strides[x] = g[op.nd + x];
          }
        } else {
          const at::Tensor* t = tensor_at(op.pos);
          if (t == nullptr) return miss(7);
          d.dtype = (int32_t)t->scalar_type();
          int64_t a = input_addr(op.pos);
          if (g != nullptr) {
            d.ptr = a + g[2 * op.nd] * t->element_size();
            d.ndim = op.nd;
            for (int x = 0; x < op.nd; ++x) {
              d.sizes[x] = g[x];
              d.strides[x] = g[op.nd + x];
            }
          } else {
            d.ptr = a;
            d.ndim = (int32_t)t->dim();
            if (d.ndim > 8) return miss(7);
            for (int x = 0; x < d.ndim; ++x) {
              d.sizes[x] = t->size(x);
              d.strides[x] = t->stride(x);
            }
          }
        }
      }
      int nl = site.describe(ops_.data(), (int)ops_.size(), site.statics.c_str(),
                             launch_buf_, kMaxLaunches, arg_buf_, sizeof(arg_buf_),
                             arg_sizes_, kMaxArgs, err_, sizeof(err_));
      if (nl < 0) return miss(nl == -1 ? 8 : 9);
      per_site_[s] = nl;
      size_t at = 0, ai = 0;
      for (int l = 0; l < nl; ++l) {
        const dg_launch& L = launch_buf_[l];
        if (L.pdl) return miss(10);
        const auto& info = params_of(L.func);
        if (info.size() != L.nargs) return miss(10);
        Described D;
        D.func = L.func;
        unsigned dims[7] = {L.grid[0], L.grid[1], L.grid[2], L.block[0],
                            L.block[1], L.block[2], L.smem};
        std::memcpy(D.dims, dims, sizeof(dims));
        D.cluster = (int)L.cluster;
        int total = 1;
        for (const auto& pi : info) total = std::max(total, pi.first + pi.second);
        D.blob.assign(total, '\0');
        for (uint32_t q = 0; q < L.nargs; ++q, ++ai) {
          if ((int)arg_sizes_[ai] != info[q].second) return miss(10);
          std::memcpy(&D.blob[info[q].first], arg_buf_ + at, arg_sizes_[ai]);
          at += arg_sizes_[ai];
          D.offs.push_back(info[q].first);
        }
        described_.push_back(std::move(D));
      }
    }

    // The graph captured with this launch structure.
    ExecInfo* X = nullptr;
    for (auto& x : execs_) {
      bool ok = x.nodes.size() == P.sites.size();
      size_t at = 0;
      for (size_t s = 0; ok && s < P.sites.size(); ++s) {
        if ((int)x.nodes[s].size() != per_site_[s]) ok = false;
        for (int l = 0; ok && l < per_site_[s]; ++l)
          if (x.clusters[s][l] != described_[at + l].cluster) ok = false;
        at += per_site_[s];
      }
      if (ok) {
        X = &x;
        break;
      }
    }
    if (X == nullptr) return miss(11);

    e.ex = X->ex;
    e.token = py::int_(++token_);
    e.applied = py::dict();
    e.step = X->step;
    e.host = X->host;
    e.ext = X->ext;
    e.child = X->child;
    e.has_inline = true;
    e.own_batch = true;
    e.exec = X->exec;
    InlineBatch& b = e.batch;
    size_t at = 0;
    for (size_t s = 0; s < P.sites.size(); ++s)
      for (int l = 0; l < per_site_[s]; ++l, ++at) {
        Described& D = described_[at];
        b.nodes.push_back(X->nodes[s][l]);
        b.funcs.push_back(D.func);
        b.dims.insert(b.dims.end(), D.dims, D.dims + 7);
        b.lens.push_back((int)D.blob.size());
        b.nparam.push_back((int)D.offs.size());
        b.poffs.insert(b.poffs.end(), D.offs.begin(), D.offs.end());
        b.blob_store.push_back(std::move(D.blob));
      }
    for (size_t i = 0; i < P.o_kind.size(); ++i) {
      Out o;
      o.kind = P.o_kind[i];
      const int64_t* g = P.o_gat[i] >= 0 ? &geom_[P.o_gat[i]] : nullptr;
      int nd = P.o_nd[i];
      if (o.kind == kArena) {
        // geometry: element offset, sizes, strides
        o.off = slot_off_[P.o_slot[i]] + g[0] * P.o_item[i];
        o.dt = P.o_dt[i];
        o.sizes.assign(g + 1, g + 1 + nd);
        o.strides.assign(g + 1 + nd, g + 1 + 2 * nd);
      } else if (o.kind == kInView) {
        // geometry: sizes, strides, element offset
        o.idx = P.o_slot[i];
        o.sizes.assign(g, g + nd);
        o.strides.assign(g + nd, g + 2 * nd);
        o.off = g[2 * nd];
      } else if (o.kind == kInput) {
        o.idx = P.o_slot[i];
      } else if (o.kind == kFixed) {
        o.fixed = P.o_fixed[i];
      }
      e.outs.push_back(std::move(o));
    }
    ++hits_;
    ++made_;
    Entry& kept = insert(key_, std::move(e));
    return serve(kept, arena);
  }

  Entry& insert(std::vector<int64_t> key, Entry e) {
    if (entries_.size() >= 8192) entries_.clear();
    Entry& slot = entries_[std::move(key)];
    slot = std::move(e);
    if (slot.own_batch) slot.batch.seal();
    return slot;
  }

  py::object miss(int why) {
    ++misses_;
    if ((int)why_.size() <= why) why_.resize(why + 1, 0);
    ++why_[why];
    return py::none();
  }

  static constexpr int kMaxLaunches = 16;
  static constexpr int kMaxArgs = 256;

  std::vector<int> sym_pos_, ext_read_;
  std::vector<char> inplace_;
  int64_t* in_ptrs_;
  std::vector<int> mut_copy_, mut_patch_;
  InlineFn inline_fn_;
  int64_t device_;
  std::vector<at::Tensor> stores_;
  std::vector<int> static_pos_, static_of_;
  std::vector<int64_t> static_ptr_;
  std::vector<at::Tensor> protos_;
  std::unordered_map<std::vector<int64_t>, Entry, KeyHash> entries_;
  int64_t hits_ = 0, misses_ = 0, made_ = 0, token_ = 0;
  std::vector<int64_t> why_;

  bool has_prog_ = false;
  Program prog_;
  std::vector<ExecInfo> execs_;
  ParamInfoFn param_info_ = nullptr;
  std::unordered_map<uint64_t, std::vector<std::pair<int, int>>> params_;

  // per call
  PyObject* in_ = nullptr;
  Py_ssize_t n_in_ = 0;
  std::vector<int64_t> key_, plan_ptr_, slot_off_, geom_;
  std::vector<int> copy_, moved_, per_site_;
  std::vector<dg_operand> ops_;
  std::vector<Described> described_;
  dg_launch launch_buf_[kMaxLaunches];
  char arg_buf_[16384];
  uint32_t arg_sizes_[kMaxArgs];
  char err_[512];
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Region>(m, "Region")
      .def(py::init<std::vector<int>, std::vector<int>, std::vector<int>, uintptr_t,
                    std::vector<int>, std::vector<int>, uintptr_t, int64_t>())
      .def("set_store", &Region::set_store)
      .def("set_static", &Region::set_static)
      .def("add_proto", &Region::add_proto)
      .def("put", &Region::put)
      .def("set_program", &Region::set_program)
      .def("add_exec", &Region::add_exec)
      .def("clear_execs", &Region::clear_execs)
      .def("n_execs", &Region::n_execs)
      .def("has_program", &Region::has_program)
      .def("call", &Region::call)
      .def("clear", &Region::clear)
      .def("size", &Region::size)
      .def("stats", &Region::stats);
}
