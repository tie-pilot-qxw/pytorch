"""Give DeepGEMM's describe a C entry point (after patch_dgd.py): DynaGraph's C++ runtime calls it
directly, with raw operand descriptors, instead of going through Python per call site.

The ABI is DynaGraph's launch-describe ABI v1 (torch/utils/_capture_launch.py, `c_describe`):

  struct dg_operand { uint64_t ptr; int32_t dtype; int32_t ndim; int64_t sizes[8]; int64_t strides[8]; };
  struct dg_launch  { uint64_t func; uint32_t grid[3], block[3], smem, cluster, pdl, nargs; };
  int describe(const dg_operand* ops, int nops, const char* statics,
               dg_launch* out, int max_launches, char* args, int64_t args_cap,
               uint32_t* arg_sizes, int sizes_cap, char* err, int err_cap);

returning the number of launches (their argument bytes back to back in `args`, one size per argument
in `arg_sizes`), -1 when the library raised (message in `err`), -2 when a buffer is too small.
"""
import os
root = os.path.join(os.environ.get("DG_DEPS", "/workspace/_deps"), "deepgemm-src/csrc")
p = f"{root}/python_api.cpp"
s = open(p).read()
assert "dg_describe_bf16_gemm_nt" not in s
entry = r'''
#include <ATen/cuda/CUDAContext.h>
#include <cstring>

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

static torch::Tensor dg_tensor(const dg_operand& o) {
    return torch::from_blob(reinterpret_cast<void*>(o.ptr), at::IntArrayRef(o.sizes, o.ndim),
                            at::IntArrayRef(o.strides, o.ndim),
                            torch::TensorOptions()
                                .dtype(static_cast<c10::ScalarType>(o.dtype))
                                .device(torch::kCUDA, c10::cuda::current_device()));
}

// deep_gemm.bf16_gemm_nt(a, b, d, compiled_dims=statics), described, not launched.
extern "C" __attribute__((visibility("default"))) int dg_describe_bf16_gemm_nt(
        const dg_operand* ops, int nops, const char* statics,
        dg_launch* out, int max_launches, char* args, int64_t args_cap,
        uint32_t* arg_sizes, int sizes_cap, char* err, int err_cap) {
    auto& sink = deep_gemm::describe_sink();
    try {
        if (nops != 3 or sink.on)
            throw std::runtime_error("dg_describe_bf16_gemm_nt: 3 operands, not inside another describe");
        sink.launches.clear();
        sink.on = true;
        deep_gemm::gemm::bf16_gemm_nt(dg_tensor(ops[0]), dg_tensor(ops[1]), dg_tensor(ops[2]),
                                      std::nullopt, statics ? statics : "nk");
        sink.on = false;
    } catch (const std::exception& e) {
        sink.on = false;
        sink.launches.clear();
        if (err and err_cap > 0) {
            std::strncpy(err, e.what(), err_cap - 1);
            err[err_cap - 1] = 0;
        }
        return -1;
    }
    const auto& ls = sink.launches;
    if ((int)ls.size() > max_launches) return -2;
    int64_t at = 0;
    int ai = 0;
    for (size_t i = 0; i < ls.size(); ++ i) {
        const auto& d = ls[i];
        out[i] = dg_launch{d.func, {d.grid[0], d.grid[1], d.grid[2]}, {d.block[0], d.block[1], d.block[2]},
                           d.smem, d.cluster, d.pdl ? 1u : 0u, static_cast<uint32_t>(d.args.size())};
        for (const auto& a: d.args) {
            if (ai >= sizes_cap or at + (int64_t)a.size() > args_cap) return -2;
            std::memcpy(args + at, a.data(), a.size());
            at += a.size();
            arg_sizes[ai ++] = static_cast<uint32_t>(a.size());
        }
    }
    int n = static_cast<int>(ls.size());
    sink.launches.clear();
    return n;
}
'''
old = '#ifndef TORCH_EXTENSION_NAME'
assert s.count(old) == 1
s = s.replace(old, entry + "\n" + old)
open(p, "w").write(s)
print("patched", p)
