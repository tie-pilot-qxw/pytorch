"""Add a describe sink to DeepGEMM: between describe_begin() and describe_end() every kernel launch is
recorded (function, grid, block, smem, cluster, PDL, argument bytes) and not issued."""
import os
import re
root = os.path.join(os.environ.get("DG_DEPS", "/workspace/_deps"), "deepgemm-src/csrc")
p = f"{root}/jit/handle.hpp"
s = open(p).read()
assert "DescribeSink" not in s
sink = '''
// Describe mode: a caller that wants the launches an API call would make (to put them into a CUDA
// graph of its own and rewrite them per shape) turns this on; launch_kernel then records each launch
// instead of issuing it. Everything else the call does on the host (heuristics, JIT, TMA descriptor
// encoding) runs as usual, so what is recorded is exactly what would have been launched.
struct DescribedLaunch {
    uint64_t func;
    unsigned grid[3], block[3];
    unsigned smem;
    unsigned cluster;
    bool pdl;
    std::vector<std::string> args;
};

struct DescribeSink {
    bool on = false;
    std::vector<DescribedLaunch> launches;
};

static DescribeSink& describe_sink() {
    static thread_local DescribeSink sink;
    return sink;
}

template <typename... ActTypes>
static std::vector<std::string> describe_args(ActTypes&&... args) {
    std::vector<std::string> out;
    (out.emplace_back(reinterpret_cast<const char*>(&args), sizeof(args)), ...);
    return out;
}
'''
s = s.replace("namespace deep_gemm {\n", "namespace deep_gemm {\n" + sink, 1)
s = s.replace("#include <filesystem>\n", "#include <filesystem>\n#include <string>\n#include <vector>\n", 1)
rt_old = '''static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    void *ptr_args[] = { &args... };
    return cudaLaunchKernelExC(&config, kernel, ptr_args);
}'''
rt_new = '''static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    if (auto& sink = describe_sink(); sink.on) {
        DescribedLaunch d{reinterpret_cast<uint64_t>(kernel),
                          {config.gridDim.x, config.gridDim.y, config.gridDim.z},
                          {config.blockDim.x, config.blockDim.y, config.blockDim.z},
                          static_cast<unsigned>(config.dynamicSmemBytes), 1, false,
                          describe_args(std::forward<ActTypes>(args)...)};
        for (unsigned i = 0; i < config.numAttrs; ++ i) {
            if (config.attrs[i].id == cudaLaunchAttributeClusterDimension)
                d.cluster = config.attrs[i].val.clusterDim.x;
            if (config.attrs[i].id == cudaLaunchAttributeProgrammaticStreamSerialization)
                d.pdl = config.attrs[i].val.programmaticStreamSerializationAllowed != 0;
        }
        sink.launches.push_back(std::move(d));
        return cudaSuccess;
    }
    void *ptr_args[] = { &args... };
    return cudaLaunchKernelExC(&config, kernel, ptr_args);
}'''
drv_old = '''static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    void *ptr_args[] = { &args... };
    return lazy_cuLaunchKernelEx(&config, kernel, ptr_args, nullptr);
}'''
drv_new = '''static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    if (auto& sink = describe_sink(); sink.on) {
        DescribedLaunch d{reinterpret_cast<uint64_t>(kernel),
                          {config.gridDimX, config.gridDimY, config.gridDimZ},
                          {config.blockDimX, config.blockDimY, config.blockDimZ},
                          config.sharedMemBytes, 1, false,
                          describe_args(std::forward<ActTypes>(args)...)};
        for (unsigned i = 0; i < config.numAttrs; ++ i) {
            if (config.attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
                d.cluster = config.attrs[i].value.clusterDim.x;
            if (config.attrs[i].id == CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION)
                d.pdl = config.attrs[i].value.programmaticStreamSerializationAllowed != 0;
        }
        sink.launches.push_back(std::move(d));
        return CUDA_SUCCESS;
    }
    void *ptr_args[] = { &args... };
    return lazy_cuLaunchKernelEx(&config, kernel, ptr_args, nullptr);
}'''
assert s.count(rt_old) == 1 and s.count(drv_old) == 1
s = s.replace(rt_old, rt_new).replace(drv_old, drv_new)
open(p, "w").write(s)

p = f"{root}/apis/runtime.hpp"
s = open(p).read()
assert "describe_begin" not in s
old = "static void register_apis(pybind11::module_& m) {\n"
new = old + '''    m.def("describe_begin", [&]() {
        auto& sink = describe_sink();
        DG_HOST_ASSERT(not sink.on and "describe_begin without describe_end");
        sink.launches.clear();
        sink.on = true;
    });
    // One (func, grid, block, smem, cluster, pdl, [argument bytes]) per launch the calls since
    // describe_begin() would have made, in order.
    m.def("describe_end", [&]() {
        auto& sink = describe_sink();
        sink.on = false;
        pybind11::list out;
        for (const auto& d: sink.launches) {
            pybind11::list args;
            for (const auto& a: d.args)
                args.append(pybind11::bytes(a));
            out.append(pybind11::make_tuple(
                d.func, pybind11::make_tuple(d.grid[0], d.grid[1], d.grid[2]),
                pybind11::make_tuple(d.block[0], d.block[1], d.block[2]), d.smem, d.cluster, d.pdl, args));
        }
        sink.launches.clear();
        return out;
    });
'''
assert s.count(old) == 1
s = s.replace(old, new)
if '#include "../jit/handle.hpp"' not in s:
    s = s.replace('#include "../jit/device_runtime.hpp"', '#include "../jit/device_runtime.hpp"\n#include "../jit/handle.hpp"', 1)
open(p, "w").write(s)
print("patched")
