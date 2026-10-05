#!/usr/bin/env python3
r"""Host-path overlap: while the exec is running on the GPU (a ~5 ms spin kernel), the host changes its parameters with
`cuGraphExecKernelNodeSetParams` -- (1) the call itself does not wait for the GPU (returns in a few us), (2) the running launch
is unaffected, (3) the next launch uses the new parameters. The CUDA docs say "already enqueued or running launches are not
affected"; this measures it, since the host-path overlap is built on it. Does not go through Inductor."""
import ctypes, sys, time
import torch
from cuda.bindings import driver as cd
from torch._inductor import dynagraph as dg

SRC = r"""
extern "C" __global__ void spin(long long* out, long long v, long long idx, long long iters) {
  long long acc = v;
  for (long long i = 0; i < iters; ++i) acc = acc * 6364136223846793005LL + i;
  if (threadIdx.x == 0) out[idx] = v + (acc == 12345 ? 1 : 0);
}
"""


def ck(res):
    err, *rest = res
    assert err == cd.CUresult.CUDA_SUCCESS, err
    return rest[0] if len(rest) == 1 else rest


def main() -> int:
    torch.empty(1, device="cuda")
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}" + ("a" if (major, minor) >= (9, 0) else "")
    cubin = dg._nvrtc_cubin(SRC, arch)
    mod = ctypes.c_void_p()
    assert dg._cuda().cuModuleLoadData(ctypes.byref(mod), cubin) == 0
    fn = ctypes.c_void_p()
    assert dg._cuda().cuModuleGetFunction(ctypes.byref(fn), mod, b"spin") == 0
    out = torch.zeros(4, dtype=torch.int64, device="cuda")
    iters = 5_000_000
    def launch(v, idx):
        holders = [ctypes.c_void_p(out.data_ptr()), ctypes.c_int64(v), ctypes.c_int64(idx), ctypes.c_int64(iters)]
        arr = (ctypes.c_void_p * 4)(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        rc = dg._cuda().cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), arr, None)
        assert rc == 0, rc
    # calibrate
    t0 = time.perf_counter(); launch(0, 3); torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1e3
    print(f"  one spin kernel {ms:.1f} ms")
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        launch(1, 0)
    g.instantiate()
    raw, ex = g.raw_cuda_graph(), g.raw_cuda_graph_exec()
    n = ck(cd.cuGraphGetNodes(raw, 0))[1] if False else None
    nodes, cnt = ck(cd.cuGraphGetNodes(raw, 8))
    node = nodes[0]
    p = ck(cd.cuGraphKernelNodeGetParams(node))
    # our own packed buffer, patched by byte offset (as the generated patcher does)
    buf = (ctypes.c_char * 32)()
    ctypes.memmove(buf, ctypes.c_void_p(out.data_ptr()).__class__ and ctypes.byref(ctypes.c_void_p(out.data_ptr())), 8)
    ctypes.memmove(ctypes.addressof(buf) + 8, ctypes.byref(ctypes.c_int64(1)), 8)
    ctypes.memmove(ctypes.addressof(buf) + 16, ctypes.byref(ctypes.c_int64(0)), 8)
    ctypes.memmove(ctypes.addressof(buf) + 24, ctypes.byref(ctypes.c_int64(iters)), 8)
    size = ctypes.c_size_t(32)
    END = 0  # CU_LAUNCH_PARAM_END
    extra = (ctypes.c_void_p * 5)(1, ctypes.addressof(buf), 2, ctypes.addressof(size), END)  # BUFFER_POINTER=1, BUFFER_SIZE=2
    p.kernelParams = 0
    p.extra = ctypes.addressof(extra)

    def set_params(v, idx):
        ctypes.memmove(ctypes.addressof(buf) + 8, ctypes.byref(ctypes.c_int64(v)), 8)
        ctypes.memmove(ctypes.addressof(buf) + 16, ctypes.byref(ctypes.c_int64(idx)), 8)
        t0 = time.perf_counter()
        ck(cd.cuGraphExecKernelNodeSetParams(ex, node, p))
        return (time.perf_counter() - t0) * 1e6

    out.zero_(); torch.cuda.synchronize()
    set_params(1, 0)
    g.replay()                       # launch A: v=1 -> out[0], runs ~ms
    us = set_params(2, 1)            # while A runs: patch for the next launch
    t1 = time.perf_counter(); g.replay(); host_launch_us = (time.perf_counter() - t1) * 1e6   # launch B: v=2 -> out[1]
    torch.cuda.synchronize()
    got = out.tolist()
    ok = got[0] == 1 and got[1] == 2 and us < 1000
    print(f"  patching params while A runs: SetParams host time {us:.1f} us, second launch host time {host_launch_us:.1f} us")
    print(f"  result out = {got[:2]}: first launch used old params {'OK' if got[0] == 1 else 'FAIL'}, second used new params {'OK' if got[1] == 2 else 'FAIL'}, patch did not wait for GPU {'OK' if us < 1000 else 'FAIL'}")
    print("\n  " + ("all passed: host-side graph patching overlaps the running launch, and each launch uses its own params" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
