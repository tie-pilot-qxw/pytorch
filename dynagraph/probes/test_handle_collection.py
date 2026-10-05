#!/usr/bin/env python3
"""
End-to-end check: can device node handles be obtained in launch order during capture?

This is the first gate for hooking DynaGraph into PyTorch. Already shown:
  - the mechanism itself works (microbench/dynagraph_proto.cu, elementwise cross-check passes)
  - the planner can be generated mechanically (probes/planner_codegen.py, compiles and links)
This checks the third piece: **on real Inductor compile output**, can every kernel node be
marked device-updatable at capture time and its handle fetched back in order.

Why "in launch order" is key: the order cuGraphGetNodes returns is unspecified,
and the same Triton kernel launched twice with different numel yields two nodes with the same func pointer --
after the fact there is no telling which is which. A handle only has an identity if it comes back from the launch call itself.

No timing, runs on a shared card.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")


def main():
    import torch
    import torch._inductor.config as ic

    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1

    L = getattr(torch._C, "_StaticCudaLauncher", None)
    for name in ("_begin_device_node_collection", "_end_device_node_collection"):
        if not hasattr(L, name):
            print(f"  torch._C._StaticCudaLauncher is missing {name} -- torch_python not relinked?")
            return 1
    print("  new interface is in place")
    print(f"  use_static_cuda_launcher = {ic.use_static_cuda_launcher}")
    if not ic.use_static_cuda_launcher:
        print("  static launcher is off, the change will not take effect"); return 1

    ic.force_disable_caches = True
    # Move mm/addmm from cuBLAS to a Triton template. extern_kernels are not Triton kernels and
    # do not go through the static launcher, so those nodes never get a handle -- routing is a prerequisite for DynaGraph,
    # not an option. Set DYNAGRAPH_ROUTE_GEMM=0 to reproduce the handle-count mismatch without routing.
    if os.environ.get("DYNAGRAPH_ROUTE_GEMM", "1") == "1":
        ic.max_autotune_gemm = True
        ic.max_autotune_gemm_backends = "TRITON"
        print("  GEMM routed to Triton templates")
    else:
        print("  GEMM not routed (handle count expected to mismatch)")

    class M(torch.nn.Module):
        """Deliberately written so it does not fuse into a single kernel.

        The reduction cuts pointwise fusion, and softmax and cumsum each form their own segment,
        so this really tests "multiple handles, in order, all distinct",
        not just the trivial single-kernel case."""

        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(256, 256)
            self.l2 = torch.nn.Linear(256, 256)

        def forward(self, x):
            h = torch.relu(self.l1(x))
            h = h - h.mean(dim=-1, keepdim=True)      # reduction, breaks fusion
            h = torch.softmax(h, dim=-1)              # another segment
            h = torch.relu(self.l2(h))
            h = h + h.cumsum(dim=-1)                  # scan, breaks it again
            return h.sum(dim=-1)                      # last reduction

    m = M().cuda().eval()
    x = torch.randn(512, 256, device="cuda")
    f = torch.compile(m, dynamic=True)

    # Compile + warm up first to flush out compile-time work, so none of it leaks into the capture
    with torch.no_grad():
        for _ in range(3):
            f(x)
    torch.cuda.synchronize()

    # Count how many kernels are launched without collection, as a control
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        f(x)
        torch.cuda.synchronize()
    n_kernels = sum(
        r.count for r in prof.key_averages()
        if (getattr(r, "device_time_total", 0) or 0) > 0
        and r.key not in ("cudaLaunchKernel", "cudaDeviceSynchronize",
                          "cudaStreamSynchronize"))
    print(f"  device kernels per forward ~ {n_kernels}")

    # The real check: collect during capture
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    handles = None
    with torch.cuda.stream(s):
        with torch.no_grad():
            f(x)                       # run once more before capture, through the warmup path
    torch.cuda.current_stream().wait_stream(s)

    try:
        L._begin_device_node_collection()
        with torch.cuda.graph(g):
            with torch.no_grad():
                f(x)
        handles = L._end_device_node_collection()
    except Exception as e:
        try:
            L._end_device_node_collection()
        except Exception:
            pass
        print(f"  capture failed: {type(e).__name__}: {str(e)[:200]}")
        return 1

    print(f"  collected {len(handles)} device node handles")
    if handles:
        print(f"    first few: {[hex(h) for h in handles[:4]]}")
    uniq = len(set(handles))
    print(f"    all distinct: {uniq == len(handles)}  (unique={uniq})")

    # The real criterion: the handle count must match the number of kernels actually launched.
    # A mismatch means some kernel went through Triton's own launcher (can_statically_launch bypasses
    # some), and those nodes get no handle -- so the planner cannot change them either.
    matched = len(handles) == n_kernels
    print(f"    handles vs kernels: {len(handles)} vs {n_kernels}  "
          f"{'match' if matched else '**mismatch, some kernel did not go through the static launcher**'}")

    ok = len(handles) > 1 and uniq == len(handles) and matched
    if ok:
        print("\nHandle collection works end to end: multiple kernels, distinct handles, counts match.")
    else:
        print("\nMismatch. The most likely cause is extern_kernels (cuBLAS addmm/mm) -- "
              "they are not Triton kernels, do not go through the static launcher, and get no handle."
              "\nRoute GEMM to Triton/CUTLASS with max_autotune_gemm_backends and try again.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
