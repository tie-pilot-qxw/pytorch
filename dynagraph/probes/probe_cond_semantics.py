#!/usr/bin/env python3
r"""Semantics of device-side update / first launch / recapture with conditional nodes + child nodes -- a minimal repro that bypasses Inductor.

The four things the runner works around (docs/notes/EXTERN.md section 8) each get a minimal version here:
  P1  pointer persistence: on a device-updatable worker node, the planner first changes pointer+scalar, then next time only the scalar --
      is the pointer still there? Tested once with and once without a SWITCH node in the same graph.
  P2  first launch: on a new exec with a child node, do the parameters the planner changes take effect in the first launch?
      Tested once with and once without an external event node in the child.
  P3  recapture: from the same set of collected small graphs, first capture graph A with a SWITCH and launch it, then capture graph B with one more body
      (old graph destroyed / kept, both variants), launch B; then one more graph C. At which step does it start going wrong or crash?
The criterion at every step is that the values of y are bitwise equal to the expected ones.
"""
from __future__ import annotations

import ctypes
import sys

import torch
from cuda.bindings import runtime as cr
from torch.cuda._utils import _check_cuda_bindings as ck
from torch._inductor import dynagraph as dg

SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>
// y[i] = x[i] * k + b.   params: x@0 (8) y@8 (8) n@16 (4) k@20 (4) b@24 (4)
extern "C" __global__ void worker(const float* x, float* y, int n, float k, float b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * k + b;
}
// ctx: [0] mode bits: 1 patch x pointer, 2 patch b, 4 set conditional
//      [1] conditional handle (0: none)  [2] conditional value
//      [3] new x pointer  [4] new b (float bits)  [5] worker device node handle
extern "C" __global__ void planner(const long long* ctx) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  long long mode = ctx[0];
  cudaGraphDeviceNode_t node = (cudaGraphDeviceNode_t)ctx[5];
  if (mode & 1) { const float* p = (const float*)ctx[3]; cudaGraphKernelNodeSetParam(node, 0, &p, 8); }
  if (mode & 2) { float b = __int_as_float((int)ctx[4]); cudaGraphKernelNodeSetParam(node, 24, &b, 4); }
  if ((mode & 4) && ctx[1] != 0)
    cudaGraphSetConditional((cudaGraphConditionalHandle)ctx[1], (unsigned)ctx[2]);
}
"""
N = 1024


def fbits(v: float) -> int:
    return ctypes.c_int.from_buffer(ctypes.c_float(v)).value


def launch_worker(f, x, y, k, b, stream):
    holders = [ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), ctypes.c_int(N),
               ctypes.c_float(k), ctypes.c_float(b)]
    arr = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
    rc = dg._cuda().cuLaunchKernel(ctypes.c_void_p(f), (N + 127) // 128, 1, 1, 128, 1, 1, 0,
                                   ctypes.c_void_p(stream), arr, None)
    assert rc == 0, rc


def small_graph(fn, external_events=False):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g, stream=s):
        if external_events:
            ev = ck(cr.cudaEventCreate())
            ck(cr.cudaStreamWaitEvent(s.cuda_stream, ev, cr.cudaEventWaitExternal))
        fn()
        if external_events:
            ck(cr.cudaEventRecordWithFlags(ev, s.cuda_stream, cr.cudaEventRecordExternal))
    torch.cuda.synchronize()
    return g


def worker_devnode(raw_graph, f_worker_name=b"worker"):
    """Make the captured worker node device-updatable after the fact and return its device handle."""
    from cuda.bindings import driver as cd
    n = ck(cr.cudaGraphGetNodes(raw_graph))[1]
    for nd in ck(cr.cudaGraphGetNodes(raw_graph, n))[0]:
        if ck(cr.cudaGraphNodeGetType(nd)) != cr.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            continue
        pp = ck(cd.cuGraphKernelNodeGetParams(nd))
        if ck(cd.cuFuncGetName(pp.func)) != f_worker_name:
            continue
        val = cr.cudaKernelNodeAttrValue()   # the kernel-node flavour of cudaLaunchAttributeValue
        val.deviceUpdatableKernelNode.deviceUpdatable = 1
        ck(cr.cudaGraphKernelNodeSetAttribute(nd, cr.cudaLaunchAttributeID.cudaLaunchAttributeDeviceUpdatableKernelNode, val))
        got = ck(cr.cudaGraphKernelNodeGetAttribute(nd, cr.cudaLaunchAttributeID.cudaLaunchAttributeDeviceUpdatableKernelNode))
        return int(got.deviceUpdatableKernelNode.devNode)
    raise RuntimeError("worker node not found")


class Built:
    def __init__(self, f_planner, f_worker, ctx, x0, y, bodies=None, child=None, upload=False):
        self.ctx, self.y = ctx, y
        self.g = torch.cuda.CUDAGraph(keep_graph=True)
        self.handle = 0
        self.body_nodes = []
        with torch.cuda.graph(self.g):
            st = torch.cuda.current_stream().cuda_stream
            dg._launch(f_planner, [ctx.data_ptr()], 1, 32, st)
            info = ck(cr.cudaStreamGetCaptureInfo(st))
            cap, deps = info[2], list(info[3] or [])
            if bodies:
                self.handle = int(ck(cr.cudaGraphConditionalHandleCreate(cap, 0, 0)))
                prm = cr.cudaGraphNodeParams()
                prm.type = cr.cudaGraphNodeType.cudaGraphNodeTypeConditional
                prm.conditional.handle = self.handle
                prm.conditional.type = cr.cudaGraphConditionalNodeType.cudaGraphCondTypeSwitch
                prm.conditional.size = len(bodies)
                node = ck(cr.cudaGraphAddNode(cap, deps, None, len(deps), prm))
                for j, bg in enumerate(bodies):
                    self.body_nodes.append(ck(cr.cudaGraphAddChildGraphNode(
                        prm.conditional.phGraph_out[j], None, 0, bg.raw_cuda_graph())))
                ck(cr.cudaStreamUpdateCaptureDependencies(st, [node], None, 1,
                    cr.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies))
            if child is not None:
                node = ck(cr.cudaGraphAddChildGraphNode(cap, deps, len(deps), child.raw_cuda_graph()))
                ck(cr.cudaStreamUpdateCaptureDependencies(st, [node], None, 1,
                    cr.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies))
            launch_worker(f_worker, x0, y, 3.0, 0.0, st)
        self.devnode = worker_devnode(self.g.raw_cuda_graph())
        self.g.instantiate()
        if upload:
            # Upload the exec before its first launch, so the launch does not
            # do it and overwrite the device-side updates the planner makes.
            ck(cr.cudaGraphUpload(self.g.raw_cuda_graph_exec(), torch.cuda.current_stream().cuda_stream))
            torch.cuda.synchronize()
        ctx[5] = self.devnode
        ctx[1] = self.handle

    def run(self, mode, xptr=None, b=None, cond=None):
        self.ctx[0] = mode
        if xptr is not None: self.ctx[3] = xptr
        if b is not None: self.ctx[4] = fbits(b)
        if cond is not None: self.ctx[2] = cond
        self.g.replay(); torch.cuda.synchronize()


def check(tag, y, want, bad, expect_fail=False):
    """Rows without upload document the effect and are expected to fail; only
    the rows the runner relies on (upload before first launch) count."""
    ok = torch.equal(y, want)
    mark = "OK" if ok else "FAIL y[0]=%g want %g" % (y[0].item(), want[0].item())
    if expect_fail:
        mark += "  (expected: no upload)" if not ok else "  (unexpected pass)"
    print(f"    {tag:<58} {mark}")
    return bad + ((not ok) if not expect_fail else 0)


def main() -> int:
    xA = torch.arange(N, device="cuda", dtype=torch.float32)
    xB = xA * 10 + 1
    y = torch.zeros(N, device="cuda")
    fns = dg._compile_module(SRC, ["planner", "worker"])
    if fns is None:
        print("FAIL compile failed"); return 1
    f_planner, f_worker = fns
    t = torch.zeros(N, device="cuda")
    g0 = small_graph(lambda: torch.mul(xA, 2, out=t))
    g1 = small_graph(lambda: torch.mul(xA, 3, out=t))
    g2 = small_graph(lambda: torch.mul(xA, 5, out=t))
    gext = small_graph(lambda: torch.mul(xA, 7, out=t), external_events=True)
    bad = 0

    print("\n  P1 pointer persistence: change pointer+scalar first, then only the scalar (upload = cudaGraphUpload before the first launch)")
    for name, bodies, up in (("no SWITCH", None, False), ("with SWITCH", [g0, g1], False), ("with SWITCH+upload", [g0, g1], True)):
        ctx = torch.zeros(6, dtype=torch.int64, device="cuda")
        B = Built(f_planner, f_worker, ctx, xA, y, bodies=bodies, upload=up)
        xf = bodies is not None and not up
        B.run(1 | 2 | 4, xptr=xB.data_ptr(), b=1.0, cond=0)
        bad = check(f"{name}: L1 pointer -> xB, b=1", y, xB * 3 + 1, bad, xf)
        B.run(2 | 4, b=2.0, cond=1 if bodies else 0)
        bad = check(f"{name}: L2 only b=2 (switch body) -- pointer should still be xB", y, xB * 3 + 2, bad, xf)
        B.run(0)
        bad = check(f"{name}: L3 planner does nothing -- everything should persist", y, xB * 3 + 2, bad, xf)
        B.run(2 | 4, b=3.0, cond=0)
        bad = check(f"{name}: L4 only b=3 (back to body 0)", y, xB * 3 + 3, bad, xf)

    print("\n  P2 first launch of a new exec: do the planner's parameter changes take effect in the first launch?")
    for name, child, up in (("child, no external events", g0, False), ("child with external event nodes", gext, False), ("child with external events+upload", gext, True)):
        ctx = torch.zeros(6, dtype=torch.int64, device="cuda")
        B = Built(f_planner, f_worker, ctx, xA, y, child=child, upload=up)
        B.run(1 | 2, xptr=xB.data_ptr(), b=1.0)
        bad = check(f"{name}: launch #1", y, xB * 3 + 1, bad, child is gext and not up)
        B.run(1 | 2, xptr=xB.data_ptr(), b=1.0)
        bad = check(f"{name}: launch #2 (same state)", y, xB * 3 + 1, bad)

    print("\n  P3 recapture (all with upload): A(2 body) -> B(3 body) -> C(3 body), old graph destroyed / kept")
    for keep in (True, False):
        holds = []
        try:
            ctxA = torch.zeros(6, dtype=torch.int64, device="cuda")
            A = Built(f_planner, f_worker, ctxA, xA, y, bodies=[g0, g1], upload=True)
            A.run(1 | 2 | 4, xptr=xB.data_ptr(), b=1.0, cond=1)
            bad = check(f"keep={keep}: A selects body1", y, xB * 3 + 1, bad)
            if keep: holds.append(A)
            del A
            ctxB = torch.zeros(6, dtype=torch.int64, device="cuda")
            Bg = Built(f_planner, f_worker, ctxB, xA, y, bodies=[g0, g1, g2], upload=True)
            Bg.run(1 | 2 | 4, xptr=xB.data_ptr(), b=2.0, cond=2)
            bad = check(f"keep={keep}: B selects body2 (first recapture)", y, xB * 3 + 2, bad)
            Bg.run(2 | 4, b=3.0, cond=0)
            bad = check(f"keep={keep}: B switches to body0, only b changes", y, xB * 3 + 3, bad)
            if keep: holds.append(Bg)
            del Bg
            ctxC = torch.zeros(6, dtype=torch.int64, device="cuda")
            C = Built(f_planner, f_worker, ctxC, xA, y, bodies=[g0, g1, g2], upload=True)
            C.run(1 | 2 | 4, xptr=xB.data_ptr(), b=4.0, cond=1)
            bad = check(f"keep={keep}: C selects body1 (second recapture)", y, xB * 3 + 4, bad)
            C.run(2 | 4, b=5.0, cond=2)
            bad = check(f"keep={keep}: C switches to body2, only b changes", y, xB * 3 + 5, bad)
            holds.append(C)
        except Exception as e:  # noqa: BLE001
            print(f"    keep={keep}: crashed {type(e).__name__}: {str(e)[:80]}"); bad += 1
            return 1
    print("\n  " + ("all passed" if not bad else f"{bad} failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
