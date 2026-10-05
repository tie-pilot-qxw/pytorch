#!/usr/bin/env python3
r"""Get the SWITCH node working end to end from Python: extern call sites whose topology changes depend on it.

`microbench/switch_patch.cu` proved in C++ that nodes inside a SWITCH body can be patched and that the condition can be set by
a device-side kernel within the same launch. The runner is Python + cuda.bindings, so this walks the whole chain in
Python, every step written the way the runner will write it:

  1. during stream capture of the main graph: GetCaptureInfo -> ConditionalHandleCreate on the capturing graph ->
     cudaGraphAddNode(SWITCH, size=2) -> put one child-graph node in each of the two bodies
     (small graphs captured separately beforehand, simulating a "harvest") -> UpdateCaptureDependencies moved onto the SWITCH
  2. the condition value is set by a device kernel (simulating the planner) that reads ctx and calls cudaGraphSetConditional;
     it is ordered before the SWITCH in the main graph
  3. main graph keep_graph=True + explicit instantiate
  4. replay: ctx selects body 0 / body 1 / out of range (beyond size = nothing runs)
  5. use cudaGraphExecChildGraphNodeSetParams to swap the child in body 1 for another small graph with the same topology
     (simulating "same topology, different kernel/params"), then replay again

Criterion: at every step the value of y is what the small graph of the selected body computes, bitwise identical.
"""
from __future__ import annotations

import sys

import torch
from cuda.bindings import runtime as cr
from torch.cuda._utils import _check_cuda_bindings as ck
from torch._inductor import dynagraph as dg

SRC = r"""
#include <cuda_runtime.h>
extern "C" __global__ void setcond(const long long* ctx) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    cudaGraphSetConditional((cudaGraphConditionalHandle)ctx[0], (unsigned)ctx[1]);
}
"""


def small_graph(fn):
    """Capture `fn()` into its own graph, warmed up once (what a harvest does)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    return g


def main() -> int:
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    # After the first tensor: cuModuleLoad needs the context to exist.
    funcs = dg._compile_module(SRC, ["setcond"])
    if funcs is None:
        print("FAIL setcond compile/load failed (see the warning log)"); return 1
    f_setcond = funcs[0]

    y = torch.zeros_like(x)
    z = torch.zeros_like(x)
    ctx = torch.zeros(2, dtype=torch.int64, device="cuda")

    g0 = small_graph(lambda: torch.mul(x, 2, out=y))
    g1 = small_graph(lambda: torch.mul(x, 3, out=y))
    g1b = small_graph(lambda: torch.mul(x, 5, out=y))        # same topology, different params
    y.zero_()

    main = torch.cuda.CUDAGraph(keep_graph=True)
    body_nodes = []
    with torch.cuda.graph(main):
        st = torch.cuda.current_stream().cuda_stream
        dg._launch(f_setcond, [ctx.data_ptr()], 1, 32, st)
        info = ck(cr.cudaStreamGetCaptureInfo(st))
        cap_graph, deps = info[2], list(info[3] or [])
        handle = ck(cr.cudaGraphConditionalHandleCreate(cap_graph, 0, 0))
        params = cr.cudaGraphNodeParams()
        params.type = cr.cudaGraphNodeType.cudaGraphNodeTypeConditional
        params.conditional.handle = handle
        params.conditional.type = cr.cudaGraphConditionalNodeType.cudaGraphCondTypeSwitch
        params.conditional.size = 2
        node = ck(cr.cudaGraphAddNode(cap_graph, deps, None, len(deps), params))
        bodies = params.conditional.phGraph_out
        for j, g in enumerate((g0, g1)):
            body_nodes.append(ck(cr.cudaGraphAddChildGraphNode(bodies[j], None, 0, g.raw_cuda_graph())))
        ck(cr.cudaStreamUpdateCaptureDependencies(
            st, [node], None, 1,
            cr.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies))
        torch.add(y, 1, out=z)                       # a node after the SWITCH, depending on it
    main.instantiate()
    ctx[0] = int(handle)
    print(f"  SWITCH node built: handle={int(handle):#x}, {len(bodies)} bodies, "
          f"{len(body_nodes)} child nodes in the bodies")

    bad = 0

    def check(name, sel, want_y):
        nonlocal bad
        ctx[1] = sel
        main.replay()
        torch.cuda.synchronize()
        ok_y = torch.equal(y, want_y)
        ok_z = torch.equal(z, want_y + 1)
        bad += not (ok_y and ok_z)
        print(f"    {name:<34} y {'ok' if ok_y else 'FAIL'}  z=y+1 {'ok' if ok_z else 'FAIL'}")

    check("select body 0 (x*2)", 0, x * 2)
    check("select body 1 (x*3)", 1, x * 3)
    check("select body 0 again", 0, x * 2)
    y.fill_(-1.0)
    check("out of range sel=2: no body runs, y stays -1", 2, torch.full_like(x, -1.0))
    ex = main.raw_cuda_graph_exec()
    ck(cr.cudaGraphExecChildGraphNodeSetParams(ex, body_nodes[1], g1b.raw_cuda_graph()))
    check("body 1 child swapped to x*5, select 1", 1, x * 5)
    check("body 0 unaffected after the swap", 0, x * 2)
    print("\n  " + ("all passed: SWITCH + child nodes in bodies + device-side condition select + child swap inside a body, working from Python"
                    if not bad else f"{bad} item(s) failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
