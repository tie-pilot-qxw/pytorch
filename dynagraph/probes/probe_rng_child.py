#!/usr/bin/env python3
r"""How random numbers behave in a graph: a randint captured in the main graph (torch's graph-safe RNG) produces new numbers on every replay;
a randint captured as its own small graph and inserted into the main graph as a child node has nobody advancing the philox offset on main-graph replay --
every replay gives the same numbers. That is why seed-style ops are run on the host once before each call (`_EAGER_OP_NAMES`)
instead of as a child."""
import sys
import torch
from cuda.bindings import runtime as cr
from torch.cuda._utils import _check_cuda_bindings as ck


def main() -> int:
    torch.manual_seed(0)
    bad = 0
    # A: randint captured inside the main graph
    buf = torch.zeros(4, dtype=torch.int64, device="cuda")
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.randint(-2**63, 2**63 - 1, [4], out=buf)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        torch.randint(-2**63, 2**63 - 1, [4], out=buf)
    vals = []
    for _ in range(3):
        g.replay(); torch.cuda.synchronize(); vals.append(buf.clone())
    same_a = all(torch.equal(vals[0], v) for v in vals[1:])
    print(f"  randint captured in main graph: 3 replays {'all identical' if same_a else 'all different'}  ({'FAIL' if same_a else 'OK graph-safe RNG'})")
    bad += same_a

    # B: randint captured into its own graph, then a child node of a main graph
    buf2 = torch.zeros(4, dtype=torch.int64, device="cuda")
    child = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(child):
        torch.randint(-2**63, 2**63 - 1, [4], out=buf2)
    main = torch.cuda.CUDAGraph(keep_graph=True)
    y = torch.zeros(4, dtype=torch.int64, device="cuda")
    with torch.cuda.graph(main):
        st = torch.cuda.current_stream().cuda_stream
        info = ck(cr.cudaStreamGetCaptureInfo(st))
        cap, deps = info[2], list(info[3] or [])
        node = ck(cr.cudaGraphAddChildGraphNode(cap, deps, len(deps), child.raw_cuda_graph()))
        ck(cr.cudaStreamUpdateCaptureDependencies(st, [node], None, 1, cr.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies))
        torch.add(buf2, 1, out=y)
    main.instantiate()
    vals = []
    for _ in range(3):
        main.replay(); torch.cuda.synchronize(); vals.append(y.clone())
    same_b = all(torch.equal(vals[0], v) for v in vals[1:])
    print(f"  randint in a child node: 3 replays {'all identical' if same_b else 'all different'}  ({'OK so it must run on the host' if same_b else 'FAIL unexpected: child also gives new numbers'})")
    bad += not same_b

    # C: seed op run on the host before each replay, graph reads the slot
    buf3 = torch.zeros(4, dtype=torch.int64, device="cuda")
    y3 = torch.zeros(4, dtype=torch.int64, device="cuda")
    g3 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g3):
        torch.add(buf3, 1, out=y3)
    vals = []
    for _ in range(3):
        torch.randint(-2**63, 2**63 - 1, [4], out=buf3)
        g3.replay(); torch.cuda.synchronize(); vals.append(y3.clone())
    same_c = all(torch.equal(vals[0], v) for v in vals[1:])
    print(f"  host runs randint before each replay: 3 times {'all identical' if same_c else 'all different'}  ({'FAIL' if same_c else 'OK'})")
    bad += same_c
    print("\n  " + ("all passed" if not bad else f"{bad} checks failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
