#!/usr/bin/env python3
"""How the cost of cudaGraphExecUpdate grows with the node count.

This settles a design question: when a site's cluster changes, do we swap it with a whole-graph ExecUpdate,
or pre-capture a SWITCH body and switch to it. SWITCH is a constant ~9us (one switch, independent of graph size),
ExecUpdate diffs the whole graph, so there must be a crossover point. This measures it.

The two graphs have the same topology and different scalar params on every node; alternate A->B->A->B and time only ExecUpdate itself.
"""
import os, statistics, time
import torch
from cuda.bindings import runtime as cr
from torch.cuda._utils import _check_cuda_bindings as ck

REPS = int(os.environ.get("REPS", "40"))
NS = [int(v) for v in os.environ.get("NS", "1,8,48,200,1000").split(",")]
x = torch.randn(256, 256, device="cuda")
HOLD = []  # the output tensors of both graphs must stay alive: the graph freezes their addresses, and if they are freed it stomps on someone else's memory


def build(n, k):
    ys = [torch.empty_like(x) for _ in range(n)]
    HOLD.append(ys)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(n):
            torch.add(x, float(k + i), out=ys[i])
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        for i in range(n):
            torch.add(x, float(k + i), out=ys[i])
    torch.cuda.synchronize()
    return g


print(f"REPS={REPS}  alternating A<->B, timing only cudaGraphExecUpdate")
print("nodes    median_us   per_node_us min_us    p90us")
rows = []
for n in NS:
    ga, gb = build(n, 1), build(n, 1000)
    ga.replay(); torch.cuda.synchronize()
    ex = ga.raw_cuda_graph_exec()
    ts = []
    for r in range(REPS):
        tgt = gb if r % 2 == 0 else ga
        t0 = time.perf_counter_ns()
        rc = cr.cudaGraphExecUpdate(ex, tgt.raw_cuda_graph())
        dt = (time.perf_counter_ns() - t0) / 1000.0
        if int(rc[0]) != 0:
            print(f"n={n} ExecUpdate failed err={int(rc[0])}"); ts = []; break
        ts.append(dt)
    if not ts:
        continue
    ga.replay(); torch.cuda.synchronize()
    v = sorted(ts)
    med = statistics.median(v)
    rows.append((n, med))
    print(f"{n:<8} {med:<11.2f} {med / n:<11.3f} {v[0]:<9.2f} {v[int(len(v) * 0.9)]:.2f}")
    torch.cuda.synchronize()

if len(rows) >= 2:
    (n1, t1), (n2, t2) = rows[0], rows[-1]
    slope = (t2 - t1) / (n2 - n1)
    fixed = t1 - slope * n1
    print(f"\nfit: fixed overhead {fixed:.2f} us + per node {slope:.3f} us")
    print(f"vs the constant ~9 us of SWITCH: crossover at about {max(0, (9 - fixed) / slope):.0f} nodes")
