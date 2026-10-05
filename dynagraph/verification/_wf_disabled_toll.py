#!/usr/bin/env python3
"""How much does each replay pay for nothing when disabled nodes stay in the graph?

This is the last number for deciding whether "preplace V variants linearly + SetEnabled" is usable.
The switch cost is one-time (2.13us), but the disabled nodes sit in the graph on **every replay**.
If each costs ~0.8us (docs/notes/FEASIBILITY.md records "inherent overhead of serial nodes in a graph
0.78-0.80us/node"), then preplacing V variants is a standing tax on every step, not something
paid only when switching.

Same small kernel: put N nodes in the graph but enable only 1, sweep N.
"""
import os, statistics
import torch
from cuda.bindings import runtime as cr
from torch.cuda._utils import _check_cuda_bindings as ck

REPS = int(os.environ.get("REPS", "200"))
NS = [int(v) for v in os.environ.get("NS", "1,2,4,8,16,32").split(",")]
x = torch.ones(1024, device="cuda")
HOLD = []

# Build all graphs first, then measure **interleaved**: the card always has other jobs on it, and
# if one N gets its own time window, their jitter is recorded as that N's cost (see docs/METHODOLOGY.md).
ROUNDS = int(os.environ.get("ROUNDS", "9"))
graphs = {}
for n in NS:
    ys = [torch.empty_like(x) for _ in range(n)]
    HOLD.append(ys)
    s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        for i in range(n): torch.add(x, float(i), out=ys[i])
    torch.cuda.current_stream().wait_stream(s_)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        for i in range(n): torch.add(x, float(i), out=ys[i])
    torch.cuda.synchronize()
    HOLD.append(g)
    raw = g.raw_cuda_graph()
    g.replay(); torch.cuda.synchronize()
    ex = g.raw_cuda_graph_exec()
    nn = ck(cr.cudaGraphGetNodes(raw))[1]
    nodes = list(ck(cr.cudaGraphGetNodes(raw, nn))[0])
    for k, nd in enumerate(nodes):
        ck(cr.cudaGraphNodeSetEnabled(ex, nd, 1 if k == 0 else 0))
    for _ in range(20): g.replay()
    graphs[n] = g
torch.cuda.synchronize()


def once(g):
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(REPS): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / REPS


samples = {n: [] for n in NS}
for _ in range(ROUNDS):
    for n in NS:
        samples[n].append(once(graphs[n]))

print(f"Each graph has N identical small kernels, only #0 enabled. REPS={REPS} ROUNDS={ROUNDS}, interleaved")
print("N      median us/replay  min us    us per disabled node (vs N=1)")
base = None
for n in NS:
    v = sorted(samples[n])
    med = statistics.median(v)
    if base is None: base = med
    per = (med - base) / (n - 1) if n > 1 else 0.0
    print(f"{n:<6} {med:<17.2f} {v[0]:<9.2f} {per:.3f}")
meds = [statistics.median(samples[n]) for n in NS]
slope = (meds[-1] - meds[0]) / (NS[-1] - NS[0])
print(f"\nFitted slope: {slope:.3f} us/replay per disabled node"
      f"  (enabled serial nodes cost 0.78-0.80 us/node, docs/notes/FEASIBILITY.md)")
