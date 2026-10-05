#!/usr/bin/env python3
"""Linearly preplaced variant nodes + SetEnabled: can each one carry a different cluster? Are disabled nodes really off?

`docs/notes/FEASIBILITY.md` (section "2. All measured data") recorded "preplaced nodes + SetEnabled, ~0.85us per extra slot, cheaper than
SWITCH when V<=13", but two things were never tested:
  (1) whether the preplaced variant nodes can **each carry a different cluster** -- this is the only wall for cuBLAS/DeepGEMM.
      cluster is a node attribute, fixed at instantiate time; if each variant is its own node,
      each should carry its own, needing neither SWITCH nor ExecUpdate.
  (2) `docs/notes/SOLUTION.md` (section "Probe problems found by the review (showing how easily the criteria themselves run idle)") notes that the probe back then did not prove the disable path:
      "all that was actually proven is that SetEnabled(0)/(1) are mutually consistent". Here sentinel values check "did a disabled node write".

Method: capture three addmm with M=512/1024/4096 (measured clusters (0,0,0)/(1,8,1)/(2,1,1) respectively)
into **the same graph**, enable only one at a time, and check the sentinels.
"""
import ctypes, os, statistics, time
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

SENT = 12288.0
K = N = 512
MS = [int(v) for v in os.environ.get("MS", "512,1024,4096").split(",")]
REPS = int(os.environ.get("REPS", "40"))
torch.manual_seed(0)
x = torch.randn(max(MS), K, device="cuda", dtype=torch.float32)
w = torch.randn(K, N, device="cuda", dtype=torch.float32)
b = torch.randn(N, device="cuda", dtype=torch.float32)
outs = [torch.empty(M, N, device="cuda", dtype=torch.float32) for M in MS]

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        for i, M in enumerate(MS):
            torch.addmm(b, x[:M], w, out=outs[i])
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(g):
    for i, M in enumerate(MS):
        torch.addmm(b, x[:M], w, out=outs[i])
torch.cuda.synchronize()
raw = g.raw_cuda_graph()
g.replay(); torch.cuda.synchronize()
ex = g.raw_cuda_graph_exec()

nn = ck(cr.cudaGraphGetNodes(raw))[1]
nodes = list(ck(cr.cudaGraphGetNodes(raw, nn))[0])
info = []
for nd in nodes:
    if int(getattr(ck(cr.cudaGraphNodeGetType(nd)), "value", 0)) != 0:
        continue
    dp = ck(cd.cuGraphKernelNodeGetParams(nd))
    try:
        cl = ck(cd.cuGraphKernelNodeGetAttribute(nd, cd.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION))
        cluster = (cl.clusterDim.x, cl.clusterDim.y, cl.clusterDim.z)
    except RuntimeError:
        cluster = None
    info.append((nd, ck(cd.cuFuncGetName(dp.func)).decode()[:52], cluster,
                 (dp.gridDimX, dp.gridDimY, dp.gridDimZ), int(dp.sharedMemBytes)))

print(f"{len(info)} kernel nodes in one graph:")
for i, (_, nm, cl, gr, sm) in enumerate(info):
    print(f"  node{i}  M={MS[i] if i < len(MS) else '?':<6} cluster={str(cl):<12} grid={str(gr):<14} smem={sm:<8} {nm}")
clusters = {c for _, _, c, _, _ in info}
print(f"  -> {len(clusters)} distinct clusters in the same instantiated graph: {sorted(clusters, key=str)}\n")


def set_only(k):
    for i, (nd, *_) in enumerate(info):
        ck(cr.cudaGraphNodeSetEnabled(ex, nd, 1 if i == k else 0))


print("Enable only one, disable the rest (sentinels check 'did a disabled node secretly write')")
print("on      cluster        this node's values  other nodes written?")
for k in range(len(info)):
    set_only(k)
    for t in outs:
        t.fill_(SENT)
    g.replay(); torch.cuda.synchronize()
    ref = torch.addmm(b, x[:MS[k]], w)
    err = (outs[k] - ref).abs().max().item()
    unwritten = int((outs[k] == SENT).sum().item())
    dirty = sum(int((outs[j] != SENT).sum().item()) for j in range(len(outs)) if j != k)
    ok = "OK" if (err == 0 and unwritten == 0 and dirty == 0) else "**BAD**"
    print(f"  {k}     {str(info[k][2]):<14} max|err|={err:<9.4g} unwritten={unwritten:<6} others_written={dirty:<8} {ok}")

# Switch cost: alternate only-k-on -> only-(k+1)-on, timing the SetEnabled calls themselves
print(f"\nSwitch cost (two SetEnabled calls, REPS={REPS})")
ts = []
for r in range(REPS):
    k = r % len(info)
    prev = (k - 1) % len(info)
    t0 = time.perf_counter_ns()
    ck(cr.cudaGraphNodeSetEnabled(ex, info[prev][0], 0))
    ck(cr.cudaGraphNodeSetEnabled(ex, info[k][0], 1))
    ts.append((time.perf_counter_ns() - t0) / 1000.0)
v = sorted(ts)
print(f"  median {statistics.median(v):.2f} us   min {v[0]:.2f}   p90 {v[int(len(v)*0.9)]:.2f}"
      f"   (SWITCH is a constant ~9-11 us)")

# Do disabled nodes still cost anything during replay
print("\nDo disabled nodes still cost something on every replay")
for label, on in (("all on", None), ("only 1 on", 0)):
    if on is None:
        for nd, *_ in info:
            ck(cr.cudaGraphNodeSetEnabled(ex, nd, 1))
    else:
        set_only(on)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(REPS):
        g.replay()
    ev1.record(); torch.cuda.synchronize()
    print(f"  {label:<10} {ev0.elapsed_time(ev1) * 1000 / REPS:.1f} us/replay")
