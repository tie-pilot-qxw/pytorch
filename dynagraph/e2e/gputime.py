"""Per step in compile mode: wall clock, host launch time (no sync), total GPU kernel time, kernel count. Tells whether it is launch-bound.
Usage: MODE=compile python e2e/gputime.py pointcloud [args...]
"""
import sys, os, time, importlib
sys.path.insert(0, os.path.dirname(__file__))
name = sys.argv[1]
sys.argv = [name + ".py"] + sys.argv[2:]
import harness, torch
cap = {}
harness.run = lambda make, batches, modes, **kw: cap.update(make=make, batches=batches)
importlib.import_module(name).main()
mode = os.environ.get("MODE", "compile")
model, step = cap["make"]()
f = harness.compiled(model, mode)
bs = cap["batches"]
for b in bs[:10]:
    step(f, b)
torch.cuda.synchronize()
host, wall = [], []
for b in bs[10:30]:
    torch.cuda.synchronize(); t0 = time.perf_counter()
    step(f, b); t1 = time.perf_counter(); torch.cuda.synchronize(); t2 = time.perf_counter()
    host.append(t1 - t0); wall.append(t2 - t0)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for b in bs[30:40]:
        step(f, b)
    torch.cuda.synchronize()
ev = [e for e in p.events() if e.device_type.name == "CUDA"]
med = lambda x: sorted(x)[len(x) // 2] * 1e3
print(f"[{name} {mode}] per step: wall {med(wall):.2f} ms, host launch {med(host):.2f} ms, GPU kernels total {sum(e.device_time for e in ev) / 10 / 1e3:.2f} ms, kernels {len(ev) / 10:.0f}")
if os.environ.get("TOPK"):
    import collections
    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in ev:
        agg[e.name][0] += e.device_time
        agg[e.name][1] += 1
    for nm, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[: int(os.environ["TOPK"])]:
        print(f"   {t / 10 / 1e3:7.3f} ms/step  {c / 10:6.1f} calls/step  {nm[:110]}")
