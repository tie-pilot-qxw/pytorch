"""Break one workload's eager training step into: host issue time (no sync), total GPU kernel time, kernel count.
Usage: python e2e/breakdown.py sage [sage's arguments...]
"""
import sys, os, time, importlib
sys.path.insert(0, os.path.dirname(__file__))
name = sys.argv[1]
sys.argv = [name + ".py"] + sys.argv[2:] + ["--modes", "eager"]
import harness, torch
cap = {}
def fake_run(make, batches, modes, **kw):
    cap["make"], cap["batches"] = make, batches
harness.run = fake_run
mod = importlib.import_module(name)
mod.main()
make, batches = cap["make"], cap["batches"]
model, step = make()
for b in batches[:5]:
    step(model, b)
torch.cuda.synchronize()
host = []
for b in batches[5:25]:
    torch.cuda.synchronize(); t0 = time.perf_counter()
    step(model, b); t1 = time.perf_counter()
    torch.cuda.synchronize(); t2 = time.perf_counter()
    host.append((t1 - t0, t2 - t0))
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for b in batches[25:30]:
        step(model, b)
    torch.cuda.synchronize()
ev = [e for e in p.events() if e.device_type.name == "CUDA"]
k = sum(e.device_time for e in ev) / 5 / 1e3
host.sort()
print(f"[{name}] per step: host issue {host[len(host)//2][0]*1e3:.2f} ms, wall {sorted(h[1] for h in host)[len(host)//2]*1e3:.2f} ms, "
      f"GPU kernel total {k:.2f} ms, kernel/memcpy count {len(ev)/5:.0f}")
