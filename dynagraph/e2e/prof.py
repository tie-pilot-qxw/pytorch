"""Run torch.profiler on one mode over the new segment: list the most expensive kernels on the GPU and the most expensive ops on the host.
Usage: MODE=compile python e2e/prof.py sage [sage args...]
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(__file__))
name = sys.argv[1]
sys.argv = [name + ".py"] + sys.argv[2:]
import harness, torch
cap = {}
harness.run = lambda make, batches, modes, **kw: cap.update(make=make, batches=batches, warm=kw.get("warm", 20))
importlib.import_module(name).main()
mode = os.environ.get("MODE", "compile")
model, step = cap["make"]()
f = harness.compiled(model, mode)
W = cap["warm"]
for b in cap["batches"][:W]:
    step(f, b)
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for b in cap["batches"][W:W + 10]:
        step(f, b)
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=15, max_name_column_width=70))
print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20, max_name_column_width=70))
