import os, re, collections, torch, torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch._inductor.utils import run_and_get_code
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29577")
dist.init_process_group("nccl", rank=0, world_size=1)
dev="cuda"
def f(x):
    y = x * 2
    y = funcol.all_reduce(y, "sum", "0")
    z = funcol.all_gather_tensor(y, 0, "0")
    w = funcol.reduce_scatter_tensor(z, "sum", 0, "0")
    return w + 1
cf = torch.compile(f, dynamic=True)
x = torch.randn(64, 32, device=dev)
_, codes = run_and_get_code(cf, x)
CALL = re.compile(r"(torch\.ops\.[\w.]+|extern_kernels\.[\w.]+)\s*\(")
OUT = os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "_wf_externsurv_wrappers")
os.makedirs(OUT, exist_ok=True)
for i,c in enumerate(codes):
    p=f"{OUT}/comm_{i}.py"; open(p,"w").write(c)
    lines=[l.strip() for l in c.splitlines() if ("torch.ops._c10d" in l or "extern_kernels." in l) and not l.lstrip().startswith("#")]
    print(f"-- {p}")
    for l in lines: print("   ", l[:160])
# kernel names
from torch.profiler import profile, ProfilerActivity
from torch.autograd import DeviceType
cf(x); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    cf(x); torch.cuda.synchronize()
seen=[]
for ev in p.events():
    if ev.device_type==DeviceType.CUDA and ev.name not in seen: seen.append(ev.name)
print("\nCUDA kernels launched:")
for n in seen: print("   ", n[:130])
dist.destroy_process_group()
