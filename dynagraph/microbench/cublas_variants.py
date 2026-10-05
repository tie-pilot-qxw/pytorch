import torch, random
from torch.profiler import profile, ProfilerActivity
torch.manual_seed(0); random.seed(0)
K, N = 4096, 4096      # Llama-7B q_proj-like; M = tokens in the micro-batch
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
Ms = sorted(set([random.randint(1, 8192) for _ in range(300)] + [2**i for i in range(0, 14)] + [128*i for i in range(1, 65)]))
xmax = torch.randn(8192, K, device="cuda", dtype=torch.bfloat16); xs = {M: xmax[:M] for M in Ms}
for M in Ms[:5]: xs[M] @ w.t()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for M in Ms:
        xs[M] @ w.t()
    torch.cuda.synchronize()
names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
from collections import Counter
c = Counter(names)
print(f"cuBLAS bf16 GEMM [M x {K}] @ [{K} x {N}]^T over {len(Ms)} distinct M in [1, 8192]: {len(c)} distinct kernels, {len(names)} launches")
for k, v in c.most_common(): print(f"  {v:4d}x  {k[:110]}")
