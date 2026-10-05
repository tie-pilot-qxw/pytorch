"""Which library does each extern/fallback op actually launch? Collect CUDA kernel names.
No timing is reported -- names only."""
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from torch.autograd import DeviceType

dev = "cuda"

def kernels_of(fn, tag):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    names = []
    for ev in p.events():
        if ev.device_type == DeviceType.CUDA:
            names.append(ev.name)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    print(f"\n### {tag}")
    for n in out:
        print("    ", n[:150])
    return out

bf = torch.bfloat16
# --- GEMM family: does the kernel change with M? ---
for M in (128, 1024, 8192):
    a = torch.randn(M, 256, device=dev, dtype=bf)
    b = torch.randn(256, 256, device=dev, dtype=bf)
    o = torch.empty(M, 256, device=dev, dtype=bf)
    kernels_of(lambda: torch.mm(a, b, out=o), f"mm bf16 M={M} K=256 N=256")

for M in (128, 8192):
    a = torch.randn(M, 256, device=dev, dtype=bf)
    b = torch.randn(256, 256, device=dev, dtype=bf)
    bias = torch.randn(256, device=dev, dtype=bf)
    o = torch.empty(M, 256, device=dev, dtype=bf)
    kernels_of(lambda: torch.addmm(bias, a, b, alpha=1, beta=1, out=o), f"addmm bf16 M={M}")

a32 = torch.randn(1024, 256, device=dev); b32 = torch.randn(256, 256, device=dev)
o32 = torch.empty(1024, 256, device=dev)
kernels_of(lambda: torch.mm(a32, b32, out=o32), "mm fp32 M=1024")

for S in (64, 512):
    q = torch.randn(4, S, 64, device=dev, dtype=bf)
    k = torch.randn(4, 64, S, device=dev, dtype=bf)
    o = torch.empty(4, S, S, device=dev, dtype=bf)
    kernels_of(lambda: torch.bmm(q, k, out=o), f"bmm bf16 B=4 S={S}")
bb = torch.randn(4, 64, 64, device=dev, dtype=bf)
q = torch.randn(4, 64, 32, device=dev, dtype=bf); k = torch.randn(4, 32, 64, device=dev, dtype=bf)
ob = torch.empty(4, 64, 64, device=dev, dtype=bf)
kernels_of(lambda: torch.baddbmm(bb, q, k, out=ob), "baddbmm bf16")

# --- convolution ---
for H in (32, 64):
    x = torch.randn(4, 32, H, H, device=dev)
    w = torch.randn(64, 32, 3, 3, device=dev)
    kernels_of(lambda: torch.convolution(x, w, None, (1,1), (1,1), (1,1), False, (0,0), 1),
               f"convolution fp32 NCHW H={H}")
xc = torch.randn(4, 32, 32, 32, device=dev, dtype=bf).to(memory_format=torch.channels_last)
wc = torch.randn(64, 32, 3, 3, device=dev, dtype=bf).to(memory_format=torch.channels_last)
kernels_of(lambda: torch.convolution(xc, wc, None, (1,1), (1,1), (1,1), False, (0,0), 1),
           "convolution bf16 channels_last H=32")

# --- SDPA (cudnn backend, as inductor selected) ---
for S in (128, 512):
    q = torch.randn(2, 4, S, 64, device=dev, dtype=bf)
    k = torch.randn(2, 4, S, 64, device=dev, dtype=bf)
    v = torch.randn(2, 4, S, 64, device=dev, dtype=bf)
    kernels_of(lambda: torch.ops.aten._scaled_dot_product_cudnn_attention.default(q, k, v, None, False),
               f"_scaled_dot_product_cudnn_attention S={S}")
    kernels_of(lambda: torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v),
               f"_scaled_dot_product_flash_attention S={S}")

# --- sort / topk / cumsum / nonzero / scatter_add / index_put ---
x = torch.randn(8, 4096, device=dev)
kernels_of(lambda: torch.ops.aten.sort.stable(x, stable=False, dim=1, descending=False), "aten.sort.stable N=4096")
xs = torch.randn(8, 64, device=dev)
kernels_of(lambda: torch.ops.aten.sort.stable(xs, stable=False, dim=1, descending=False), "aten.sort.stable N=64")
kernels_of(lambda: torch.ops.aten.topk.default(x, 4, -1, True, True), "aten.topk.default")
kernels_of(lambda: torch.ops.aten.cumsum.default(x, 1), "aten.cumsum.default (eager)")
m = torch.randn(4096, device=dev) > 0
kernels_of(lambda: torch.ops.aten.nonzero.default(m), "aten.nonzero.default")
base = torch.randn(8, 64, device=dev); idx = torch.randint(0, 64, (8, 16), device=dev)
srcv = torch.randn(8, 16, device=dev)
kernels_of(lambda: base.clone().scatter_add_(1, idx, srcv), "aten.scatter_add_ (eager)")
kernels_of(lambda: torch.ops.aten.index_put_(base.clone(), [torch.randint(0,8,(4,),device=dev)], torch.randn(4,64,device=dev), False), "aten.index_put_ (eager)")
print("\nDONE")
