#!/usr/bin/env python3
"""User-written @triton.jit kernels (RMSNorm, element-wise scale, grid as a lambda) inside the region.

Inductor evaluates the grid lambda into an expression in the wrapper and passes it as a launch argument (FixedGrid), which the kernel table already understands;
it used to be refused with `no-kernels`: the constexpr (BLOCK) the user kernel declares is not at the call site, so argument alignment was off by one.
Under DynaGraph user kernels also go through the static launcher and have device handles: auto picks the host path (no planner needed), and forcing device also serves it.
"""
import os, sys, logging
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo, torch._inductor.config as ic
import triton, triton.language as tl
logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.INFO)
ic.triton.dynagraph = True
ic.triton.dynagraph_extern_child = True
ic.triton.dynagraph_update = os.environ.get("TORCHINDUCTOR_DYNAGRAPH_UPDATE", "auto")
ic.force_disable_caches = True
tags = []
class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "fallback [" in m: tags.append(m.split("[", 1)[1].split("]", 1)[0])
lg.addHandler(_Grab())

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    y = x * tl.rsqrt(var + eps) * tl.load(w_ptr + cols, mask=mask, other=1.0)
    tl.store(y_ptr + row * n_cols + cols, y, mask=mask)

@triton.jit
def scale_kernel(x_ptr, y_ptr, n, s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * s, mask=mask)

def rmsnorm(x, w, eps=1e-6):
    y = torch.empty_like(x)
    n_rows, n_cols = x.shape
    rmsnorm_kernel[(n_rows,)](x, w, y, n_cols, eps, BLOCK=triton.next_power_of_2(n_cols))
    return y

def scale(x, s):
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    scale_kernel[grid](x, y, n, s, BLOCK=256)
    return y

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.linspace(0.5, 1.5, 64))
        self.lin = torch.nn.Linear(64, 64)
    def forward(self, x):
        h = rmsnorm(self.lin(x), self.w)
        return scale(torch.relu(h) + 1, 0.5).sum(-1)

from torch._inductor import cudagraph_trees as ct
n_rec = {"v": 0}
_orig = ct.CUDAGraphNode.__init__
def _rec(self, *a, **kw):
    n_rec["v"] += 1
    return _orig(self, *a, **kw)
ct.CUDAGraphNode.__init__ = _rec

m = M().cuda()
f = torch.compile(m, dynamic=True, mode="reduce-overhead")
worst = 0.0
with torch.no_grad():
    for L in (64, 64, 40, 100, 7, 256, 64):
        x = torch.randn(L, 64, device="cuda")
        # The user kernel's grid lambda makes Dynamo specialize the first compile on this L (even with dynamic=True),
        # and upstream records that static graph once; marking the dim dynamic compiles a symbolic shape from the start.
        torch._dynamo.mark_dynamic(x, 0)
        out = f(x); ref = m(x)
        worst = max(worst, ((out - ref).abs().max() / ref.abs().max().clamp_min(1)).item())
upd = ic.triton.dynagraph_update
forced_device = upd == "device"
print(f"  update {upd}  records {n_rec['v']}  tags {sorted(set(tags)) or '-'}  max rel diff {worst:.1e}")
if forced_device:
    ok = worst < 1e-5 and not tags and n_rec["v"] == 0
    print("  forced device: user kernels get device handles via the static launcher, region served by one graph (device path)" if ok else "  FAILED")
else:
    ok = worst < 1e-5 and n_rec["v"] == 0 and not tags
    print("  all passed: the user-kernel region is served by one graph (host path)" if ok else "  FAILED")
sys.exit(0 if ok else 1)
