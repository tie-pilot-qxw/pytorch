#!/usr/bin/env python3
"""A user kernel with @triton.autotune and several configs: Inductor gives it a PrecomputedGrid.

The grid formulas are text computed per config at codegen time (written onto the extra launch
arguments `_launcher_sN`), and which one takes effect is only known after the autotuner settles. Both update paths must serve it, bitwise correct.
"""
import logging, sys
import torch, triton, triton.language as tl
from torch._inductor import config as ic

tags = []
class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "DynaGraph fallback [" in m:
            tags.append(m.split("[", 1)[1].split("]", 1)[0])
logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.INFO); lg.addHandler(_Grab())

ic.triton.dynagraph = True
ic.force_disable_caches = True

@triton.autotune(
    configs=[triton.Config({"BLOCK": b}, num_warps=4) for b in (128, 256, 512)],
    key=["n"],
)
@triton.jit
def scale_kernel(x_ptr, y_ptr, n, s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * s, mask=mask)

class M(torch.nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        scale_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, y, n, 2.0)
        return torch.relu(y).sum(-1)

n_rec = {"v": 0}
from torch._inductor import cudagraph_trees as ct
_orig = ct.CUDAGraphNode.__init__
def _rec(self, *a, **kw):
    n_rec["v"] += 1
    return _orig(self, *a, **kw)
ct.CUDAGraphNode.__init__ = _rec

m = M().cuda().eval()
f = torch.compile(m, dynamic=True, mode="reduce-overhead")
worst = 0.0
with torch.no_grad():
    for step, L in enumerate([64, 200, 128, 333, 96]):
        g = torch.Generator(device="cuda"); g.manual_seed(step)
        x = torch.randn(L, 96, device="cuda", generator=g)
        torch._dynamo.mark_dynamic(x, 0)
        ref = m(x); out = f(x); torch.cuda.synchronize()
        worst = max(worst, ((out - ref).abs().max() / ref.abs().max().clamp_min(1)).item())

upd = ic.triton.dynagraph_update
print(f"  update {upd}  recordings {n_rec['v']}  tags {sorted(set(tags)) or '-'}  max rel diff {worst:.1e}")
ok = worst < 1e-6 and n_rec["v"] == 0 and not tags
print("  all passed: autotuned user kernel (PrecomputedGrid) served by one graph" if ok else "  FAILED")
sys.exit(0 if ok else 1)
