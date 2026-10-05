#!/usr/bin/env python3
"""Host-side TMA descriptors on a real modern workload: mxfp dequantization from OpenAI `triton_kernels`.

`triton_kernels/numerics_details/mxfp.py:140` has
    out_desc = TensorDescriptor.from_tensor(reshaped_out, [BLOCK_OUT_DIM, BLOCK_QUANT_DIM])
    _upcast_from_mxfp[grid](out_desc, tensor_desc, ...)
that is, the descriptor is built on the host and passed straight as an argument to the @triton.jit kernel -- exactly the kind DynaGraph's
kernel table cannot understand. Using it instead of a toy kernel I wrote myself confirms that this path behaves the same on the real thing.

How to run:
    python probe_tma_real.py
Reports: how many regions were asked, how many served, recording count, fallback tags, which kernels went opaque, numeric diff.
"""
import os, sys, logging
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo, torch._inductor.config as ic

logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.DEBUG)
ic.triton.dynagraph = os.environ.get("DG", "1") == "1"
ic.triton.dynagraph_extern_child = True
ic.triton.dynagraph_update = os.environ.get("TORCHINDUCTOR_DYNAGRAPH_UPDATE", "auto")
ic.force_disable_caches = True

tags, opaque, served, asked = [], [], [0], [0]
class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "fallback [" in m:
            tags.append(m.split("[", 1)[1].split("]", 1)[0])
        elif m.startswith("DynaGraph opaque kernel "):
            opaque.append(m[len("DynaGraph opaque kernel "):].split(":", 1)[0])
        elif "DynaGraph served" in m:
            served[0] += 1
lg.addHandler(_Grab())

try:
    from triton_kernels.numerics_details.mxfp import (
        downcast_to_mxfp, upcast_from_mxfp, upcast_from_mxfp_torch,
    )
except Exception as e:
    print(f"  skipped: triton_kernels not available ({type(e).__name__}: {e})")
    sys.exit(0)

# Prove that this run really compiled a kernel with descriptor arguments; otherwise this probe tested nothing.
sigs = []
from torch._inductor.runtime import triton_heuristics as th
_o_init = th.CachingAutotuner.__init__
def _spy(self, *a, **kw):
    _o_init(self, *a, **kw)
    sigs.append(((self.inductor_meta or {}).get("kernel_name", "?"),
                 dict((self.triton_meta or {}).get("signature") or {})))
th.CachingAutotuner.__init__ = _spy

n_break = {"v": 0}
import torch._dynamo.convert_frame as _cf

from torch._inductor import cudagraph_trees as ct
n_rec = {"v": 0}
_orig = ct.CUDAGraphNode.__init__
def _rec(self, *a, **kw):
    n_rec["v"] += 1
    return _orig(self, *a, **kw)
ct.CUDAGraphNode.__init__ = _rec


def build(rows):
    x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    q, s = downcast_to_mxfp(x, torch.uint8, axis=1)
    return q, s


def f(q, s):
    # Do a bit of elementwise work after dequantizing, so the region holds more than just this one kernel.
    y = upcast_from_mxfp(q, s, torch.bfloat16, axis=1)
    return (y * 2.0 + 1.0).sum(-1)


g = torch.compile(f, dynamic=True, mode="reduce-overhead")
worst = 0.0
with torch.no_grad():
    cache = {}
    for rows in (128, 128, 256, 64, 192):
        if os.environ.get("REUSE") == "1":
            if rows not in cache:
                cache[rows] = build(rows)
            q, s = cache[rows]
        else:
            q, s = build(rows)
        torch._dynamo.mark_dynamic(q, 0)
        torch._dynamo.mark_dynamic(s, 0)
        got = g(q, s)
        ref = f(q, s)
        d = (got - ref).abs().max() / ref.abs().max().clamp_min(1)
        worst = max(worst, d.item())
        print(f"    rows={rows:4d}  rel diff {d.item():.1e}", flush=True)

desc = [(k, nm, ty) for k, sg in sigs for nm, ty in sg.items()
        if isinstance(ty, str) and (ty == "nvTmaDesc" or ty.startswith("tensordesc<"))]
print(f"  dynagraph {'on' if ic.triton.dynagraph else 'off'}  compiled kernels {len(sigs)}  descriptors in signatures {len(desc)}  recordings {n_rec['v']}  "
      f"opaque {sorted(set(opaque)) or '-'}  tags {sorted(set(tags)) or '-'}  max rel diff {worst:.1e}")
for k, nm, ty in desc[:4]:
    print(f"    {k}  {nm}: {ty}")
if not desc:
    print("  Inconclusive: no kernel in this compile has descriptor arguments, "
          "so the mxfp kernel never entered the compiled region (most likely a graph break, or it was treated as an opaque call)")
    sys.exit(2)
ok = worst < 1e-2 and not tags and (n_rec["v"] == 0 or not ic.triton.dynagraph)
print("  Result: the region with real mxfp dequantization is served by one graph" if ok else "  failed")
sys.exit(0 if ok else 1)
