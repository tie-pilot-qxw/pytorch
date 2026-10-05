#!/usr/bin/env python3
"""A descriptor freezes the shape, not just the address: the case where the address stays put but the shape changes.

Buffer addresses in the arena are often **frozen** across shapes (fixed layouts are frozen by construction; with dynamic
layouts the first slot is frozen too). So any cache that "decides whether a descriptor is still usable by its address"
will hand DynaGraph a descriptor with a stale globalDim -- upstream `expand_host_tma_descriptor` compares only data_ptr
(`static_triton_launcher.py:62-75`; shape/strides are parameters it accepts but never looks at).

This probe creates exactly that case: the descriptor is built on the **first** buffer allocated in the region (address
fixed at the start of the arena), and the shape alternates among a few values. Any numeric error means the descriptor did not follow the shape.

Usage:
    python probe_tma_shape.py
`TORCHINDUCTOR_DYNAGRAPH_UPDATE` selects the host/device path as usual.
"""
import os, sys, logging
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo, torch._inductor.config as ic
import triton, triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.DEBUG)
ic.triton.dynagraph = os.environ.get("DG", "1") == "1"
ic.triton.dynagraph_extern_child = True
ic.triton.dynagraph_update = os.environ.get("TORCHINDUCTOR_DYNAGRAPH_UPDATE", "auto")
ic.force_disable_caches = True

tags, opaque = [], []
class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "fallback [" in m:
            tags.append(m.split("[", 1)[1].split("]", 1)[0])
        elif m.startswith("DynaGraph opaque kernel "):
            opaque.append(m[len("DynaGraph opaque kernel "):].split(":", 1)[0])
lg.addHandler(_Grab())

sigs = []
from torch._inductor.runtime import triton_heuristics as th
_o = th.CachingAutotuner.__init__
def _spy(self, *a, **kw):
    _o(self, *a, **kw)
    sigs.append(dict((self.triton_meta or {}).get("signature") or {}))
th.CachingAutotuner.__init__ = _spy

# The pushed 128 bytes themselves: for two pushes with the same address and different shapes, the bytes must differ.
# Without this check, "numerics correct" might only mean this kernel never uses the descriptor's globalDim.
pushed = []
from torch._inductor import dynagraph as dgm
_ob = dgm.DynaGraphRunner._desc_blobs
def _spy_blobs(self, srcs):
    r = _ob(self, srcs)
    if r is not None:
        for buf, _m, t in r[1]:
            pushed.append((t.data_ptr(), tuple(t.shape), bytes(buf.raw)[:128]))
    return r
dgm.DynaGraphRunner._desc_blobs = _spy_blobs

from torch._inductor import cudagraph_trees as ct
n_rec = {"v": 0}
_on = ct.CUDAGraphNode.__init__
def _rec(self, *a, **kw):
    n_rec["v"] += 1
    return _on(self, *a, **kw)
ct.CUDAGraphNode.__init__ = _rec

BM, BN = 32, 256


@triton.jit
def scale_desc(xd, yd, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_M
    yd.store([off, 0], xd.load([off, 0]) * 2.0 + 1.0)


class M(torch.nn.Module):
    def forward(self, x):
        # relu first so the descriptor's source is an allocated buffer (arena
        # slot 0, whose address does not move across shapes) rather than the
        # graph input.
        h = torch.relu(x)
        y = torch.empty_like(h)
        scale_desc[(triton.cdiv(h.shape[0], BM),)](
            TensorDescriptor.from_tensor(h, [BM, BN]),
            TensorDescriptor.from_tensor(y, [BM, BN]),
            BLOCK_M=BM, BLOCK_N=BN,
        )
        return y.sum(-1)


m = M().cuda()
g = torch.compile(m, dynamic=True, mode="reduce-overhead")
worst, addrs = 0.0, set()
with torch.no_grad():
    for rows in (64, 128, 64, 192, 128):
        x = torch.randn(rows, BN, device="cuda")
        torch._dynamo.mark_dynamic(x, 0)
        got, ref = g(x), m(x)
        d = ((got - ref).abs().max() / ref.abs().max().clamp_min(1)).item()
        worst = max(worst, d)
        print(f"    rows={rows:4d}  rel diff {d:.1e}", flush=True)

desc = [(nm, ty) for sg in sigs for nm, ty in sg.items()
        if isinstance(ty, str) and (ty == "nvTmaDesc" or ty.startswith("tensordesc<"))]
print(f"  dynagraph {'on' if ic.triton.dynagraph else 'off'}  update {ic.triton.dynagraph_update}  "
      f"descriptors in signatures {len(desc)}  opaque {sorted(set(opaque)) or '-'}  records {n_rec['v']}  "
      f"tags {sorted(set(tags)) or '-'}  max rel diff {worst:.1e}")
if not desc:
    print("  inconclusive: this compile has no kernel with a descriptor argument")
    sys.exit(2)
ok = worst < 1e-5 and not tags and (n_rec["v"] == 0 or not ic.triton.dynagraph)

# When the parameter-patching route is taken, check "address frozen" and "bytes follow the shape" separately.
patched = ic.triton.dynagraph and ic.triton.dynagraph_tma_patch
if patched:
    by_addr = {}
    for a, sh, mp in pushed:
        by_addr.setdefault(a, {})[sh] = mp
    frozen = [a for a, d in by_addr.items() if len(d) > 1]
    varied = [a for a in frozen if len(set(by_addr[a].values())) > 1]
    print(f"  descriptors {len(by_addr)}  address unchanged across shapes {len(frozen)}  "
          f"of which 128 bytes changed {len(varied)}")
    if not frozen:
        print("  inconclusive: no descriptor's source landed at the same address under two shapes, "
              "so the case this probe tests never happened")
        sys.exit(2)
    if varied != frozen:
        print("  FAILED: address unchanged and shape changed, yet the pushed 128 bytes are identical -- "
              "the descriptor did not follow the shape, and correct numerics are a coincidence")
        ok = False

print("  passed: address fixed, shape changing, and the descriptor still follows the shape" if ok
      else "  FAILED: the shape changed but the descriptor did not follow, or the region was not served")
sys.exit(0 if ok else 1)
