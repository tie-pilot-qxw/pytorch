#!/usr/bin/env python3
"""TMA: what happens to DynaGraph when a kernel argument is a tensor descriptor instead of a pointer.

Triton signatures spell descriptors two ways: the old experimental `nvTmaDesc`, and the new
stable `tensordesc<fp32[64, 64]>`. Neither starts with `*`, so the kernel table treats them as
scalars; worse, one stable signature entry expands into several real parameters in the cubin
(base pointer + per-dim shape/stride + two flags + block shape/stride),
so using `args.index(nm)` as the cubin parameter index is shifted wholesale from that point on,
every later parameter's byte offset points at someone else, and cuFuncGetParamInfo still returns valid values --
no error, it just patches the wrong bytes.

Also, the descriptor itself encodes the global address and globalDim (i.e. the shape) into its 128 bytes,
so when one graph serves many shapes, the descriptor is the thing that is frozen.

Usage:
    python probe_tma.py            # Inductor generates TMA itself (use_tensor_descriptor)
    python probe_tma.py device     # tl.make_tensor_descriptor in a user kernel (descriptor built on device)
    python probe_tma.py host       # user kernel receives a host-built TensorDescriptor
"""
import os, sys, logging
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo, torch._inductor.config as ic
import triton, triton.language as tl

logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.DEBUG)
ic.triton.dynagraph = True
ic.triton.dynagraph_extern_child = True
ic.triton.dynagraph_update = os.environ.get("TORCHINDUCTOR_DYNAGRAPH_UPDATE", "auto")
ic.force_disable_caches = True

MODE = (sys.argv[1] if len(sys.argv) > 1 else "all")
if MODE == "all":
    # One process per mode: switches like use_tensor_descriptor cannot be turned back once changed.
    import subprocess
    rcs = {}
    for m in ("inductor", "device", "host", "host_undeclared"):
        rcs[m] = subprocess.run([sys.executable, __file__, m]).returncode
    print(f"  modes {rcs}")
    ok = all(v == 0 for v in rcs.values())
    route = "parameter patching" if ic.triton.dynagraph_tma_patch else "an opaque site"
    print(f"  all passed: device-side descriptors are served by one graph, host-side descriptors are also served via {route}, "
          "undeclared ones fall back to an opaque site" if ok else "  FAILED")
    sys.exit(0 if ok else 1)

tags, opaque = [], []
class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "fallback [" in m:
            tags.append(m.split("[", 1)[1].split("]", 1)[0])
        elif m.startswith("DynaGraph opaque kernel "):
            opaque.append(m[len("DynaGraph opaque kernel "):].split(":", 1)[0])
lg.addHandler(_Grab())

# Signatures of every compiled kernel, used to prove this run really went through TMA.
sigs, srcs = [], []
from torch._inductor.runtime import triton_heuristics as th
_orig_init = th.CachingAutotuner.__init__
def _spy(self, *a, **kw):
    _orig_init(self, *a, **kw)
    s = (self.triton_meta or {}).get("signature") or {}
    sigs.append(((self.inductor_meta or {}).get("kernel_name", "?"), dict(s)))
    srcs.append(getattr(getattr(self, "fn", None), "src", "") or "")
th.CachingAutotuner.__init__ = _spy

from torch._inductor import cudagraph_trees as ct
n_rec = {"v": 0}
_orig_node = ct.CUDAGraphNode.__init__
def _rec(self, *a, **kw):
    n_rec["v"] += 1
    return _orig_node(self, *a, **kw)
ct.CUDAGraphNode.__init__ = _rec


@triton.jit
def dev_desc_kernel(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Descriptor built inside the kernel: the arguments are still pointers and integers, which DynaGraph understands.
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1],
                                   block_shape=[BLOCK_M, BLOCK_N])
    yd = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1],
                                   block_shape=[BLOCK_M, BLOCK_N])
    pid = tl.program_id(0)
    off_m = pid * BLOCK_M
    v = xd.load([off_m, 0])
    yd.store([off_m, 0], v * 2.0 + 1.0)


@triton.jit
def host_desc_kernel(xd, yd, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Descriptor built on the host and passed in: the signature says tensordesc<...>, not a pointer.
    pid = tl.program_id(0)
    v = xd.load([pid * BLOCK_M, 0])
    yd.store([pid * BLOCK_M, 0], v * 2.0 + 1.0)


def run_inductor():
    ic.triton.use_tensor_descriptor = True
    ic.triton.enable_persistent_tma_matmul = True
    ic.assume_aligned_inputs = True

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(256, 256, bias=False)
        def forward(self, x):
            h = self.lin(x)
            return (torch.relu(h) * 2.0 + 1.0).sum(-1)

    m = M().cuda().to(torch.float16)
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")
    worst = 0.0
    with torch.no_grad():
        for L in (256, 256, 512, 128, 384):
            x = torch.randn(L, 256, device="cuda", dtype=torch.float16)
            torch._dynamo.mark_dynamic(x, 0)
            out, ref = f(x), m(x)
            worst = max(worst, ((out - ref).abs().max() / ref.abs().max().clamp_min(1)).item())
    return worst


def _alloc(size, align, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def run_dev():
    triton.set_allocator(_alloc)
    BM, BN = 32, 256

    def op(x):
        y = torch.empty_like(x)
        M, N = x.shape
        dev_desc_kernel[(triton.cdiv(M, BM),)](x, y, M, N, BLOCK_M=BM, BLOCK_N=BN)
        return y

    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(256, 256)
        def forward(self, x):
            return op(torch.relu(self.lin(x))).sum(-1)

    m = Mod().cuda()
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")
    worst = 0.0
    with torch.no_grad():
        for L in (64, 64, 128, 32, 96):
            x = torch.randn(L, 256, device="cuda")
            torch._dynamo.mark_dynamic(x, 0)
            out, ref = f(x), m(x)
            worst = max(worst, ((out - ref).abs().max() / ref.abs().max().clamp_min(1)).item())
    return worst


def run_host():
    from triton.tools.tensor_descriptor import TensorDescriptor
    BM, BN = 32, 256

    def op(x):
        y = torch.empty_like(x)
        M, N = x.shape
        xd = TensorDescriptor.from_tensor(x, [BM, BN])
        yd = TensorDescriptor.from_tensor(y, [BM, BN])
        host_desc_kernel[(triton.cdiv(M, BM),)](xd, yd, BLOCK_M=BM, BLOCK_N=BN)
        return y

    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(256, 256)
        def forward(self, x):
            return op(torch.relu(self.lin(x))).sum(-1)

    m = Mod().cuda()
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")
    worst = 0.0
    with torch.no_grad():
        for L in (64, 64, 128, 32, 96):
            x = torch.randn(L, 256, device="cuda")
            torch._dynamo.mark_dynamic(x, 0)
            out, ref = f(x), m(x)
            worst = max(worst, ((out - ref).abs().max() / ref.abs().max().clamp_min(1)).item())
    return worst


err = None
if MODE == "host_undeclared":
    # The producer did not declare how the descriptor is built (a third-party wrapper, or a codegen path
    # whose declaration is not wired up yet). Even with the switch on it must fall back to an opaque site, never patch
    # something it cannot read as parameters -- this mode pins down that safety property.
    from torch.utils import _capture_tma
    _capture_tma.register = lambda key, descriptors: None
try:
    worst = {"inductor": run_inductor, "device": run_dev,
             "host": run_host, "host_undeclared": run_host}[MODE]()
except Exception as e:
    worst, err = float("nan"), f"{type(e).__name__}: {e}"

desc = [(k, nm, ty) for k, s in sigs for nm, ty in s.items()
        if isinstance(ty, str) and (ty == "nvTmaDesc" or ty.startswith("tensordesc<"))]
n_dev_desc = sum(src.count("make_tensor_descriptor") for src in srcs)
served = not err and worst < 1e-3 and n_rec["v"] == 0 and not tags
print(f"  mode {MODE}  compiled kernels {len(sigs)}  device-side descriptors {n_dev_desc}  descriptors in signatures {len(desc)}  "
      f"opaque {sorted(set(opaque)) or '-'}  records {n_rec['v']}  tags {sorted(set(tags)) or '-'}  "
      f"max rel diff {worst:.1e}")
for k, nm, ty in desc[:4]:
    print(f"    {k}  {nm}: {ty}")
if err:
    print(f"  exception {err[:200]}")

if MODE in ("inductor", "device"):
    # Descriptor built inside the kernel: arguments are still pointers and integers, so it should be served anyway.
    if n_dev_desc == 0:
        print("  inconclusive: this compile produced no device-side descriptor at all, so the probe proved nothing")
        sys.exit(2)
    ok = served
    print("  passed: with device-built descriptors the arguments are still pointers+integers, served by one graph" if ok else "  FAILED")
elif MODE in ("host", "host_undeclared"):
    # Host-built descriptor: one signature entry expands into several real parameters in the cubin,
    # reading parameter offsets by signature position is shifted wholesale, and the address and shape are baked into those 128 bytes.
    # Two routes, depending on the switch:
    #   off -- the table cannot read it, so this launch becomes its own site: captured into a child graph, re-harvested per shape,
    #          with the descriptor re-encoded by Triton's own launcher;
    #   on  -- the producer declared what the descriptor is built from, so it becomes a patchable argument;
    #          for each shape the 128 bytes are re-encoded and pushed into the exec, and the kernel stays in the main graph.
    # Both routes require the region to still be served by one graph with correct numerics.
    if not desc:
        print("  inconclusive: this compile produced no host-side descriptor argument, so the probe proved nothing")
        sys.exit(2)
    patched = ic.triton.dynagraph_tma_patch and MODE != "host_undeclared"
    route_ok = (not opaque) if patched else bool(opaque)
    ok = not err and worst < 1e-5 and n_rec["v"] == 0 and not tags and route_ok
    print(f"  passed: kernels with host-side descriptors go through {'parameter patching' if patched else 'an opaque site'}, "
          "the region is still served by one graph, numerics correct" if ok
          else f"  FAILED: did not go through {'parameter patching' if patched else 'an opaque site'}, was not served, or numerics are wrong")
else:
    print(f"  unknown mode {MODE}")
    sys.exit(2)
sys.exit(0 if ok else 1)
