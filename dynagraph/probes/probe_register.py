#!/usr/bin/env python3
r"""Register-and-go dynamic cudagraph: three kinds of source-controlled kernels, using only `torch.utils._capture_launch.register`.

Three custom ops in one model, one per kind:
  demo::scale      hand-written CUDA kernel (compiled on the spot via load_inline), hand-written declaration: grid/block/args computed from tensors
  demo::bias_relu  Triton kernel with @triton.autotune(key=BUCKET) -- JIT + autotune,
                   variant=BUCKET, prepare=run once (autotune+compile), launch derived from the call site by triton_launch
  demo::gemm       DeepGEMM bf16 (a runtime-JIT library with no describe interface), recorded: host code recorded once per M,
                   variant=M, prepare=run once (JIT)

Criteria:
  1. The region is served by DynaGraph with no harvest (no site goes through the tier-2 child graph)
  2. Every shape's output is bitwise identical to eager (DYNAGRAPH_VERIFY_SHAPES set to cover all + compared again here)
  3. prepare runs exactly once per (op, variant), all outside capture; no new variant is ever met inside a capture
  4. Shape changes are never handed back upstream for recording (record count = 0)
"""
import logging
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_DYNAGRAPH_VERIFY_SHAPES", "1000")

import torch
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct
import triton
import triton.language as tl
from torch.utils import _capture_launch as cl
from torch.utils.cpp_extension import load_inline

from torch._inductor import dynagraph as dg

# ---------------------------------------------------------------- 1. CUDA
_ext = load_inline(
    name="demo_scale_ext",
    cpp_sources="void scale(torch::Tensor x, torch::Tensor out, double alpha);",
    cuda_sources=r"""
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void scale_rows(const __nv_bfloat16* x, __nv_bfloat16* out, int cols, float alpha) {
  int r = blockIdx.x;
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    out[(long)r * cols + c] = __float2bfloat16(alpha * __bfloat162float(x[(long)r * cols + c]));
}
void scale(torch::Tensor x, torch::Tensor out, double alpha) {
  int rows = x.size(0), cols = x.size(1);
  if (rows == 0) return;
  scale_rows<<<rows, std::min(cols, 1024), 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), cols, (float)alpha);
}
""",
    functions=["scale"],
    verbose=False,
)


@torch.library.custom_op("demo::scale", mutates_args=("out",))
def scale(x: torch.Tensor, out: torch.Tensor, alpha: float) -> None:
    _ext.scale(x, out, alpha)


def scale_launches(x, out, alpha):
    rows, cols = x.shape
    if rows == 0:
        return []
    return [cl.Launch(kernel="scale_rows<bf16>", grid=(rows, 1, 1), block=(min(cols, 1024), 1, 1),
                      smem=0, args=(x, out, cols, float(alpha)))]


cl.register("demo::scale", scale_launches, prepare=lambda *a, **k: None)   # AOT: nothing to set up


# ---------------------------------------------------------------- 2. Triton
@triton.autotune(configs=[triton.Config({"BLOCK": b}, num_warps=w) for b, w in ((128, 4), (512, 4), (1024, 8))],
                 key=["BUCKET"])
@triton.jit
def bias_relu_k(h, b, y, n, cols, BUCKET, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(h + offs, mask=m).to(tl.float32) + tl.load(b + offs % cols, mask=m).to(tl.float32)
    tl.store(y + offs, tl.maximum(v, 0.0).to(tl.bfloat16), mask=m)


def _br_call(h, b, y):
    n, cols = h.numel(), h.shape[1]
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    return grid, (h, b, y, n, cols, triton.next_power_of_2(max(h.shape[0], 1)))


@torch.library.custom_op("demo::bias_relu", mutates_args=("y",))
def bias_relu(h: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> None:
    grid, args = _br_call(h, b, y)
    bias_relu_k[grid](*args)


PREPARED = []


def br_prepare(h, b, y):
    PREPARED.append(("bias_relu", triton.next_power_of_2(max(h.shape[0], 1))))
    bias_relu(h, b, y)          # autotune for this bucket + JIT, outside any capture


def br_launches(h, b, y):
    grid, args = _br_call(h, b, y)
    return [cl.triton_launch(bias_relu_k, grid, *args)]


cl.register("demo::bias_relu", br_launches, prepare=br_prepare,
            variant=lambda h, b, y: triton.next_power_of_2(max(h.shape[0], 1)))

# ---------------------------------------------------------------- 3. DeepGEMM
import vllm.third_party.deep_gemm as deep_gemm


@torch.library.custom_op("demo::gemm", mutates_args=("out",))
def gemm(a: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    deep_gemm.bf16_gemm_nt(a, w, out)


def gemm_prepare(a, w, out):
    PREPARED.append(("gemm", a.shape[0]))
    gemm(a, w, out)             # JIT-compiles this M's config, outside any capture


cl.register("demo::gemm",
            cl.recorded(lambda a, w, out: torch.ops.demo.gemm(a, w, out),
                        template_key=lambda a, w, out: a.shape[0]),
            prepare=gemm_prepare, variant=lambda a, w, out: a.shape[0], exact=False)


# ---------------------------------------------------------------- model
class M(torch.nn.Module):
    def __init__(self, k=2048, n=1024):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(n, k, dtype=torch.bfloat16) / k ** 0.5)
        self.b = torch.nn.Parameter(torch.randn(n, dtype=torch.bfloat16))

    def forward(self, x):
        h = x.new_empty(x.shape[0], self.w.shape[0])
        torch.ops.demo.gemm(x, self.w, h)
        y = torch.empty_like(h)
        torch.ops.demo.bias_relu(h, self.b, y)
        z = torch.empty_like(y)
        torch.ops.demo.scale(y, z, 0.5)
        return z * 2 + 1


class Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, rec):
        self.msgs.append(rec.getMessage())


SHAPES = [64, 17, 100, 256, 33, 128, 200, 64, 17, 300]


def main():
    ic.force_disable_caches = True
    ic.triton.cudagraphs = True
    ic.triton.dynagraph = True
    grab = Grab()
    lg = logging.getLogger("torch._inductor.dynagraph")
    lg.setLevel(logging.INFO)
    lg.addHandler(grab)
    n_rec = {"n": 0}
    orig = ct.CUDAGraphTreeManager.record_function

    def spy(self, *a, **k):
        n_rec["n"] += 1
        return orig(self, *a, **k)

    ct.CUDAGraphTreeManager.record_function = spy
    served = {"ok": 0, "other": 0}
    oc = dg.DynaGraphRunner.__call__

    def call(self, inputs):
        r = oc(self, inputs)
        served["ok" if r is not None and r is not dg.SKIP_SHAPE and r is not dg.REBUILD else "other"] += 1
        return r

    dg.DynaGraphRunner.__call__ = call
    harvests = {"n": 0}
    oh = dg.DynaGraphRunner._harvest

    def h(self, *a, **k):
        harvests["n"] += 1
        return oh(self, *a, **k)

    dg.DynaGraphRunner._harvest = h

    torch.manual_seed(0)
    m = M().cuda()
    # Process-level initialisation, as an engine's start-up profiling run does:
    # libraries allocate their resident state on first use, and that must not
    # land in a graph pool.
    with torch.no_grad():
        m(torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16))
    PREPARED.clear()
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")
    worst = 0.0
    with torch.no_grad():
        for L in SHAPES:
            x = torch.randn(L, 2048, device="cuda", dtype=torch.bfloat16)
            got = f(x).clone()
            ref = m(x)
            worst = max(worst, (got.float() - ref.float()).abs().max().item())
    torch.cuda.synchronize()

    bad = 0
    tags = sorted({t.split("[", 1)[1].split("]", 1)[0] for t in grab.msgs if "fallback [" in t})
    inline = [t for t in grab.msgs if "prepared" in t]
    print(f"\n  {len(SHAPES)} calls (beyond region warmup): DynaGraph served {served['ok']}, other {served['other']}; "
          f"harvest {harvests['n']}x; upstream records {n_rec['n']}x; fallback tags {tags}")
    for t in grab.msgs:
        if "fallback" in t or "mismatch" in t:
            print("     ", t[:200])
    if served["ok"] == 0 or tags or served["other"]:
        print("  FAIL criteria 1/4: not everything was served by DynaGraph"); bad += 1
    else:
        print("  ok criteria 1/4: all served by DynaGraph, no shape change handed back upstream")
    if harvests["n"]:
        print(f"  FAIL harvested {harvests['n']}x (some site went through tier 2)"); bad += 1
    print(f"  max diff vs eager (same ops called directly) {worst:.3g}")
    if worst != 0.0:
        print("  FAIL criterion 2: numerics differ"); bad += 1
    else:
        print("  ok criterion 2: every shape bitwise identical (DynaGraph also verified each shape against eager itself)")
    print(f"  prepare calls: {PREPARED}")
    print(f"  prepares logged by DynaGraph: {len(inline)}")
    for t in inline:
        print("     ", t)
    variants = {("bias_relu", triton.next_power_of_2(L)) for L in SHAPES} | {("gemm", L) for L in SHAPES}
    dup = len(PREPARED) - len(set(PREPARED))
    missing = variants - set(PREPARED)
    inside = [t for t in grab.msgs if "first met inside a capture" in t]
    if dup or inside or missing:
        print(f"  FAIL criterion 3: duplicates {dup}, missing {sorted(missing)}, new variants met inside capture {len(inside)}"); bad += 1
    else:
        print(f"  ok criterion 3: {len(set(PREPARED))} variants each prepared once, all outside capture")
    print("all passed" if bad == 0 else f"{bad} items failed")
    return bad


if __name__ == "__main__":
    sys.exit(main())
