#!/usr/bin/env python3
"""A user kernel mutates its own argument in place, and the argument names do not follow Inductor's convention.

DynaGraph used to decide "which argument does this kernel write" by name prefix only (`in_out_ptr` /
`out_ptr`), which is Inductor's convention for its own generated kernels; user kernels can name things anything,
and here it is deliberately called `dst`.

Note what this probe tests: whether the path **a user kernel mutating an arena intermediate buffer in place** works at all
(the two earlier user-kernel probes only write into freshly allocated outputs, so this path was never touched). It does **not** cover
the `mutated_arg_names` criterion -- `written_scan` only records written **region inputs**, and intermediate buffers are not in
`argv`, so this probe still passes with that criterion removed (verified). What truly can only be identified via `mutated_arg_names`
is a user kernel mutating in place **a buffer passed in from the previous partition**, which has no test case yet.
"""
import logging
import sys

import torch
import triton
import triton.language as tl
from torch._inductor import config as ic

tags = []


class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "DynaGraph fallback [" in m:
            tags.append(m.split("[", 1)[1].split("]", 1)[0])


logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph")
lg.setLevel(logging.INFO)
lg.addHandler(_Grab())

ic.triton.dynagraph = True
ic.force_disable_caches = True


@triton.jit
def add_inplace(dst, src, n, BLOCK: tl.constexpr):
    # Deliberately not named in_out_ptr0: only mutated_arg_names can tell that dst is written
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.load(dst + offs, mask=mask) + tl.load(src + offs, mask=mask), mask=mask)


class M(torch.nn.Module):
    def forward(self, x):
        a = x * 2.0
        b = x.sin()
        n = a.numel()
        add_inplace[(triton.cdiv(n, 256),)](a, b, n, BLOCK=256)
        return torch.relu(a).sum(-1)


n_rec = {"v": 0}
from torch._inductor import cudagraph_trees as ct  # noqa: E402

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
        g = torch.Generator(device="cuda")
        g.manual_seed(step)
        x = torch.randn(L, 96, device="cuda", generator=g)
        torch._dynamo.mark_dynamic(x, 0)
        ref = m(x)
        out = f(x)
        torch.cuda.synchronize()
        worst = max(worst, ((out - ref).abs().max() / ref.abs().max().clamp_min(1)).item())

seen = sorted(set(tags))
print(f"  records {n_rec['v']}  tags {seen or '-'}  max rel diff {worst:.1e}")
ok = worst < 1e-6 and n_rec["v"] == 0 and not seen
print("  all passed: the path of a user kernel mutating an intermediate buffer in place works" if ok else "  FAILED")
sys.exit(0 if ok else 1)
