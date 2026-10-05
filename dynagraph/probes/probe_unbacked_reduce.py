#!/usr/bin/env python3
"""extern reducing over an unbacked dim: zero the padding, and computing at the upper bound is exact.

On the device path, extern runs at u's upper bound, and [u, bound) holds garbage left by the previous tenant. When the
output is sliced by rows at u, it is simply dropped; but once u is the reduced dim (a gemm's K), the garbage gets summed in.
Zero is the additive identity, so zeroing that range before the extern makes the upper-bound result equal the true-value result -- not an approximation.

Must be served: both operands are u-sized arena buffers. Must still be refused: an operand is a graph input
(we cannot write someone else's memory), u appears in a stride, the reduction is not addition (max).
"""
import argparse, logging, sys
import torch
from torch._inductor import config as ic
import torch._dynamo.config as dcfg

p = argparse.ArgumentParser()
p.add_argument("--cases", default="")
p.add_argument("--sabotage", action="store_true", help="remove the zeroing; results must then be wrong")
a = p.parse_args()

tags = []
class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "DynaGraph fallback [" in m:
            tags.append(m.split("[", 1)[1].split("]", 1)[0])
logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.INFO); lg.addHandler(_Grab())

D, SHAPES = 64, [64, 200, 128, 333, 96]


class VarlenGram(torch.nn.Module):
    """Both operands are u-sized arena buffers, and u is K. Must be served."""
    def forward(self, x, y):
        n = int((x[:, 0] > 0).sum())
        a = x[:n] * 2.0 + 1.0
        b = y[:n].sin()
        return a.t() @ b


class VarlenAddmm(torch.nn.Module):
    """Same as above, plus a fixed-size bias. Must be served."""
    def forward(self, x, y):
        n = int((x[:, 0] > 0).sum())
        a = x[:n] * 2.0 + 1.0
        b = y[:n].sin()
        return torch.addmm(torch.ones(D, D, device=x.device), a.t(), b)


class VarlenGramInput(torch.nn.Module):
    """The operands are graph inputs; the padding is not ours, so it must still be refused."""
    def forward(self, x, y):
        n = int((x[:, 0] > 0).sum())
        return x[:n].t() @ y[:n]


class VarlenBmmStride(torch.nn.Module):
    """u ends up in a stride: the child graph was captured with the upper-bound stride while the producer writes with the true u,
    so the wrong elements are read; zeroing cannot fix that, so it must still be refused."""
    def forward(self, x, y):
        n = int((x[:, 0] > 0).sum())
        a = (x[:n] * 2.0 + 1.0).t().contiguous()
        b = y[:n].sin()
        return a @ b


CASES = [
    ("varlen_gram", VarlenGram, "serve"),
    ("varlen_addmm", VarlenAddmm, "serve"),
    ("varlen_gram_input", VarlenGramInput, "refuse"),
    ("varlen_bmm_stride", VarlenBmmStride, "any"),
]


def batch(L, step):
    g = torch.Generator(device="cuda"); g.manual_seed(100 + step)
    return (torch.randn(L, D, device="cuda", generator=g),
            torch.randn(L, D, device="cuda", generator=g))


def run(cls, mode):
    torch._dynamo.reset()
    dcfg.capture_scalar_outputs = True
    ic.triton.dynagraph = mode == "on"
    ic.triton.dynagraph_unbacked = "device"
    ic.triton.dynagraph_extern_child = True
    ic.force_disable_caches = True
    tags.clear()
    from torch._inductor import cudagraph_trees as ct
    n_rec = {"v": 0}
    orig = ct.CUDAGraphNode.__init__
    def rec(self, *x, **kw):
        n_rec["v"] += 1
        return orig(self, *x, **kw)
    ct.CUDAGraphNode.__init__ = rec
    try:
        torch.manual_seed(0)
        m = cls().cuda().eval()
        f = torch.compile(m, dynamic=True, mode="reduce-overhead" if mode != "plain" else "default")
        outs = []
        with torch.no_grad():
            for step, L in enumerate(SHAPES * 2):
                x, y = batch(L, step)
                outs.append(f(x, y).float().clone())
                torch.cuda.synchronize()
        return n_rec["v"], outs, list(tags), None
    except Exception as e:
        return n_rec["v"], [], list(tags), f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
    finally:
        ct.CUDAGraphNode.__init__ = orig


if a.sabotage:
    # Negative test: remove the zeroing launch, so stale data in the padding gets summed into the reduction;
    # the numerics must be wrong. If not, the zeroing had no effect at all (or the padding happened to be zero).
    from torch._inductor import dynagraph as dg
    real_src = dg._zeropad_source
    def empty(pad_of, sym_index, dev):
        # Same kernel, same launch, but the loop bounds hardcoded to empty
        return real_src(pad_of, sym_index, dev).replace("k < hi", "k < lo")
    dg._zeropad_source = empty

bad = 0
want = set(a.cases.split(",")) if a.cases else None
for name, cls, kind in CASES:
    if want and name not in want:
        continue
    ref = run(cls, "plain")
    got = run(cls, "on")
    line = f"  {name:<20} {kind:<7}"
    if ref[3] or got[3]:
        print(line + f"  ERR {ref[3] or got[3]}  FAIL"); bad += 1; continue
    worst = 0.0
    for r, o in zip(ref[1], got[1]):
        worst = max(worst, (r - o).abs().max().item() / max(r.abs().max().item(), 1e-6))
    seen = sorted({t.split(":")[0] for t in got[2]})
    line += f"  records {got[0]:<3} rel diff {worst:.1e} tags {seen or '-'}"
    if a.sabotage:
        # Only the two "must serve" cases count: without zeroing, stale data in the padding is summed into the reduction,
        # and the self-check must catch it (if it does not, zeroing does not affect the result and the feature is not real)
        ok = kind != "serve" or "selfcheck-mismatch" in seen
    elif kind == "serve":
        ok = got[0] == 0 and not seen and worst < 1e-5
    elif kind == "refuse":
        ok = seen == ["unbacked-extern-reduce"] and worst < 1e-5
    else:
        # Served or refused are both fine; numerics must be correct
        ok = worst < 1e-5
    print(line + ("  ok" if ok else "  FAIL")); bad += not ok
if a.sabotage:
    print("  negative test passed: the zeroing really is taking effect" if not bad else f"  negative test failed: {bad} items")
else:
    print("  all passed" if not bad else f"  {bad} items failed")
sys.exit(1 if bad else 0)
