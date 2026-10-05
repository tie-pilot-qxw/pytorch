#!/usr/bin/env python3
"""symint tier 2: the value of `.item()` stays in device memory, and a planner node in the graph reads it and changes the grid and scalars of later nodes.

    python probe_unbacked_device.py [--only off|on] [--cases a,b]

Each case is compiled with `capture_scalar_outputs=True` (Inductor does not split the graph at `.item()`,
`dynagraph_unbacked="device"`), shape stream x 2 passes. Criteria: with DynaGraph on, 0 recordings, no fallback tags,
and every output of every shape in every pass matches plain torch.compile (no cudagraph).
"""
import argparse, logging, os, sys
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo, torch._dynamo.config as dcfg, torch._inductor.config as ic
import triton, triton.language as tl

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--cases", default="")
ap.add_argument("--time", action="store_true", help="also time one fixed shape per mode (exclusive card)")
a = ap.parse_args()

logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.INFO)
tags: list[str] = []
served = {"n": 0}
class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "fallback [" in m:
            tags.append(m.split("[", 1)[1].split("]", 1)[0] + (": " + m.split("]: ", 1)[1][:90] if "]: " in m else ""))
        if "captured graph" in m:
            served["n"] += 1
lg.addHandler(_Grab())

D = 64
SHAPES = [64, 200, 128, 333, 96]


class ItemSliceTriton(torch.nn.Module):
    """count -> slice -> pointwise + reduction, all Triton; output size is the count."""
    def forward(self, x):
        n = int((x[:, 0] > 0).sum())
        y = x[:n]
        return (y * 2 + 1).sum(-1)


class ItemSliceFixedOut(torch.nn.Module):
    """count -> slice -> reduction over the unbacked rows; output size is fixed."""
    def forward(self, x):
        n = int((x[:, 0] > 0).sum())
        return x[:n].sum(0) + x[0]


class ItemSliceLinear(torch.nn.Module):
    """count -> slice -> Linear (an extern, run at the bound) -> reduction."""
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(D, D)
    def forward(self, x):
        n = int((x[:, 0] > 0).sum())
        return self.a(x[:n]).sum(-1)


class ItemBoolScale(torch.nn.Module):
    """a bool read as a scalar; no size depends on it."""
    def forward(self, x):
        flag = bool((x.sum() > 0).item())
        return x * (2 if flag else 3) + 1


class TwoItems(torch.nn.Module):
    """two counts, two slices, two planner points."""
    def forward(self, x):
        n = int((x[:, 0] > 0).sum())
        m = int((x[:, 1] > 0).sum())
        return (x[:n] * 2).sum(-1).sum() + (x[:m] + 1).sum(-1).sum()


def _t_sum(x):
    return x.sin().sum(0)


def _f_sum(x):
    return x.cos().sum(0)


def _t_pw(x):
    return x * 2 + 1


def _f_pw(x):
    return x.sin() - 1


def _t_two(x):
    return x * 2, x + 1


def _f_two(x):
    return x.sin(), x.cos()


def _f_red(x):
    return x.sin().sum(-1, keepdim=True) * x


class CondPointwise(torch.nn.Module):
    """torch.cond on a data-dependent predicate; two one-kernel branches; a reduction after."""
    @staticmethod
    def pred(x):
        return bool((x[:, 0] > 0).sum() > (x.shape[0] // 2))
    def forward(self, x):
        pred = (x[:, 0] > 0).sum() > (x.shape[0] // 2)
        return torch.cond(pred, _t_pw, _f_pw, (x,)).sum(-1)


class CondTwoOutputs(torch.nn.Module):
    """branches returning two tensors each."""
    @staticmethod
    def pred(x):
        return bool(x.sum() > 0)
    def forward(self, x):
        pred = x.sum() > 0
        a, b = torch.cond(pred, _t_two, _f_two, (x,))
        return (a * b).sum(-1)


class CondUnevenBranches(torch.nn.Module):
    """one branch is one kernel, the other two (a reduction feeding a pointwise)."""
    @staticmethod
    def pred(x):
        return bool(x[:, 2].sum() > 0)
    def forward(self, x):
        pred = x[:, 2].sum() > 0
        return torch.cond(pred, _t_pw, _f_red, (x,)).sum(-1)


class CondThenItem(torch.nn.Module):
    """a cond, then a count and a slice of its result."""
    @staticmethod
    def pred(x):
        return bool(x.sum() > 0)
    def forward(self, x):
        pred = x.sum() > 0
        y = torch.cond(pred, _t_pw, _f_pw, (x,))
        n = int((y[:, 0] > 0).sum())
        return y[:n].sum(0)


@triton.jit
def _scale_kernel(x_ptr, y_ptr, n, s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * s, mask=mask)


def _scale(x, s):
    y = torch.empty_like(x)
    n = x.numel()
    _scale_kernel[(triton.cdiv(n, 256),)](x, y, n, s, BLOCK=256)
    return y


class UserKernelItem(torch.nn.Module):
    """a user @triton.jit kernel and a count in one region: the user kernel must
    take the static launcher (it needs a device handle for the planner)."""
    def forward(self, x):
        y = _scale(x, 2.0)
        n = int((x[:, 0] > 0).sum())
        return y[:n].sum(0)


class CondOverSlice(torch.nn.Module):
    """Slice by a count first, then run a cond over this u-sized tensor: the branches' inputs and allocations both carry u.
    The branch kernels are now plain nodes in the graph, so the planner can change their grids by u."""
    @staticmethod
    def pred(x):
        return bool(x[:, 2].sum() > 0)
    def forward(self, x):
        n = int((x[:, 0] > 0).sum())
        y = x[:n] * 2.0
        pred = x[:, 2].sum() > 0
        return torch.cond(pred, _t_sum, _f_sum, (y,))


class ItemThenCond(torch.nn.Module):
    """a count, a slice sized by it, then a cond over the slice: the branches are
    sized by u0, which the planner cannot patch inside a child graph, so the
    device path must refuse this region (a tag, never a wrong number)."""
    @staticmethod
    def pred(x):
        n = int((x[:, 0] > 0).sum())
        return bool(x[:n].sum() > 0)
    def forward(self, x):
        n = int((x[:, 0] > 0).sum())
        y = x[:n]
        pred = y.sum() > 0
        return torch.cond(pred, _t_pw, _f_pw, (y,)).sum(0)


# Must be refused cleanly (numerics must still be correct, and the tag must be exactly this one): boundaries not wired up yet
REFUSE = {
    "item_then_cond": "unmodelled",       # the partition receives elements of a tuple input
    "cond_over_slice": "unbacked-source",  # the selector reads a tensor passed in from the previous partition
}

CASES = [
    ("cond_pointwise", CondPointwise),
    ("cond_two_outputs", CondTwoOutputs),
    ("cond_uneven", CondUnevenBranches),
    ("cond_then_item", CondThenItem),
    ("item_then_cond", ItemThenCond),
    ("cond_over_slice", CondOverSlice),
    ("user_kernel_item", UserKernelItem),
    ("item_slice_triton", ItemSliceTriton),
    ("item_slice_fixed_out", ItemSliceFixedOut),
    ("item_slice_linear", ItemSliceLinear),
    ("item_bool_scale", ItemBoolScale),
    ("two_items", TwoItems),
]


def batch(L, step):
    g = torch.Generator(device="cuda"); g.manual_seed(100 + step)
    return torch.randn(L, D, device="cuda", generator=g)


def run(cls, mode):
    torch._dynamo.reset()
    dcfg.capture_scalar_outputs = True
    ic.triton.dynagraph = mode == "on"
    ic.triton.dynagraph_unbacked = "device"
    ic.triton.dynagraph_extern_child = True
    ic.force_disable_caches = True
    tags.clear()
    served["n"] = 0
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
                y = f(batch(L, step))
                torch.cuda.synchronize()
                outs.append(y.float().clone())
        t_call = t_loop = 0.0
        if a.time:
            import time
            x = batch(200, 7)
            with torch.no_grad():
                for _ in range(20):
                    f(x)
                torch.cuda.synchronize()
                ts = []
                for _ in range(100):
                    t0 = time.perf_counter(); f(x); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
                t_call = sorted(ts)[50] * 1e6
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(200):
                    f(x)
                torch.cuda.synchronize(); t_loop = (time.perf_counter() - t0) / 200 * 1e6
        return n_rec["v"], outs, list(tags), None, (t_call, t_loop, served["n"])
    except Exception as e:
        return n_rec["v"], [], list(tags), f"{type(e).__name__}: {str(e).splitlines()[0][:120]}", (0.0, 0.0, served["n"])
    finally:
        ct.CUDAGraphNode.__init__ = orig


def main() -> int:
    bad = 0
    want = set(a.cases.split(",")) if a.cases else None
    for name, cls in CASES:
        if want and name not in want:
            continue
        ref = run(cls, "plain")
        if hasattr(cls, "pred"):
            taken = [cls.pred(batch(L, step)) for step, L in enumerate(SHAPES * 2)]
            print(f"  {name:<22} branch taken: true {sum(taken)} / false {len(taken) - sum(taken)}")
        res = {}
        for mode in ("off", "on"):
            if a.only and mode != a.only:
                continue
            res[mode] = run(cls, mode)
        line = f"  {name:<22}"
        ok = True
        if ref[3] is not None:
            # Upstream cannot compile this case even without cudagraphs, so
            # numerics are out of reach; DynaGraph must still refuse cleanly.
            tg = res.get("on", (0, [], [], None, ()))[2]
            line += f"  upstream plain also fails to compile ({ref[3]}), checking only DG=on tags {sorted(set(t.split(':')[0] for t in tg)) or '-'}"
            ok = "on" not in res or not any(t.startswith("exception") for t in tg)
            print(line + ("  ok" if ok else "  FAIL"), flush=True)
            bad += not ok
            continue
        for mode, r in res.items():
            n, outs, tg, err, tm = r
            if err:
                line += f"  DG={mode} ERR {err}"; ok = False; continue
            worst = 0.0
            for y, z in zip(ref[1], outs):
                if y.shape != z.shape:
                    worst = float("inf"); break
                worst = max(worst, (y - z).abs().max().item() / max(y.abs().max().item(), 1e-6))
            seen = {}
            for t in tg: seen[t.split(":")[0]] = seen.get(t.split(":")[0], 0) + 1
            line += f"  DG={mode} recordings {n:<3} served {tm[2]} rel diff {worst:.1e} tags {seen or '-'}"
            if a.time:
                line += f" single {tm[0]:.0f}us back-to-back {tm[1]:.0f}us"
            want_tag = REFUSE.get(name)
            if mode == "on" and want_tag:
                # must refuse with this tag, and the result must still be correct
                if worst > 1e-4 or set(seen) != {want_tag}:
                    ok = False
            elif mode == "on" and (n != 0 or seen or worst > 1e-4 or tm[2] == 0):
                ok = False
        if a.time:
            line += f"  plain single {ref[4][0]:.0f}us back-to-back {ref[4][1]:.0f}us"
        print(line + ("  ok" if ok else "  FAIL"), flush=True)
        bad += not ok
    print("  all passed" if not bad else f"  {bad} item(s) failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
