#!/usr/bin/env python3
"""launch-boundness probe -- 100% CPU, meta tensors only, never touches CUDA.

Why not survey/runner.py --fake:
  that path still does model.to_empty(device="cuda") and Inductor CUDA codegen,
  and Inductor queries torch.cuda.get_device_properties / get_device_capability in
  codecache.py:348, utils.py:2186, codegen/triton.py:3014,4829, compile_fx.py:470 ...
  every one of those calls torch.cuda._lazy_init.  --fake avoids *allocating* device
  memory; it does not avoid *taking a CUDA context*.  So it is not usable on a box
  whose GPUs belong to someone else.

What this does instead: run the model under a TorchDispatchMode on **meta** tensors,
record bytes moved and FLOPs per aten op, group pointwise chains the way Inductor
would, then apply the machine's measured roofline to predict
    Tbar  = mean GPU kernel duration
    verdict: launch-bound iff  N*c_host > T_gpu + N*g_eager   <=>   Tbar < c_host - g_eager
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only

# --- machine constants, all measured on this box (see launchbound.py header) ---
BW, FLOPS, FLOPS_SMALL = 1800e9, 400e12, 100e12
T_EXEC_FLOOR, G_EAGER, G_GRAPH = 1.0e-6, 2.0e-6, 0.8e-6
C_HOST = 5.0e-6

VIEW_OPS = {
    "view", "_unsafe_view", "reshape", "permute", "transpose", "t", "expand",
    "squeeze", "unsqueeze", "slice", "select", "detach", "alias", "as_strided",
    "contiguous", "_to_copy", "lift_fresh", "split", "unbind", "narrow",
}
MATMUL_OPS = {"mm", "bmm", "addmm", "baddbmm", "matmul", "linear", "einsum",
              "convolution", "_scaled_dot_product_flash_attention",
              "_scaled_dot_product_efficient_attention", "scaled_dot_product_attention"}
REDUCE_OPS = {"sum", "mean", "max", "min", "amax", "amin", "var", "std", "prod",
              "softmax", "_softmax", "log_softmax", "_log_softmax", "native_layer_norm",
              "native_batch_norm", "_native_batch_norm_legit", "cumsum", "argmax",
              "index_add", "index_add_", "scatter_add", "scatter_add_", "index_select",
              "gather", "topk", "sort", "norm", "linalg_vector_norm"}

def _nbytes(t):
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0

def _flops(name, args, out):
    def sz(t): return tuple(t.shape) if isinstance(t, torch.Tensor) else ()
    try:
        if name in ("mm", "addmm"):
            a = args[-2] if name == "addmm" else args[0]
            b = args[-1]
            return 2 * a.shape[0] * a.shape[1] * b.shape[1]
        if name in ("bmm", "baddbmm"):
            a = args[-2] if name == "baddbmm" else args[0]
            b = args[-1]
            return 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
        if name == "convolution":
            x, w = args[0], args[1]
            o = out[0] if isinstance(out, (tuple, list)) else out
            k = 1
            for d in w.shape[2:]: k *= d
            return 2 * o.numel() * w.shape[1] * k
        if name.startswith("_scaled_dot_product"):
            q, k = args[0], args[1]
            B = 1
            for d in q.shape[:-2]: B *= d
            return 4 * B * q.shape[-2] * k.shape[-2] * q.shape[-1]
    except Exception:
        pass
    return 0.0

def t_kernel(b, f):
    return max(T_EXEC_FLOOR, b / BW, f / (FLOPS if f > 1e9 else FLOPS_SMALL))

class LaunchBoundProbe(TorchDispatchMode):
    def __init__(self):
        self.ops = []          # (name, bytes, flops, kind)
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func._schema.name.split("::")[-1].rstrip("_") if hasattr(func, "_schema") else str(func)
        if name not in VIEW_OPS:
            b = 0
            tree_map_only(torch.Tensor, lambda t: None, args)
            for a in args:
                b += _nbytes(a)
            for v in kwargs.values():
                b += _nbytes(v)
            outs = out if isinstance(out, (tuple, list)) else (out,)
            for o in outs:
                b += _nbytes(o)
            f = _flops(name, args, out)
            kind = "matmul" if name in MATMUL_OPS else ("reduce" if name in REDUCE_OPS else "pointwise")
            self.ops.append((name, b, f, kind))
        return out

    def report(self, label, c_host=C_HOST):
        # Inductor-like fusion: collapse maximal runs of pointwise ops into one kernel
        fused, run = [], None
        for name, b, f, kind in self.ops:
            if kind == "pointwise":
                if run is None: run = [name, b, f]
                else:
                    run[0] += "+" + name
                    run[1] += b * 0.35   # fused chain re-reads far less than the sum
                    run[2] += f
            else:
                if run: fused.append(tuple(run)); run = None
                fused.append((name, b, f))
        if run: fused.append(tuple(run))

        N_aten = len(self.ops)
        for tag, lst in (("unfused (eager aten)", [(n, b, f) for n, b, f, _ in self.ops]),
                         ("Inductor-fused est.", fused)):
            N = len(lst)
            T = sum(t_kernel(b, f) for _, b, f in lst)
            Tbar = T / max(N, 1)
            eager = max(N * c_host, T + N * G_EAGER)
            graph = T + N * G_GRAPH
            lb = N * c_host > T + N * G_EAGER
            print(f"  {label:26s} {tag:22s} N={N:6d}  Tbar={Tbar*1e6:8.2f}us  "
                  f"GPU={T*1e3:8.3f}ms  eager={eager*1e3:8.3f}ms  "
                  f"graph_speedup={eager/graph:5.2f}x  {'LAUNCH-BOUND' if lb else 'gpu-bound'}")
        return N_aten


if __name__ == "__main__":
    import torch.nn as nn
    print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
    print(f"  criterion: launch-bound iff Tbar < c_host - g_eager = "
          f"{(C_HOST-G_EAGER)*1e6:.1f} us  (c_host={C_HOST*1e6:.0f}us)")
    print()

    def probe(label, build):
        with torch.device("meta"):
            m, xs = build()
        p = LaunchBoundProbe()
        with p:
            m(*xs)
        p.report(label)

    # 1. a transformer block stack at two sizes -- the "big kernel" archetype
    def tf(d, L, n):
        layer = nn.TransformerEncoderLayer(d, 8, 4*d, batch_first=True, norm_first=True)
        enc = nn.TransformerEncoder(layer, n)
        return enc, (torch.randn(1, L, d),)
    probe("transformer d=1024 L=2048", lambda: tf(1024, 2048, 12))
    probe("transformer d=256  L=32",   lambda: tf(256, 32, 12))

    # 2. deep stack of tiny ops -- the "many small kernels" archetype
    class Tiny(nn.Module):
        def __init__(s, n, d):
            super().__init__()
            s.ls = nn.ModuleList([nn.Linear(d, d) for _ in range(n)])
            s.ns = nn.ModuleList([nn.LayerNorm(d) for _ in range(n)])
        def forward(s, x):
            for l, nm in zip(s.ls, s.ns):
                x = torch.sigmoid(nm(l(x))) * x + x
            return x
    probe("tiny-op stack 200x d=128", lambda: (Tiny(200, 128), (torch.randn(300, 128),)))
    probe("tiny-op stack 200x d=4096", lambda: (Tiny(200, 4096), (torch.randn(8192, 4096),)))

    # 3. GraphSAGE-ish: few kernels, medium tensors
    class Sage(nn.Module):
        def __init__(s, cin, ch, nl):
            super().__init__()
            s.l = nn.ModuleList([nn.Linear(cin if i == 0 else ch, ch) for i in range(nl)])
            s.r = nn.ModuleList([nn.Linear(cin if i == 0 else ch, ch) for i in range(nl)])
        def forward(s, x, idx, nout):
            for li, ri in zip(s.l, s.r):
                agg = torch.zeros(nout, x.shape[1], device=x.device, dtype=x.dtype)
                agg = agg.index_add(0, idx[:x.shape[0]].clamp(max=nout-1), x)
                x = torch.relu(li(agg) + ri(agg))
            return x
    probe("GraphSAGE 3L B=1024", lambda: (Sage(128, 256, 3),
          (torch.randn(30000, 128), torch.zeros(150000, dtype=torch.long), 29000)))
