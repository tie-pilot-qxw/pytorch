#!/usr/bin/env python3
"""The implicit premise of record-at-max: every buffer's size is monotonically non-decreasing in the symint.

"Record at the maximum of the range, and smaller shapes naturally fit" -- this only holds when sizes are monotonic.
If even one buffer is larger at a smaller M, it crosses the boundary fixed at capture time
and tramples its neighbor, and it is a **silent memory corruption**, with no error.

Whether we need to reimplement a cache allocator on the GPU depends on whether this premise actually breaks.
Measure it directly here: pull every `empty_strided_cuda(...)` size/stride expression out of the wrapper,
evaluate them over a series of M, and check whether the value at max really dominates all smaller M.

CPU only; just parses source + evaluates expressions (does not run the model).
"""
from __future__ import annotations

import ast
import os
import re
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")


def split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip()); cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return out


def find_allocs(src: str):
    """Return [(variable name, list of size expressions, list of stride expressions)]."""
    out = []
    for m in re.finditer(r"(\w+)\s*=\s*empty_strided_cuda\(", src):
        name, i = m.group(1), m.end()
        depth, j = 1, i
        while j < len(src) and depth:
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
            j += 1
        args = split_top(src[i:j - 1])
        if len(args) < 2:
            continue
        sz = split_top(args[0].strip()[1:-1]) if args[0].strip().startswith("(") else []
        st = split_top(args[1].strip()[1:-1]) if args[1].strip().startswith("(") else []
        out.append((name, sz, st))
    return out


def ev(expr: str, env: dict[str, int]):
    """Safe evaluation: only arithmetic and known symbols are allowed."""
    try:
        node = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    def go(n):
        if isinstance(n, ast.Expression):
            return go(n.body)
        if isinstance(n, ast.Constant):
            return int(n.value)
        if isinstance(n, ast.Name):
            return env.get(n.id)
        if isinstance(n, ast.BinOp):
            a, b = go(n.left), go(n.right)
            if a is None or b is None:
                return None
            o = n.op
            if isinstance(o, ast.Add): return a + b
            if isinstance(o, ast.Sub): return a - b
            if isinstance(o, ast.Mult): return a * b
            if isinstance(o, ast.FloorDiv): return a // b
            if isinstance(o, ast.Div): return a // b
            if isinstance(o, ast.Mod): return a % b
            return None
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = go(n.operand); return None if v is None else -v
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and \
                n.func.id in ("min", "max"):
            vs = [go(a) for a in n.args]
            return None if any(v is None for v in vs) else \
                (min(vs) if n.func.id == "min" else max(vs))
        return None

    return go(node)


def analyse(src: str, symbols: list[str], mmax: int):
    allocs = find_allocs(src)
    print(f"  {len(allocs)} empty_strided_cuda allocations in the wrapper")
    Ms = [mmax] + [m for m in (mmax // 2, mmax // 3, 777, 512, 333, 128, 64,
                               17, 8, 7, 3, 2, 1) if 0 < m <= mmax]
    violations = []
    for name, sz, st in allocs:
        rows = []
        for M in Ms:
            env = {s: M for s in symbols}
            sizes = [ev(e, env) for e in sz]
            strides = [ev(e, env) for e in st]
            if any(v is None for v in sizes) or any(v is None for v in strides):
                rows.append((M, None, None)); continue
            # span actually occupied = 1 + sum((size_i - 1) * stride_i), in elements
            span = 1 + sum((s_ - 1) * t_ for s_, t_ in zip(sizes, strides))
            rows.append((M, tuple(sizes), span))
        ok_rows = [r for r in rows if r[2] is not None]
        if not ok_rows:
            print(f"    {name:<8} cannot parse expressions, skipping: size={sz} stride={st}")
            continue
        at_max = next(r[2] for r in ok_rows if r[0] == mmax)
        worst = max(ok_rows, key=lambda r: r[2])
        flag = ""
        if worst[2] > at_max:
            violations.append((name, mmax, at_max, worst[0], worst[2]))
            flag = f"   <- **larger at M={worst[0]} ({worst[2]} > {at_max})**"
        print(f"    {name:<8} at M={mmax} takes {at_max:>8} elements; "
              f"max over the range {worst[2]:>8} (M={worst[0]}){flag}")
        if any(t is None for t in (r[2] for r in rows)):
            print(f"      (parsing failed for some M)")
    return violations


def main():
    import torch
    import torch._inductor.config as ic
    from torch._inductor import codecache

    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1
    ic.force_disable_caches = True
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"

    MMAX, D = 1024, 256

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(D, D)
            self.l2 = torch.nn.Linear(D, D)

        def forward(self, x):
            h = torch.relu(self.l1(x))
            h = h - h.mean(dim=-1, keepdim=True)
            h = torch.softmax(h, dim=-1)
            h = torch.relu(self.l2(h))
            # mix several reduction shapes to force out as many different buffer layouts as possible
            return h * h.sum(dim=0, keepdim=True) + h.amax(dim=1, keepdim=True)

    mods, srcs = [], {}
    orig = codecache.PyCodeCache.load_by_key_path

    def spy(key, path, *a, **kw):
        mod = orig(key, path, *a, **kw)
        try: srcs[id(mod)] = open(path).read()
        except Exception: pass
        mods.append(mod); return mod

    codecache.PyCodeCache.load_by_key_path = staticmethod(spy)
    try:
        with torch.no_grad():
            torch.compile(M().cuda().eval(), dynamic=True)(
                torch.randn(MMAX, D, device="cuda"))
    finally:
        codecache.PyCodeCache.load_by_key_path = staticmethod(orig)

    wrapper = next(m for m in mods if "call" in vars(m))
    src = srcs[id(wrapper)]
    symbols = sorted(set(re.findall(r"\b(s\d+)\b", src)))
    print(f"  symbols {symbols}, MMAX={MMAX}")

    v = analyse(src, symbols, MMAX)
    print()
    if v:
        print("FAIL monotonicity **does not hold**: the buffers above are larger at a smaller M.")
        print("  record-at-max would let them overrun into their neighbors -- device-side memory planning is required.")
        return 1
    print("ok every buffer reaches its maximum over the range at M=MMAX; the record-at-max premise holds.")
    print("  (for this model only; rerun for another model. A real system should make this check a hard assertion.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
