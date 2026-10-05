#!/usr/bin/env python3
"""
Extract the grid expression (a symbolic expression in the symints) of every Inductor kernel.

This is the **input** to planner codegen: DynaGraph's planner kernel has to compute each node's gridDim
on the device from these expressions, so the first step is to confirm that these expressions are actually
obtainable, and that they really are pure functions of the symints.

Design doc section 5 says "Inductor's ShapeEnv already holds sympy expressions; codegen them mechanically into the planner kernel" --
this script checks whether that statement holds.

Needs a GPU (Inductor's CUDA codegen needs a real device; without one it fails with
hasPrimaryContext expects a valid device index), but it **does not time anything**, so a shared card is fine.
"""
from __future__ import annotations

import os
import re
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")


def main():
    import torch
    import torch._inductor.config as ic
    ic.force_disable_caches = True
    # keep the generated code on disk so it can be read line by line
    ic.debug = False

    from torch._inductor import codecache

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(256, 512)
            self.l2 = torch.nn.Linear(512, 256)

        def forward(self, x):
            return torch.relu(self.l2(torch.relu(self.l1(x)))) + x

    dev = "cuda"
    m = M().to(dev).eval()
    x = torch.randn(8, 256, device=dev)

    captured = {}
    orig_write = codecache.PyCodeCache.load_by_key_path

    def spy(key, path, *a, **kw):
        try:
            src = open(path).read()
            if "def call(" in src or "triton_" in src:
                captured[path] = src
        except Exception:
            pass
        return orig_write(key, path, *a, **kw)

    codecache.PyCodeCache.load_by_key_path = staticmethod(spy)
    with torch.no_grad():
        torch.compile(m, dynamic=True)(x)
    codecache.PyCodeCache.load_by_key_path = staticmethod(orig_write)

    print(f"captured {len(captured)} generated files\n")
    total = 0
    for path, src in captured.items():
        # the lines in the wrapper that launch each kernel, of the form
        #   triton_poi_fused_0.run(arg0, arg1, xnumel, stream=...)
        # the grid is computed in the launcher from xnumel, so what we need is xnumel's symbolic expression
        launches = re.findall(r"(\w*triton_\w+)\.run\(([^)]*)\)", src)
        symints = re.findall(r"\b([su]\d+)\b", src)
        if not launches:
            continue
        print(f"--- {os.path.basename(path)}")
        print(f"    symints present: {sorted(set(symints))}")
        for name, args in launches:
            # the last positional argument is usually numel
            pos = [a.strip() for a in args.split(",") if "=" not in a]
            numel = pos[-1] if pos else "?"
            print(f"    {name:<34} numel = {numel}")
            total += 1
        # the assignments to the numel variables are the real symbolic expressions
        for line in src.splitlines():
            t = line.strip()
            if re.match(r"^\w*(xnumel|ynumel|rnumel)\s*=", t) or \
               re.match(r"^\w+_xnumel\s*=", t):
                print(f"      {t}")
        # the grid rule is in the launcher's meta
        for gm in re.findall(r"'?grid'?\s*[:=]\s*([^\n,]+)", src)[:3]:
            print(f"      grid rule: {gm.strip()}")
        # how the symbols are recovered from the input tensors
        for line in src.splitlines():
            t = line.strip()
            if re.match(r"^s\d+\s*=", t) or "assert_size_stride" in t:
                print(f"      [symbol source] {t[:110]}")
    print(f"\n{total} kernel launches in total")
    print("""
How to read this
----------------
If the numel column holds **arithmetic expressions** in the symints (e.g. 512*s0, s0*s1),
then section 5's "codegen mechanically into the planner kernel" holds -- the planner only needs s0, s1
to compute every node's grid on the device.
If something shows up that cannot be expressed in symints (a host-side variable, a data-dependent u0),
that is the case hard-problem list item #1 "unbacked SymInt end to end" has to handle.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
