#!/usr/bin/env python3
"""
Extract the table the planner needs from a compiled model.

For each kernel node the planner needs to know three things:
  1. the **symbolic expression** of numel (like `512*s77`) -- determines the grid
  2. XBLOCK -- grid = ceil(numel / XBLOCK)
  3. the **byte offset** of the numel argument in the kernel parameter buffer -- needed by SetParam

Item 3 cannot be computed; you have to ask the driver. Measured (see docs/notes/SOLUTION.md):
PyTorch never builds a flat parameter buffer; every launch path uses cuLaunchKernel's pointer-array form,
and the real offsets are laid out in the cubin by the PTX ABI -- natural C alignment, int32 is **not** padded to 8 bytes.
So a formula like `8*N` is wrong: two i32 are packed back to back (16, 20), and an i32 followed by an i64 gets 4 bytes of padding.
The only reliable source is `cuFuncGetParamInfo`.

Needs a card (to load the cubin), but no timing, so it can run on a shared card.
"""
from __future__ import annotations

import ctypes
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

_libcuda = None


def _cuda():
    global _libcuda
    if _libcuda is None:
        _libcuda = ctypes.CDLL("libcuda.so.1")
    return _libcuda


def param_info(func_handle: int, index: int):
    """Ask the driver for the (offset, size) of parameter number index. Returns None if out of range."""
    off = ctypes.c_size_t()
    size = ctypes.c_size_t()
    rc = _cuda().cuFuncGetParamInfo(
        ctypes.c_void_p(func_handle), ctypes.c_size_t(index),
        ctypes.byref(off), ctypes.byref(size))
    if rc != 0:
        return None
    return off.value, size.value


def collect_from_modules():
    """Take the kernels from the globals of the loaded generated modules.

    Hooking `CachingAutotuner.__init__` does not work: most of what that catches are autotune
    **candidates** (measured: a two-layer MLP yields 42, all variants of the mm template, with
    kernel_name still at `Placeholder.DESCRIPTIVE_NAME`), while not a single pointwise/reduction
    kernel that ends up in the graph is caught.

    In the generated wrapper module, every kernel actually called by `call()` is a module-level
    CachingAutotuner instance, so scanning the globals is what gets them reliably.
    """
    from torch._inductor import codecache
    from torch._inductor.runtime import triton_heuristics as th

    mods = []
    orig = codecache.PyCodeCache.load_by_key_path

    def spy(key, path, *a, **kw):
        mod = orig(key, path, *a, **kw)
        mods.append(mod)
        return mod

    codecache.PyCodeCache.load_by_key_path = staticmethod(spy)

    def finish():
        codecache.PyCodeCache.load_by_key_path = staticmethod(orig)
        seen, out = set(), []
        for mod in mods:
            # Only look at wrapper modules. Autotune candidate modules also go through PyCodeCache;
            # scanning everything, a two-layer MLP picks up 42 mm candidates, which are not in the final graph.
            # A wrapper is marked by defining call().
            if not any(n in vars(mod) for n in ("call", "async_call")):
                continue
            for name, obj in vars(mod).items():
                if isinstance(obj, th.CachingAutotuner) and id(obj) not in seen:
                    seen.add(id(obj))
                    out.append((name, obj))
        return out

    return finish


def main():
    import torch
    import torch._inductor.config as ic

    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1
    ic.force_disable_caches = True
    # Without routing GEMM, addmm goes through extern_kernels; those nodes have no handle and no numel expression
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(256, 256)
            self.l2 = torch.nn.Linear(256, 256)

        def forward(self, x):
            h = torch.relu(self.l1(x))
            h = h - h.mean(dim=-1, keepdim=True)
            h = torch.softmax(h, dim=-1)
            return torch.relu(self.l2(h)).sum(dim=-1)

    finish = collect_from_modules()
    m = M().cuda().eval()
    with torch.no_grad():
        torch.compile(m, dynamic=True)(torch.randn(512, 256, device="cuda"))
    found = finish()

    # Most of what gets caught are autotune **candidates** (21 triton_mm variants and the like),
    # with kernel_name still at Placeholder.DESCRIPTIVE_NAME, not the finally selected kernel.
    # The planner only cares about the ones that actually end up in the graph.
    print(f"got {len(found)} in-graph kernels from the generated modules\n")
    rows = []
    for _gname, at in found:
        name = at.inductor_meta.get("kernel_name", "?")
        grid_type = at.inductor_meta.get("grid_type", "?")
        sig = (at.triton_meta or {}).get("signature", {})
        # constexpr args are not in the parameter buffer; compute indices with the same filter
        runtime_args = [k for k, v in sig.items() if v != "constexpr"]
        numel_names = [k for k in runtime_args if k.endswith("numel")]

        blocks, offsets = {}, {}
        for lr in getattr(at, "launchers", []) or []:
            cfg = getattr(lr, "config", None)
            for bk in ("XBLOCK", "R0_BLOCK", "YBLOCK"):
                if bk in getattr(cfg, "kwargs", {}) if cfg else {}:
                    blocks[bk] = cfg.kwargs[bk]

        # The CUfunction is at compile_results[i].kernel.function.
        # **Not on the launcher** -- the launcher is a closure created by exec,
        # and it only carries config / n_regs / shared and the like, not the kernel.
        for cr in getattr(at, "compile_results", []) or []:
            k = getattr(cr, "kernel", None)
            fh = getattr(k, "function", None)
            if not fh and getattr(k, "functions", None):
                fh = next(iter(k.functions.values()), None)
            if not fh:
                continue
            for nm in numel_names:
                pi = param_info(fh, runtime_args.index(nm))
                if pi:
                    offsets[nm] = pi
            # Also print the offsets of all parameters, to verify the "not a multiple of 8" claim
            offsets["__all__"] = [param_info(fh, i) for i in range(len(runtime_args) + 2)]
            break
        rows.append((name, grid_type, runtime_args, numel_names, blocks, offsets))

    for name, gt, args, nns, blocks, offs in rows:
        print(f"--- {name}   grid_type={gt}")
        print(f"    runtime args({len(args)}): {args}")
        print(f"    block: {blocks or '(not found)'}")
        for nm in nns:
            o = offs.get(nm)
            print(f"    {nm:<10} offset={o[0] if o else '?'} bytes  size={o[1] if o else '?'}")
        allp = offs.get("__all__")
        if allp:
            print(f"    all params (offset,size): "
                  f"{[x for x in allp if x is not None]}")
    print("""
How to read this
----------------
With the (numel expression, XBLOCK, param offset) triple, planner_codegen.Node can be built directly.
Offsets must come from cuFuncGetParamInfo -- if the numbers above are not multiples of 8
(e.g. two numels at 16 and 20), the 8*N formula is wrong, which is exactly why we ask the driver.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
