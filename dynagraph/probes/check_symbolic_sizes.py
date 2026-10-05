#!/usr/bin/env python3
"""With dynamic shapes, does Inductor's memory planning produce symbolic expressions or concrete integers?

This is the most uncertain step on the arena re-layout path. The planner has to compute each
slot's size on the device from the runtime symints, which requires getting the sizes in
**symbolic form** at compile time; if Inductor has already concretized the sizes into integers
at this layer, the path needs a different entry point.

`memory.py:138 compute_size_for_scheduler_buffer` is annotated as
`dict[str, tuple[int, int]]`, but the values come from `get_allocation_size`,
whose signature is `-> Sequence[Expr]`. The annotation and reality may disagree, so measure it.

Runs on a shared card; no timing.
"""
import os, sys
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")


def main():
    import torch
    import torch._inductor.config as ic
    from torch._inductor import memory as im

    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1
    ic.force_disable_caches = True

    captured = {}
    orig_size = im.compute_size_for_scheduler_buffer
    orig_tl = im.compute_memory_timeline

    def spy_size(name_to_buf):
        out = orig_size(name_to_buf)
        captured.setdefault("sizes", []).append(out)
        return out

    def spy_tl(*a, **kw):
        out = orig_tl(*a, **kw)
        captured.setdefault("timeline", []).append(out)
        return out

    im.compute_size_for_scheduler_buffer = spy_size
    im.compute_memory_timeline = spy_tl

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.l1 = torch.nn.Linear(256, 256)
        def forward(self, x):
            h = torch.relu(self.l1(x))
            h = h - h.mean(dim=-1, keepdim=True)
            return torch.softmax(h, dim=-1) * h.sum(dim=0, keepdim=True)

    try:
        with torch.no_grad():
            torch.compile(M().cuda().eval(), dynamic=True)(
                torch.randn(1024, 256, device="cuda"))
    finally:
        im.compute_size_for_scheduler_buffer = orig_size
        im.compute_memory_timeline = orig_tl

    import sympy
    sizes = captured.get("sizes") or []
    print(f"  compute_size_for_scheduler_buffer called {len(sizes)} times")
    n_sym = n_int = 0
    for d in sizes:
        for name, (alloc, free) in d.items():
            is_sym = isinstance(alloc, sympy.Expr) and alloc.free_symbols
            n_sym += bool(is_sym); n_int += (not is_sym)
            print(f"    {name:<8} size_alloc={alloc!r:<34} "
                  f"{'symbolic' if is_sym else 'concrete int'}")
    tls = captured.get("timeline") or []
    if tls:
        bufinfos = None
        for t in tls:
            for item in (t if isinstance(t, (tuple, list)) else [t]):
                if isinstance(item, list) and item and hasattr(item[0], "start_step"):
                    bufinfos = item
        if bufinfos:
            print(f"\n  memory timeline: {len(bufinfos)} buffers")
            for b in bufinfos[:8]:
                nm = getattr(getattr(b, "buffer", None), "get_name", lambda: "?")()
                print(f"    {nm:<8} live steps [{b.start_step}, {b.end_step}]  "
                      f"alloc={b.size_alloc!r}")
    print()
    if n_sym:
        print(f"OK: got symbolic sizes ({n_sym} symbolic / {n_int} constant) -- "
              "the planner can compute slot sizes from symints, so arena re-layout is feasible.")
        return 0
    print("FAIL: all concrete integers; this layer has already baked in the sizes, so the entry point has to be further upstream "
          "(get_allocation_size in graph.py or empty_strided_cuda in the wrapper).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
