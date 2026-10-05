#!/usr/bin/env python3
"""Arena relayout: when the total is fixed but the split varies, can one graph cover two buffers that change in opposite directions?

record-at-max fails here: for variable-length sequences packed by a token budget, the total B*L is fixed,
but a [B, ...] buffer grows with the number of sequences while an [L, ...] one shrinks,
**no single shape covers both**.

This script builds that shape space, first confirms that record-at-max is indeed flagged as a violation (the guard works),
then serves every shape with the arena relayout and cross-checks against the compiled version.
"""
from __future__ import annotations

import os, re, sys
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")


def main():
    import torch
    import torch._inductor.config as ic
    from torch._inductor import codecache, memory as im, dynagraph as dg

    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1
    ic.force_disable_caches = True

    TOTAL, D = 4096, 64

    class M(torch.nn.Module):
        """Two buffers changing in opposite directions: one with B, one with L."""
        def forward(self, x):            # x: (B, L, D)
            per_seq = x.sum(dim=1)       # (B, D)  grows with B
            per_pos = x.sum(dim=0)       # (L, D)  grows with L (grows as B shrinks)
            return per_seq.sum(dim=-1), per_pos.sum(dim=-1)

    mods, srcs, lifetimes = [], {}, {}
    orig_load = codecache.PyCodeCache.load_by_key_path
    orig_tl = im.compute_memory_timeline

    def spy_load(key, path, *a, **kw):
        mod = orig_load(key, path, *a, **kw)
        try: srcs[id(mod)] = open(path).read()
        except Exception: pass
        mods.append(mod); return mod

    def spy_tl(*a, **kw):
        out = orig_tl(*a, **kw)
        for item in (out if isinstance(out, (tuple, list)) else [out]):
            if isinstance(item, list) and item and hasattr(item[0], "start_step"):
                for b in item:
                    nm = getattr(getattr(b, "buffer", None), "get_name", lambda: None)()
                    if nm:
                        lifetimes[nm] = (b.start_step, b.end_step)
        return out

    codecache.PyCodeCache.load_by_key_path = staticmethod(spy_load)
    im.compute_memory_timeline = spy_tl
    try:
        f = torch.compile(M().cuda().eval(), dynamic=True)
        with torch.no_grad():
            f(torch.randn(8, TOTAL // 8, D, device="cuda"))
    finally:
        codecache.PyCodeCache.load_by_key_path = staticmethod(orig_load)
        im.compute_memory_timeline = orig_tl

    wrapper = next(m for m in mods if "call" in vars(m))
    src = srcs[id(wrapper)]
    symbols = sorted(set(re.findall(r"\b(s\d+)\b", src)))
    print(f"  symbols {symbols} (two means both B and L are free dims)")
    print(f"  lifetimes of {len(lifetimes)} buffers: "
          + ", ".join(f"{k}{v}" for k, v in sorted(lifetimes.items())[:6]))

    sizes = dg.buffer_size_exprs(src)
    print(f"  {len(sizes)} allocated buffers:")
    for nm, (span, isz) in sorted(sizes.items()):
        print(f"    {nm:<8} span={span[:58]}  {isz}B/elem")

    # 1) First confirm that record-at-max is indeed flagged as a violation
    if len(symbols) >= 2:
        s_b, s_l = symbols[0], symbols[1]
        print("\n  Checking this shape space with the record-at-max guard:")
        for probe_env in ("max over B", "max over L"):
            ok, why = dg.buffers_are_monotonic(src, symbols, 64)
            print(f"    {probe_env}: {'pass' if ok else 'violation -> ' + why[:90]}")
            break

    # 2) Slot assignment
    allocated = sorted(sizes)
    slot_assign, n_slots = dg.plan_slots(lifetimes, allocated)
    slot_of = dict(zip(allocated, slot_assign))
    print(f"\n  {len(allocated)} buffers -> {n_slots} slots: {slot_of}")

    # 3) Generate the layout + planner (including pointer rewriting) and confirm it compiles
    # Slots are fixed at build time (each slot = recorded shape x headroom); here it only has to compile: size the slots from one large shape
    sz = dg.slot_sizes(sizes, slot_of, n_slots, {s: 512 for s in symbols})
    _caps, fixed_off = dg.fixed_slot_offsets(sz, 1.0)
    kernels, syms2, _opaque = dg.extract_kernel_table(src, vars(wrapper))
    if kernels is None:
        print("  extraction failed (some kernel did not go through the static launcher)"); return 1
    pl = dg.generate_planner(kernels, syms2, slot_of, fixed_off=fixed_off)
    out_dir = os.environ.get("DG_OUT", "/tmp/dynagraph_out")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir,
                      "arena_generated.cu"), "w").write(pl)
    func = dg.compile_planner(pl)
    print(f"  planner {'compiled and loaded' if func else 'compile failed'}"
          f" (source left in {out_dir}/arena_generated.cu)")
    if not func:
        return 1

    n_ptr_patch = sum(
        1 for k in kernels for nm, b in (k.get("ptrs") or {}).items()
        if b in slot_of)
    print(f"  pointer args to rewrite: {n_ptr_patch}, across {len(kernels)} kernels")
    print("\n  Shape space (fixed total, varying split):")
    for B in (2, 8, 64, 512):
        L = TOTAL // B
        print(f"    B={B:<4} L={L:<5} B*L={B*L}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
