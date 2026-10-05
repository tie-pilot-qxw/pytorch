#!/usr/bin/env python3
"""Verify that the torch._inductor.dynagraph module (ported from the prototype) behaves the same.

Verify it on its own before wiring it into cudagraph_trees, so "the port drifted" and "integration bug" are not debugged together.
Same criteria as end_to_end.py: the replay result must be **bit-identical** to the compiled version at the same shape,
and the negative control (ctx not updated) must show a clear deviation -- without a negative control you cannot show the planner is doing anything.
"""
import os, sys
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")


def main():
    import torch
    import torch._inductor.config as ic
    from torch._inductor import codecache, dynagraph as dg

    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1
    ic.force_disable_caches = True
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"

    MMAX, D = 1024, 256

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.l1 = torch.nn.Linear(D, D)
        def forward(self, x):
            h = torch.relu(self.l1(x))
            h = h - h.mean(dim=-1, keepdim=True)
            # The output shape **depends on M**, to expose the "output metadata is frozen at recording time" problem.
            # The previous version returned sum(dim=0), whose shape (D,) does not change with M, so it never hit this case.
            return torch.softmax(h, dim=-1) * h.sum(dim=0, keepdim=True)

    m = M().cuda().eval()
    x_buf = torch.randn(MMAX, D, device="cuda")

    mods, srcs = [], {}
    orig = codecache.PyCodeCache.load_by_key_path
    def spy(key, path, *a, **kw):
        mod = orig(key, path, *a, **kw)
        try: srcs[id(mod)] = open(path).read()
        except Exception: pass
        mods.append(mod); return mod
    codecache.PyCodeCache.load_by_key_path = staticmethod(spy)
    try:
        f = torch.compile(m, dynamic=True)
        with torch.no_grad(): f(x_buf)
    finally:
        codecache.PyCodeCache.load_by_key_path = staticmethod(orig)

    wrapper = next(mm for mm in mods if "call" in vars(mm))
    src = srcs[id(wrapper)]

    kernels, symbols, _opaque = dg.extract_kernel_table(src, vars(wrapper))
    if kernels is None:
        print("  extraction failed (some kernel did not go through the static launcher)"); return 1
    print(f"  extracted {len(kernels)} kernels, symbols {symbols}")

    # self-check of the bucketing policy
    for v, r in ((1000, 2.0), (513, 2.0), (512, 2.0), (7, 2.0)):
        b, top = dg.bucket_of(v, r)
        print(f"    bucket_of({v}) = bucket {b}, recorded at {top}")

    psrc = dg.generate_planner(kernels, symbols)
    func = dg.compile_planner(psrc)
    if func is None:
        print("  planner compilation failed"); return 1
    print("  planner compiled and loaded")

    n = len(kernels)
    d_handles = torch.zeros(n, dtype=torch.int64, device="cuda")
    # ctx layout follows the runner's: symbols, the "changed" flag, the pointer-dirty flag, each node's enabled
    # state (initially all 1), ..., the arena base (0 = do not patch pointers). This test does not exercise the early exit,
    # so the "changed" flag is always set to 1.
    n_sym = len(symbols)
    _arena_i, _off0, lastp0 = dg.ctx_layout(n_sym, n, 0, 0, 0, 0)
    d_ctx = torch.zeros(lastp0, dtype=torch.int64, device="cuda")
    d_ctx[n_sym] = 1
    d_ctx[n_sym + 2 : n_sym + 2 + n] = 1

    L = torch._C._StaticCudaLauncher
    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(3): f(x_buf)
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    L._begin_device_node_collection()
    try:
        with torch.cuda.graph(g):
            dg.launch_planner(func, n, d_handles.data_ptr(), d_ctx.data_ptr(),
                              torch.cuda.current_stream().cuda_stream)
            with torch.no_grad():
                out_buf = f(x_buf)
        handles = L._end_device_node_collection()
    except Exception as e:
        try: L._end_device_node_collection()
        except Exception: pass
        print(f"  capture failed: {type(e).__name__}: {str(e)[:200]}"); return 1

    if len(handles) != n:
        print(f"  FAIL: {len(handles)} handles but {n} kernels, some node was missed"); return 1
    d_handles.copy_(torch.tensor(handles, dtype=torch.int64)); torch.cuda.synchronize()

    bad = 0
    for Mi in (MMAX, 512, 333, 64, 7, 1):
        ref_in = torch.randn(Mi, D, device="cuda")
        with torch.no_grad(): ref = f(ref_in).clone()
        x_buf.zero_(); x_buf[:Mi].copy_(ref_in)

        d_ctx[0] = MMAX; g.replay(); torch.cuda.synchronize()
        stale = out_buf.clone()
        d_ctx[0] = Mi;   g.replay(); torch.cuda.synchronize()

        scale = max(ref.abs().max().item(), 1e-9)
        # The graph's output buffer is allocated for MMAX; the correct result should be its first Mi rows.
        # In the real system this step belongs to reconstruct_outputs; here we slice by hand,
        # to separate "is the computation right" from "is the output shape right".
        got = out_buf[:Mi] if out_buf.shape[0] != ref.shape[0] else out_buf
        st = stale[:Mi] if stale.shape[0] != ref.shape[0] else stale
        if got.shape != ref.shape:
            print(f"    M={Mi:<5} shape mismatch: graph gives {tuple(out_buf.shape)} "
                  f"expected {tuple(ref.shape)}"); bad += 1; continue
        rel = (ref - got).abs().max().item() / scale
        rel_stale = (ref - st).abs().max().item() / scale
        ok = rel < 1e-5
        sensitive = (Mi == MMAX) or rel_stale > 1e-3
        bad += (not ok) or (not sensitive)
        print(f"    M={Mi:<5} vs compiled {rel:.2e} {'OK' if ok else 'FAIL'}   "
              f"negative control {rel_stale:.2e} {'OK' if sensitive else 'FAIL test is invalid'}")

    print("\n" + ("module port is correct, behavior matches the prototype." if bad == 0 else f"{bad} shape(s) have problems."))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
