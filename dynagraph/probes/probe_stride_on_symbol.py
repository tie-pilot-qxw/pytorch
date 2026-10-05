#!/usr/bin/env python3
"""Probe: a buffer's STRIDE itself depends on a symbol (not just its size).

The risk targeted:
  transpose/permute makes an intermediate buffer non-contiguous, and Inductor allocates it as
      buf1 = empty_strided_cuda((s77, 128), (1, s77), torch.float32)
  Here s77 shows up in two unrelated places at once:
    1. in the kernel as the store's stride multiplier (`out_ptr1 + (idx_m + idx_n*ks0)`),
       and the planner must change ks0 to the current shape's value;
    2. in the arena's layout kernel as the span (1 + (s-1)*1 + 127*s) used to size this buffer in bytes.
  The two must agree. If either one stays at the recorded shape there is no error: either it writes into the
  neighbouring buffer's space, or the as_strided view reads with the wrong row pitch -- numerics silently go wrong.

  This path was never tested before, so the probe's first job is not to judge right or wrong but to prove it
  "really got there": parse empty_strided_cuda from the wrapper source DynaGraph itself reads, and require at least one
  allocation whose stride tuple contains a symbol and whose last-dim stride is not 1 (i.e. really non-contiguous); then
  find the lines in the kernel source that really use this symbolic argument as an index multiplier. If either fails,
  the probe fails on the spot, so it never prints a pass when "nothing was actually tested".

KIND: A -- DynaGraph should be able to serve this construct: dynagraph=True records no graph at all,
the control records one per shape, and numerics match the control.

Measured: the symbolic stride appears in two roles, and the planner patches both:
  * in the mm template triton_tem_..._0, ks0 is both M and the store's row pitch
    (`out_ptr1 + (idx_m + idx_n*ks0)`); here the stride happens to equal size[0];
  * in the reduction kernel triton_red_fused_sum_1, ks0 is a pure stride
    (`in_ptr0 + (r0_1 + ks0*x0)`); xnumel=128 and r0_numel=s77 are two other arguments,
    and ks0 is only used to step across rows. This one is the clean "the stride itself is a symbol" case.
A dense transposed layout's stride always equals some dimension's size; Inductor never emits an allocation whose stride
is unrelated to every size, so these two are what this probe can cover.
"""
import os, sys, logging

# The first shape must be the largest: DynaGraph records on the first shape it sees, and sizes the arena by
# the symbol values at that time (headroom=2.0). Small-then-large would push later shapes out of the arena; that is another probe's job.
SHAPES = tuple(int(v) for v in os.environ.get("DG_SHAPES", "512,333,256,96,7").split(","))
FEAT = 64          # input feature dim, kept constant; only the row count is symbolic
DUMP = os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "stride_on_symbol_planner.cu")
os.makedirs(os.path.dirname(DUMP), exist_ok=True)

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
os.environ["TORCHINDUCTOR_DYNAGRAPH_DUMP"] = DUMP

# Fallback reasons are only printed at INFO. KIND A should not fall back, but if it does we need to see which reason
# right away, not just "the record count is wrong".
logging.basicConfig(format="[log] %(name)s: %(message)s")
logging.getLogger("torch._inductor.cudagraph_trees").setLevel(logging.INFO)
logging.getLogger("torch._inductor.dynagraph").setLevel(logging.INFO)


def build_model(torch):
    class Net(torch.nn.Module):
        """h.t().contiguous() and back again, forcing Inductor to pick a non-contiguous layout for the intermediate.

        The chain is fused into one mm template kernel: its epilogue writes out directly with stride (1, s77),
        so the symbolic stride exists as a kernel scalar argument (ks0) -- exactly the kind of argument
        the planner must patch and is most likely to miss.
        """

        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(FEAT, 128)

        def forward(self, x):                    # x: (s, 64)
            h = torch.relu(self.l(x))            # (s, 128) contiguous
            c = h.t().contiguous()               # (128, s)
            d = torch.sigmoid(c.t()) * 2.0       # (s, 128), Inductor allocates it with stride (1, s)
            return d, d.sum(dim=0)               # second consumer: makes the non-contiguous buffer get read as well

    return Net


def run(dynagraph: bool):
    import torch
    import torch._inductor.config as ic

    torch._dynamo.reset()
    ic.force_disable_caches = True
    ic.triton.dynagraph = dynagraph
    # GEMMs must be routed to Triton: extern_kernels.mm bypasses the static launcher, so no node handle is available;
    # the whole graph would fall back on a handle-count mismatch, and then control and experiment cannot be compared.
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"

    import torch._inductor.cudagraph_trees as ct
    import torch._inductor.dynagraph as dg

    n_record = {"n": 0}
    orig = ct.CUDAGraphTreeManager.record_function

    def spy(self, *a, **kw):
        n_record["n"] += 1
        return orig(self, *a, **kw)

    ct.CUDAGraphTreeManager.record_function = spy

    # Keep a copy of the wrapper source DynaGraph itself reads. The hit check is based on it:
    # the probe's claim "tested a symbolic stride" needs evidence, not a guess from how the model is written.
    captured = {"src": None}
    o_src = dg._wrapper_source

    def spy_src(model):
        s = o_src(model)
        if s and captured["src"] is None:
            captured["src"] = s
        return s

    dg._wrapper_source = spy_src

    Net = build_model(torch)
    # Both runs must get the same weights and the same inputs, or outputs cannot be compared bitwise.
    torch.manual_seed(0)
    m = Net().cuda().eval()
    f = torch.compile(m, dynamic=True, mode="reduce-overhead")

    outs = {}
    try:
        torch.manual_seed(1)
        for S in SHAPES:
            x = torch.randn(S, FEAT, device="cuda")
            with torch.no_grad():
                # Call each shape twice: the first time cudagraph_trees sees a FunctionID it only does an
                # eager warmup, and records on the second. With one call the control never runs a cudagraph at all.
                f(x)
                y = f(x)
                # Read strides before moving to CPU: DynaGraph's outputs are as_strided views on the arena,
                # with strides computed for the current shape -- exactly what is under test.
                strides = tuple(t.stride() for t in y)
                vals = tuple(t.float().cpu().clone() for t in y)   # the next replay overwrites it
                ref = tuple(t.float().cpu().clone() for t in m(x))
                outs[S] = (strides, vals, ref)
    finally:
        ct.CUDAGraphTreeManager.record_function = orig
        dg._wrapper_source = o_src

    return n_record["n"], outs, captured["src"]


def allocations(src):
    """(name, sizes, strides) of every empty_strided_cuda in the wrapper.

    Uses dynagraph's own parser so the probe and the code under test cannot read the same text two different ways.
    """
    from torch._inductor.dynagraph import _find_allocations

    return [(n, sz, st) for n, sz, st, _dt in _find_allocations(src)]


def symbolic_strided(src):
    """Allocations whose stride contains a symbol and that are not row-major contiguous.

    "stride contains s\\d+" alone is not enough: a contiguous (s77, 128) has a symbolic size but constant strides;
    and a contiguous (128, s77) has strides (s77, 1), which are symbolic but still an ordinary contiguous layout,
    so no extra stride multiplier appears in the kernel. What must be tested is the case where the last-dim stride is not 1.
    """
    import re

    out = []
    for name, sizes, strides in allocations(src):
        if not strides or len(sizes) != len(strides):
            continue
        has_sym = any(re.search(r"\bs\d+\b", s) for s in strides)
        noncontig = strides[-1].strip() != "1"
        if has_sym and noncontig:
            out.append((name, sizes, strides))
    return out


def patched_params():
    """Arguments the planner really rewrites via SetParam, read from the dump's end-of-line comments, split into scalars and pointers.

    generate_planner only emits SetParam for scalars where `_is_symbolic` is true, and leaves
    `// ks0 = s77` at the end of the line. Only if that line is present was the symbolic stride really patched as a kernel
    argument, rather than happening to be baked into a constant. The pointer group is arena relocation, a separate matter from strides, listed separately.
    """
    import re

    if not os.path.exists(DUMP):
        return None, None
    txt = open(DUMP).read()
    if os.environ.get("TORCHINDUCTOR_DYNAGRAPH_UPDATE") == "host":
        # Host path: each patch is one line `... nd->buf.data() + off ...  // name = expr`;
        # comments for pointers carry a parenthesized note (extern output / input by address) or are just bufN.
        all_ = re.findall(r"nd->buf\.data\(\)[^/\n]*//\s*(\S+) = (.+)", txt)
        is_ptr = lambda b: bool(re.fullmatch(r"buf\d+", b.strip())) or "(" in b
    else:
        all_ = re.findall(r"cudaGraphKernelNodeSetParam\(handles\[i\][^/]*//\s*(\S+) = (.+)", txt)
        is_ptr = lambda b: bool(re.fullmatch(r"buf\d+", b.strip()))
    scalars = [f"{a} = {b.strip()}" for a, b in all_ if not is_ptr(b)]
    ptrs = [f"{a} = {b.strip()}" for a, b in all_ if is_ptr(b)]
    return scalars, ptrs


def index_lines(src, syms=("ks0", "ks1", "ks2")):
    """The kernel lines that really use a symbolic argument as an index multiplier.

    This is direct evidence that "the symbolic stride is used as a stride": empty_strided_cuda alone only shows
    the buffer layout is non-contiguous; we also need to see tl.load/tl.store addresses really multiply by this argument
    before we can say the planner patching this argument is patching a stride.
    """
    out = []
    for ln in src.splitlines():
        t = ln.strip()
        if ("tl.load(" in t or "tl.store(" in t) and any(k in t for k in syms):
            out.append(t)
    return out


def main():
    import torch

    if not torch.cuda.is_available():
        print("no CUDA device available")
        return 1

    res, rec, srcs = {}, {}, {}
    for flag in (False, True):
        n, outs, src = run(flag)
        res[flag], rec[flag], srcs[flag] = outs, n, src
        print(f"\n  dynagraph={flag}  recorded {n}x")

    bad = 0

    # ---- 1. first prove the probe hit its target; otherwise the comparisons below mean nothing ----
    print("\n  [hit] allocations in the wrapper:")
    src = srcs[True]
    if not src:
        print("    FAIL no wrapper source (DynaGraph never even reached _wrapper_source)")
        return 1
    for name, sizes, strides in allocations(src):
        print(f"    {name}: sizes=({', '.join(sizes)}) strides=({', '.join(strides)})")
    hits = symbolic_strided(src)
    if not hits:
        print("    FAIL no allocation is 'symbolic stride + non-contiguous'; this probe did not test what it is meant to")
        return 1
    for name, sizes, strides in hits:
        print(f"    ok {name} stride ({', '.join(strides)}) is symbolic and non-contiguous -- hit")
    scalars, ptrs = patched_params()
    print(f"    scalar args rewritten by the planner: {scalars if scalars else '(none -- the symbolic stride did not become a patched kernel argument)'}")
    print(f"    pointer args relocated by the planner: {ptrs if ptrs else '(none)'}")
    for ln in index_lines(src)[:4]:
        print(f"    symbolic arg used in index: {ln[:150]}")
    if not scalars:
        bad += 1

    # ---- 2. output strides must follow the shape, not stay at the recorded shape ----
    # This is the probe's most direct criterion: DynaGraph's outputs are as_strided on the arena;
    # a wrong stride raises no exception, it just reads with the wrong row pitch.
    print("\n  [stride] stride of output 0 (expected (1, S), i.e. follows the shape):")
    for S in SHAPES:
        ctl = res[False][S][0][0]
        dyn = res[True][S][0][0]
        ok = ctl == dyn == (1, S)
        print(f"    S={S} control {ctl} | dyna {dyn}" + ("  ok" if ok else "  FAIL"))
        bad += not ok

    # ---- 3. numerics: the reference is the same compile path with dynagraph off, not eager ----
    # GEMMs are routed to Triton, which differs algorithmically from eager's cuBLAS; using eager as the criterion
    # would read algorithmic differences as correctness problems. Eager is only used to set the scale.
    print("\n  [numerics] criterion: no farther from eager than the control")
    for S in SHAPES:
        (_, a, ea), (_, b, eb) = res[False][S], res[True][S]
        for i in range(len(a)):
            if a[i].shape != b[i].shape:
                print(f"    S={S} out{i} shape {tuple(b[i].shape)} != {tuple(a[i].shape)}  FAIL")
                bad += 1
                continue
            # The eager references of the two runs must be bitwise identical; otherwise it is a seeding problem, not DynaGraph's fault.
            seed_ok = (ea[i] - eb[i]).abs().max().item() == 0
            scale = max(ea[i].abs().max().item(), 1e-9)
            c = (a[i] - ea[i]).abs().max().item() / scale
            d = (b[i] - eb[i]).abs().max().item() / scale
            ok = seed_ok and d <= max(c * 1.5, 1e-6)
            print(
                f"    S={S} out{i} control<->dyna {(a[i] - b[i]).abs().max().item():.2e}"
                f" | control<->eager {c:.1e} | dyna<->eager {d:.1e}"
                f" | same seed {seed_ok}" + ("  ok" if ok else "  FAIL")
            )
            bad += not ok

    # ---- 4. the point of it all: the control records one graph per shape, DynaGraph serves all with one ----
    print(f"\n  record count {rec[False]} -> {rec[True]} ({len(SHAPES)} shapes)")
    if rec[True] != 0 or rec[False] != len(SHAPES):
        print("  FAIL record count wrong: DynaGraph should record zero times, the control once per shape")
        bad += 1

    print("\n  " + ("all passed" if not bad else f"{bad} items failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
