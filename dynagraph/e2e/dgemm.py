"""bf16 GEMM through DeepGEMM: an operator that writes into a given output + its DynaGraph declaration + an Inductor lowering.

After install(), Inductor lowers eligible aten.mm / aten.addmm to dgemm::mm_out (an extern call), and
DynaGraph inlines it per the declaration as an ordinary kernel node in the main graph (tier 3) instead of
the tier-2 cuBLAS harvest.

DeepGEMM bf16_gemm_nt(a (M,K), b (N,K), d (M,N)): whether a and b are each K-major or MN-major is read from
the strides, so one function covers all four transpose cases of mm. Measured (verification/_wf_dgemm_layouts.py,
verification/_wf_dgemm_padstride.py), the only requirement is TMA's 16-byte alignment: every non-unit stride
is a multiple of 8 (bf16). The dims themselves are unrestricted (dW's K = node count, SchNet's 50 Gaussian bases
all work), so a misaligned operand is copied into a layout whose stride is padded to 8, and the output is
allocated with the padded stride too.

The declaration goes through DeepGEMM's describe ($DG_DEPS/deepgemm-src, DG_DEPS defaulting to /workspace/_deps;
a sink added in launch_kernel, see docs/notes/E2E.md): called between describe_begin/end, the library's host code
runs as usual (heuristics, JIT, TMA descriptors) and the launches are recorded instead of issued. ~12 us per site
per new shape, same as a direct call; a bare capture is ~58 us.
Without this DeepGEMM build it falls back to recorded (bare capture).
"""
import torch
from torch.utils import _capture_launch as cl

try:
    import deep_gemm

    DESCRIBE = hasattr(deep_gemm._C, "describe_begin") and __import__("os").environ.get("DGEMM_DESCRIBE", "1") == "1"
except ImportError:
    import vllm.third_party.deep_gemm as deep_gemm

    DESCRIBE = False

aten = torch.ops.aten



@torch.library.custom_op("dgemm::mm_out", mutates_args=("out",))
def mm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, compiled_dims: str) -> None:
    deep_gemm.bf16_gemm_nt(a, b.t(), out, compiled_dims=compiled_dims)
    if _CHECK and not torch.cuda.is_current_stream_capturing():
        ref = a.float() @ b.float()
        err = ((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
        if not err < 3e-2:
            print(f"dgemm BAD err {err:.3g} a {tuple(a.shape)} {a.stride()} b {tuple(b.shape)} {b.stride()} "
                  f"out {out.stride()} dims {compiled_dims!r}", flush=True)


_CHECK = __import__("os").environ.get("DGEMM_CHECK") == "1"
SHARE = __import__("os").environ.get("DGEMM_SHARE", "1") == "1"


@mm_out.register_fake
def _(a, b, out, compiled_dims):
    return None


def _prepare(a, b, out, compiled_dims):
    # DeepGEMM JIT-compiles a config the first time a call picks it; do the first one outside capture.
    mm_out(a, b, out, compiled_dims)


def _described(a, b, out, compiled_dims):
    """What mm_out would launch, from DeepGEMM itself, nothing issued."""
    deep_gemm._C.describe_begin()
    try:
        deep_gemm.bf16_gemm_nt(a, b.t(), out, compiled_dims=compiled_dims)
    finally:
        got = deep_gemm._C.describe_end()
    return [_launch(*x) for x in got]


if DESCRIBE:
    # Sites with the same geometry (every layer) share one describe; the rest get its launch
    # with the addresses moved. DGEMM_SHARE=0 describes every site.
    cl.register("dgemm::mm_out", _described, prepare=_prepare, share=SHARE)
else:
    cl.register("dgemm::mm_out", cl.recorded(mm_out, template_key=lambda *a: 0, allocates=False),
                prepare=_prepare, exact=False)


@torch.library.custom_op("dgemm::nt_out", mutates_args=("out",))
def nt_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, compiled_dims: str) -> None:
    """out = a @ b.T, b (N, K): deep_gemm.bf16_gemm_nt as it is, so the C describe below is the
    library's own entry point with nothing of this wrapper in it (mm_out transposes b itself)."""
    deep_gemm.bf16_gemm_nt(a, b, out, compiled_dims=compiled_dims)


@nt_out.register_fake
def _(a, b, out, compiled_dims):
    return None


def _nt_described(a, b, out, compiled_dims):
    deep_gemm._C.describe_begin()
    try:
        deep_gemm.bf16_gemm_nt(a, b, out, compiled_dims=compiled_dims)
    finally:
        got = deep_gemm._C.describe_end()
    return [_launch(*x) for x in got]


def _c_describe():
    """The address of DeepGEMM's C describe entry (patch_dgd_c.py), or None without it."""
    import ctypes

    try:
        lib = ctypes.CDLL(deep_gemm._C.__file__)
        return ctypes.cast(lib.dg_describe_bf16_gemm_nt, ctypes.c_void_p).value
    except (OSError, AttributeError):
        return None


if DESCRIBE:
    cl.register("dgemm::nt_out", _nt_described, prepare=lambda *a: nt_out(*a),
                c_describe=_c_describe(), c_statics=lambda a, b, out, dims: dims.encode())

# nt: lower to nt_out (b passed as an (N, K) view; what the C describe takes); mm: mm_out.
OP = __import__("os").environ.get("DGEMM_OP", "nt")


@torch.library.custom_op("dgemm::grouped_mm_out", mutates_args=("out",))
def grouped_mm_out(a: torch.Tensor, w: torch.Tensor, layout: torch.Tensor, out: torch.Tensor) -> None:
    """out[r] = a[r] @ w[layout[r]].T for rows whose layout is >= 0 (MoE-style contiguous grouping,
    each group's rows padded to get_mk_alignment_for_contiguous_layout(); padding rows are garbage)."""
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, w, out, layout)


@grouped_mm_out.register_fake
def _(a, w, layout, out):
    return None


def _grouped_described(a, w, layout, out):
    deep_gemm._C.describe_begin()
    try:
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, w, out, layout)
    finally:
        got = deep_gemm._C.describe_end()
    return [_launch(*x) for x in got]


def _launch(func, grid, block, smem, cluster, pdl, args):
    if pdl:
        # Programmatic dependent launch is an edge property in a graph, not a node one.
        raise cl.Mismatch("DeepGEMM launched with PDL")
    return cl.Launch(func, tuple(grid), tuple(block), smem, tuple(args), None if cluster == 1 else (cluster, 1, 1))


if DESCRIBE:
    cl.register("dgemm::grouped_mm_out", _grouped_described, prepare=grouped_mm_out, share=SHARE)


def _static_ok(x) -> bool:
    """bf16, 2-D, fixed static strides: one of them 1 and the other a multiple of 8."""
    if x.get_dtype() != torch.bfloat16 or len(x.get_size()) != 2:
        return False
    st = x.get_stride()
    if not all(isinstance(s, int) or getattr(s, "is_number", False) for s in st):
        return False
    st = [int(s) for s in st]
    unit = [i for i in range(2) if st[i] == 1 or x.get_size()[i] == 1]
    if not unit:
        return False
    other = 1 - unit[0]
    return st[other] % 8 == 0


def _up8(v):
    return -(-int(v) // 8) * 8


def _static(v) -> bool:
    return isinstance(v, int) or getattr(v, "is_number", False)


def _aligned(x):
    """x, or a copy of it whose non-unit stride is a multiple of 8 (None if no static dim to pad)."""
    from torch._inductor import ir

    if _static_ok(x):
        return x
    size, st = x.get_size(), x.get_stride()
    # Keep the dim that is already contiguous contiguous; pad the other one's stride.
    inner = 0 if (st[0] == 1 and st[1] != 1) else 1
    if not _static(size[inner]):
        inner = 1 - inner
        if not _static(size[inner]):
            return None
    exact = [_up8(size[1]), 1] if inner == 1 else [1, _up8(size[0])]
    x = ir.ExternKernel.require_strides(x, exact_strides=exact)
    return x if _static_ok(x) else None


def _fixed(x):
    from torch._inductor import ir

    x = ir.ExternKernel.realize_input(x)
    if ir.is_storage_and_layout(x):
        x.freeze_layout()
    return x


STATS = {"deepgemm": 0, "kept": 0}


def install():
    from torch._inductor import ir
    from torch._inductor import lowering as L

    orig_mm = L.lowerings[aten.mm.default]
    orig_addmm = L.lowerings[aten.addmm.default]

    def try_mm(a, b):
        if a.get_dtype() != torch.bfloat16 or b.get_dtype() != torch.bfloat16:
            return None
        n = b.get_size()[1]
        if not _static(n):
            return None
        if OP == "nt":
            b = L.lowerings[aten.permute.default](b, [1, 0])
        a, b = _aligned(_fixed(a)), _aligned(_fixed(b))
        if a is None or b is None:
            return None
        m, k = a.get_size()
        # DeepGEMM compiles the dims it is told are fixed into the kernel (default "nk"); a dim
        # that is a symbol here would make it JIT a kernel per value (nvcc, seconds), so only
        # the static ones are named. M never is, as in DeepGEMM's own default.
        dims = "".join(c for c, v in (("n", n), ("k", k)) if isinstance(v, int) or getattr(v, "is_number", False))
        out = L.empty_strided([m, n], [_up8(n), 1], dtype=torch.bfloat16, device=a.get_device())
        if OP == "nt":
            ir.FallbackKernel.create(torch.ops.dgemm.nt_out.default, a, b, out, dims)
        else:
            ir.FallbackKernel.create(torch.ops.dgemm.mm_out.default, a, b, out, dims)
        return out

    def mm(a, b, *args, **kw):
        if not args and not kw:
            r = try_mm(a, b)
            if r is not None:
                STATS["deepgemm"] += 1
                return r
        STATS["kept"] += 1
        return orig_mm(a, b, *args, **kw)

    def addmm(inp, a, b, *, alpha=1, beta=1, **kw):
        if alpha == 1 and beta == 1 and not kw:
            r = try_mm(a, b)
            if r is not None:
                STATS["deepgemm"] += 1
                return L.lowerings[aten.add.Tensor](r, inp)
        STATS["kept"] += 1
        return orig_addmm(inp, a, b, alpha=alpha, beta=beta, **kw)

    L.lowerings[aten.mm.default] = mm
    L.lowerings[aten.addmm.default] = addmm
    # Process-level set-up: DeepGEMM's DeviceRuntime allocates a resident tensor on the first call,
    # which must not land in whatever graph pool the first compiled call happens to run in.
    x = torch.zeros(8, 8, device="cuda", dtype=torch.bfloat16)
    mm_out(x, x, torch.empty_like(x), "nk")
    nt_out(x, x, torch.empty_like(x), "nk")
    torch.cuda.synchronize()
