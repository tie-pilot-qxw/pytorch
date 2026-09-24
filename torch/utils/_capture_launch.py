"""The kernel launches an operator makes, declared by the operator.

A system that serves many shapes from one CUDA graph keeps each kernel as a
node and rewrites its parameters per call: grid, shared memory, arguments.
For a kernel whose launch it generated itself (Inductor's Triton kernels) it
knows how every one of those follows from the shape. For an operator it did
not generate -- a custom op wrapping a hand-written CUDA kernel, a JIT library
-- it does not, and the only other route is to run the operator again at every
new shape and capture what it launches. That means running the model's Python
up to that operator, which costs as much as capturing the whole graph again.

So the operator declares it. `register` takes the operator's qualified name
and up to three functions, each called with the operator's own arguments:

``launches(*args, **kwargs) -> list[Launch]``
    Every kernel this call launches, in order: which kernel, grid, block,
    dynamic shared memory, and the value of each kernel parameter. Pure host
    code: it reads shapes, strides, addresses and whatever host state the
    operator itself reads, and launches nothing. This is the whole contract.

``key(*args, **kwargs) -> Hashable`` (optional)
    What besides the arguments' shapes and addresses the launch depends on --
    typically state the operator reads from a context rather than from its
    arguments, such as an engine's per-step metadata. Called on every call,
    so it must be cheap; the consumer calls `launches` again only when it
    moves. Without one, the launch is taken to follow from the arguments.

``variant(*args, **kwargs) -> Hashable`` (optional)
    Which kernels the call takes: the host inputs the operator's dispatch
    reads to pick a template instance, a config, a number of launches -- an
    M bucket, an autotune key, a head dim. Everything that needs `prepare`
    again when it moves, and nothing that does not. Without one, every call
    is the same variant.

``prepare(*args, **kwargs) -> None`` (optional)
    One-time work for a variant that must not happen inside a capture:
    compiling a JIT kernel, autotuning, initialising a library. The consumer
    calls it once per variant, always outside any capture -- before the
    first capture of the operator and before the first launch of any new
    variant. Without one, the consumer runs the operator once to the same
    end, as an engine's warm-up run would.

This is the lifecycle serving engines already give their attention backends
(SGLang: `init_cuda_graph_state` for resident buffers, warm-up runs and
`post_warmup_hook` before `init_forward_metadata_capture_cuda_graph` at
capture and `init_forward_metadata_replay_cuda_graph` before each replay),
made per operator and per variant rather than per captured batch size:
`prepare` is the warm-up, `launches` at capture is the capture-time metadata,
`launches` at a new shape or `key` is the replay-time update.

An operator whose host code is the only description of its launches -- one
that fills a large parameter struct, or picks a kernel through heuristics it
does not expose -- can declare ``launches=recorded(op)``: each call's launches
are then whatever one call of `op` launches on a side stream, recorded and not
run. That runs the operator's host code, not the model's, and not the GPU.

`Launch.kernel` names a kernel by any hashable the operator likes (a template
instantiation, a config tuple), or by a driver function handle when the
operator has one. A name is resolved to a function handle by the consumer,
from a capture of one real call: recorded, not run. The same capture checks
the declaration: a launch whose grid or parameter bytes differ from what the
operator actually launched is refused rather than trusted.

An operator that declares nothing is served however the consumer serves an
opaque call. Modelled on `torch.utils._capture_deps`.
"""

import functools
import struct
from collections.abc import Callable, Hashable
from typing import Any, NamedTuple


__all__ = [
    "Declaration",
    "Launch",
    "Mismatch",
    "launches_of",
    "lookup",
    "pack",
    "record",
    "recorded",
    "register",
    "triton_launch",
]


class Launch(NamedTuple):
    """One kernel launch, as the operator would make it.

    `args` holds one value per kernel parameter, in declaration order: an int
    (an integer or an address), a float, a tensor (its address), None (a null
    pointer), or bytes (a by-value struct, exactly as wide as the parameter).
    Each is packed to the width the kernel itself reports for that parameter.
    `cluster` is the cluster shape, or None for a launch without one.
    """

    kernel: Hashable
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    smem: int
    args: tuple[Any, ...]
    cluster: tuple[int, int, int] | None = None
    # Whatever must outlive every use of this launch: the buffers its
    # parameters point into when the declaration allocated them.
    owner: Any = None


class Declaration(NamedTuple):
    launches: Callable[..., list[Launch]]
    key: Callable[..., Hashable] | None
    prepare: Callable[..., None] | None
    # Whether a declaration's parameter bytes can be compared with a
    # recording. Not for a by-value struct: its padding holds whatever was on
    # the stack, and buffers the operator allocates for itself sit at another
    # address on every call.
    exact: bool = True
    variant: Callable[..., Hashable] | None = None
    # With `share`, calls whose arguments have the same geometry and the same
    # non-tensor values are taken to launch the same thing up to addresses
    # (the same operator in every layer): a consumer declares once and moves
    # the addresses for the rest. `template` adds what else the launch
    # depends on and implies `share`; `sources` names the tensors besides the
    # arguments whose addresses differ between calls (a layer's cache).
    # Worth it only for an expensive declaration: moving the addresses costs
    # about what a library's own describe does (~20 us for DeepGEMM).
    template: Callable[..., Hashable] | None = None
    sources: Callable[..., Any] | None = None
    share: bool = False
    # The same declaration for a caller in C: the address of a function with
    # the launch-describe ABI below, and `c_statics(*args, **kwargs)` (tensor
    # arguments passed as None) giving the bytes it takes for the call's
    # non-tensor arguments. A consumer that has the operands as raw pointers
    # and geometry (DynaGraph's C++ runtime) then never goes through Python.
    c_describe: int | None = None
    c_statics: Callable[..., bytes] | None = None


# The launch-describe ABI, v1 (C):
#
#   struct dg_operand { uint64_t ptr; int32_t dtype; int32_t ndim;
#                       int64_t sizes[8]; int64_t strides[8]; };
#   struct dg_launch  { uint64_t func; uint32_t grid[3], block[3];
#                       uint32_t smem, cluster, pdl, nargs; };
#   int describe(const dg_operand* ops, int nops, const char* statics,
#                dg_launch* out, int max_launches, char* args, int64_t args_cap,
#                uint32_t* arg_sizes, int sizes_cap, char* err, int err_cap);
#
# `ops` are the call's tensor arguments in order (dtype a c10::ScalarType,
# strides in elements); nothing is launched. It returns the number of
# launches, their parameter values back to back in `args` with one size per
# parameter in `arg_sizes`; -1 when the operator raised (message in `err`),
# -2 when a buffer is too small.


_registry: dict[str, Declaration] = {}


def register(
    qualname: str,
    launches: Callable[..., list[Launch]],
    *,
    key: Callable[..., Hashable] | None = None,
    prepare: Callable[..., None] | None = None,
    exact: bool = True,
    variant: Callable[..., Hashable] | None = None,
    template: Callable[..., Hashable] | None = None,
    sources: Callable[..., Any] | None = None,
    share: bool = False,
    c_describe: int | None = None,
    c_statics: Callable[..., bytes] | None = None,
) -> None:
    """Declare the launches of operator `qualname` ("namespace::name")."""
    _registry[qualname] = Declaration(
        launches,
        key,
        prepare,
        exact,
        variant,
        template,
        sources,
        share or template is not None,
        c_describe,
        c_statics,
    )


def lookup(qualname: str) -> Declaration | None:
    return _registry.get(qualname)


def launches_of(qualname: str, *args: Any, **kwargs: Any) -> list[Launch]:
    """What a registered operator declares for these arguments.

    For an operator whose launches are those of another one it calls, so its
    declaration can resolve its own context and hand the rest down.
    """
    d = _registry.get(qualname)
    if d is None:
        raise Mismatch(f"{qualname} declares no launches")
    return d.launches(*args, **kwargs)


class Mismatch(Exception):
    """A declared launch disagrees with what the operator launched."""


class Recorded(NamedTuple):
    """A launch as a capture recorded it."""

    func: int
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    smem: int
    cluster: tuple[int, int, int] | None
    params: list[bytes]


@functools.cache
def param_sizes(func: int) -> list[tuple[int, int]]:
    """(offset, size) of every parameter of a kernel, from the driver."""
    from cuda.bindings import driver as cu

    from torch.cuda._utils import _check_cuda_bindings as ck

    out = []
    while True:
        try:
            off, size = ck(cu.cuFuncGetParamInfo(func, len(out)))
        except RuntimeError:
            # Past the last parameter the driver answers invalid value.
            return out
        out.append((int(off), int(size)))


def _cluster_of(node: Any) -> tuple[int, int, int] | None:
    from cuda.bindings import driver as cu

    from torch.cuda._utils import _check_cuda_bindings as ck

    attr = cu.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    v = ck(cu.cuGraphKernelNodeGetAttribute(node, attr))
    c = (int(v.clusterDim.x), int(v.clusterDim.y), int(v.clusterDim.z))
    return None if c in ((0, 0, 0), (1, 1, 1)) else c


def read_node(node: Any) -> Recorded | None:
    """A kernel node's launch, or None when the node is not a kernel."""
    import ctypes as ct

    from cuda.bindings import driver as cu

    from torch.cuda._utils import _check_cuda_bindings as ck

    n = int(node)
    if ck(cu.cuGraphNodeGetType(n)) != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
        return None
    p = ck(cu.cuGraphKernelNodeGetParams(n))
    func = int(p.func)
    sizes = param_sizes(func)
    kp = int(p.kernelParams or 0)
    blob = 0
    if not kp and int(p.extra or 0):
        # `extra` form: a null-terminated list of (key, value) pairs, key 1
        # giving the packed buffer every parameter sits in at its offset.
        at = int(p.extra)
        while True:
            k = ct.c_void_p.from_address(at).value or 0
            if not k:
                break
            v = ct.c_void_p.from_address(at + 8).value or 0
            if k == 1:
                blob = v
            at += 16
    params = []
    for j, (off, size) in enumerate(sizes):
        src = (ct.c_void_p.from_address(kp + 8 * j).value or 0) if kp else blob + off
        params.append(ct.string_at(src, size) if src else b"\0" * size)
    return Recorded(
        func,
        (int(p.gridDimX), int(p.gridDimY), int(p.gridDimZ)),
        (int(p.blockDimX), int(p.blockDimY), int(p.blockDimZ)),
        int(p.sharedMemBytes),
        _cluster_of(n),
        params,
    )


def _record_graph(
    fn: Callable[..., Any], args: Any, kwargs: Any
) -> tuple[list[Recorded], Any]:
    """Capture one call of `fn` on a side stream; its launches and the graph.

    Through torch's capture, so what the call allocates comes from a graph
    pool and stays reserved while the returned graph lives -- a recorded
    launch may point into it. A private pool per recording: a shared one is
    released as soon as no graph uses it, and most recordings are thrown
    away straight after they are read. Relaxed mode, so it can be taken
    while another capture is under way.
    """
    from cuda.bindings import runtime as cr

    import torch
    from torch.cuda._utils import _check_cuda_bindings as ck

    g = torch.cuda.CUDAGraph(keep_graph=True)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g.capture_begin(capture_error_mode="relaxed")
        try:
            fn(*args, **kwargs)
        finally:
            g.capture_end()
    raw = g.raw_cuda_graph()
    n = int(ck(cr.cudaGraphGetNodes(raw))[1])
    nodes = ck(cr.cudaGraphGetNodes(raw, n))[0] if n else []
    out = []
    for nd in _chain_order(nodes):
        r = read_node(nd)
        if r is None:
            raise Mismatch("the operator's capture holds a node that is not a kernel")
        out.append(r)
    return out, g


def _record_raw(fn: Callable[..., Any], args: Any, kwargs: Any) -> list[Recorded]:
    """`_record_graph` for an operator that allocates nothing while it
    launches: a bare stream capture, read and destroyed at once. About a
    quarter of the cost, as there is no graph pool to set up and none to keep.
    """
    from cuda.bindings import runtime as cr

    import torch
    from torch.cuda._utils import _check_cuda_bindings as ck

    s = _side_stream()
    s.wait_stream(torch.cuda.current_stream())
    mode = cr.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed
    with torch.cuda.stream(s):
        ck(cr.cudaStreamBeginCapture(s.cuda_stream, mode))
        try:
            fn(*args, **kwargs)
        finally:
            g = ck(cr.cudaStreamEndCapture(s.cuda_stream))
    try:
        n = int(ck(cr.cudaGraphGetNodes(g))[1])
        nodes = ck(cr.cudaGraphGetNodes(g, n))[0] if n else []
        out = []
        for nd in _chain_order(nodes):
            r = read_node(nd)
            if r is None:
                raise Mismatch(
                    "the operator's capture holds a node that is not a kernel"
                )
            out.append(r)
        return out
    finally:
        ck(cr.cudaGraphDestroy(g))


@functools.cache
def _side_stream() -> Any:
    import torch

    return torch.cuda.Stream()


def record(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> list[Recorded]:
    """The kernel launches one call of `fn` makes, captured and not run.

    Anything but a kernel node in the capture (a memset, a copy, an event) is
    refused: a declaration only speaks for kernels.
    """
    return _record_graph(fn, args, kwargs)[0]


def _tensors(args: Any, kwargs: Any, extra: Any) -> list[Any]:
    import torch

    out = []
    for v in [*args, *kwargs.values(), *(extra or ())]:
        for x in v if isinstance(v, (list, tuple)) else (v,):
            if isinstance(x, torch.Tensor) and x.numel():
                out.append(x)
    return out


def _scalars(args: Any, kwargs: Any) -> tuple[Any, ...]:
    """The non-tensor arguments of a call, hashable, tensors marked by place."""
    import torch

    def one(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            return torch.Tensor
        if isinstance(v, (list, tuple)):
            return tuple(one(x) for x in v)
        try:
            hash(v)
        except TypeError:
            return repr(v)
        return v

    return one(tuple(args)), tuple((k, one(v)) for k, v in sorted(kwargs.items()))


def _extent(t: Any) -> tuple[int, int]:
    lo = t.data_ptr()
    span = sum((n - 1) * abs(st) for n, st in zip(t.shape, t.stride()))
    return lo, lo + (span + 1) * t.element_size()


def _bind_plan(template: list[Launch], old: list[Any]) -> list[Any] | None:
    """Where `template` holds an address inside one of `old`: per such word
    (launch, parameter, byte offset, which tensor, distance from its start).
    None when an address lies in the storage of one of them but in none of
    their extents: which tensor it follows is then not something this can
    tell, and such a template is not reused."""
    ext = [_extent(t) for t in old]
    stores = []
    for t in old:
        st = t.untyped_storage()
        stores.append((st.data_ptr(), st.data_ptr() + st.nbytes()))
    plan = []
    for n, L in enumerate(template):
        for j, b in enumerate(L.args):
            for w in range(0, len(b) - 7, 8):
                v = int.from_bytes(b[w : w + 8], "little")
                if not v:
                    continue
                hits = [(hi - lo, k) for k, (lo, hi) in enumerate(ext) if lo <= v < hi]
                if hits:
                    k = min(hits)[1]
                    plan.append((n, j, w, k, v - old[k].data_ptr()))
                elif any(lo <= v < hi for lo, hi in stores):
                    return None
    return plan


def _rebind(template: list[Launch], plan: list[Any], new: list[Any]) -> list[Launch]:
    """`template` with the addresses `plan` names moved into `new`."""
    params = [[bytearray(b) for b in L.args] for L in template]
    for n, j, w, k, delta in plan:
        params[n][j][w : w + 8] = (new[k].data_ptr() + delta).to_bytes(8, "little")
    return [
        L._replace(args=tuple(bytes(b) for b in ps)) for L, ps in zip(template, params)
    ]


def recorded(
    fn: Callable[..., Any],
    *,
    template_key: Callable[..., Hashable] | None = None,
    sources: Callable[..., Any] | None = None,
    allocates: bool = True,
) -> Callable[..., list[Launch]]:
    """A `launches` that is whatever one call of `fn` launches (see above).

    Recording costs the operator's host code and a capture, per call. With a
    `template_key` -- what the launch's structure depends on, the same for
    every layer of a model -- one recording per key stands for every call:
    the others get its launches with the addresses moved, from each tensor
    argument and each tensor `sources` names (per-call state the operator
    reads, such as a layer's cache) to the same place in theirs. Nothing
    checks a moved launch here; a consumer that verifies its replays (as
    DynaGraph does for its first shapes) is what catches a `sources` that
    left something out. `allocates=False` promises the call takes no memory
    while it launches (a GEMM writing into a given output), which lets the
    recording skip the graph pool.
    """
    templates: dict[Hashable, Any] = {}

    def fresh(args: Any, kwargs: Any) -> list[Launch]:
        if not allocates:
            rec, g = _record_raw(fn, args, kwargs), None
        else:
            rec, g = _record_graph(fn, args, kwargs)
        return [
            Launch(r.func, r.grid, r.block, r.smem, tuple(r.params), r.cluster, g)
            for r in rec
        ]

    def launches(*args: Any, **kwargs: Any) -> list[Launch]:
        if template_key is None:
            return fresh(args, kwargs)
        src = _tensors(args, kwargs, sources(*args, **kwargs) if sources else ())
        # The arguments' geometry is part of the structure whatever the
        # operator says: a template only stands for calls shaped like it.
        tk = (
            template_key(*args, **kwargs),
            tuple((tuple(t.shape), t.stride(), t.dtype) for t in src),
        )
        t = templates.get(tk)
        if t is not None and t[1] is not None and len(src) == t[2]:
            return _rebind(t[0], t[1], src)
        got = fresh(args, kwargs)
        if len(templates) >= 256:
            templates.clear()
        templates[tk] = (got, _bind_plan(got, src), len(src))
        return got

    launches.records = True  # type: ignore[attr-defined]
    return launches


def _chain_order(nodes: list[Any]) -> list[Any]:
    """Nodes of a single-stream capture in the order they were launched."""
    from cuda.bindings import runtime as cr

    from torch.cuda._utils import _check_cuda_bindings as ck

    if len(nodes) <= 1:
        return list(nodes)
    ids = {int(n): n for n in nodes}
    nxt: dict[int, int] = {}
    has_pred: set[int] = set()
    for n in nodes:
        cnt = int(ck(cr.cudaGraphNodeGetDependentNodes(n))[2])
        deps = ck(cr.cudaGraphNodeGetDependentNodes(n, cnt))[0] if cnt else []
        if len(deps) > 1:
            raise Mismatch("the operator's launches are not a single chain")
        if deps:
            nxt[int(n)] = int(deps[0])
            has_pred.add(int(deps[0]))
    roots = [i for i in ids if i not in has_pred]
    if len(roots) != 1:
        raise Mismatch("the operator's launches are not a single chain")
    out = []
    at: int | None = roots[0]
    while at is not None:
        out.append(ids[at])
        at = nxt.get(at)
    return out


def _pack_one(v: Any, size: int) -> bytes:
    import torch

    if isinstance(v, torch.Tensor):
        v = v.data_ptr()
    elif v is None:
        v = 0
    if isinstance(v, bytes):
        if len(v) != size:
            raise Mismatch(f"a {len(v)}-byte value for a {size}-byte parameter")
        return v
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, float):
        if size == 4:
            return struct.pack("<f", v)
        if size == 8:
            return struct.pack("<d", v)
        raise Mismatch(f"a float for a {size}-byte parameter")
    if isinstance(v, int):
        if size not in (1, 2, 4, 8):
            raise Mismatch(f"an int for a {size}-byte parameter")
        return (v & ((1 << (8 * size)) - 1)).to_bytes(size, "little")
    raise Mismatch(f"cannot pack a {type(v).__name__} into a kernel parameter")


def pack(launch: Launch, func: int) -> list[bytes]:
    """`launch`'s parameter values, packed to the widths `func` declares."""
    sizes = param_sizes(func)
    if len(sizes) != len(launch.args):
        raise Mismatch(
            f"{len(launch.args)} values declared for a kernel with {len(sizes)} parameters"
        )
    return [_pack_one(v, size) for v, (_off, size) in zip(launch.args, sizes)]


def _name(func: int) -> str:
    from cuda.bindings import driver as cu

    from torch.cuda._utils import _check_cuda_bindings as ck

    return ck(cu.cuFuncGetName(func)).decode()


def check(
    declared: list[Launch], recorded: list[Recorded], exact: bool = True
) -> list[int]:
    """Compare a declaration with a recording of the same call; return the
    function handle of each launch. Raises `Mismatch` naming the first
    difference. With `exact` false the parameter bytes are not compared, only
    which kernels and how they are launched."""
    if len(declared) != len(recorded):
        raise Mismatch(
            f"declared {len(declared)} launches, the operator made {len(recorded)}"
        )
    funcs = []
    for n, (d, r) in enumerate(zip(declared, recorded)):
        if (
            isinstance(d.kernel, int)
            and d.kernel != r.func
            and _name(d.kernel) != _name(r.func)
        ):
            # Another module's copy of the same kernel (a library built again
            # to describe its launches) is the same kernel; any other is not.
            raise Mismatch(
                f"launch {n}: declared {_name(d.kernel)}, launched {_name(r.func)}"
            )
        for what, a, b in (
            ("grid", tuple(d.grid), r.grid),
            ("block", tuple(d.block), r.block),
            ("smem", d.smem, r.smem),
            ("cluster", d.cluster, r.cluster),
        ):
            if a != b:
                raise Mismatch(f"launch {n}: {what} declared {a}, launched {b}")
        got = pack(d, r.func)
        for j, (x, y) in enumerate(zip(got, r.params) if exact else ()):
            if x != y:
                raise Mismatch(
                    f"launch {n} parameter {j}: declared {x.hex()}, launched {y.hex()}"
                )
        funcs.append(r.func)
    return funcs


def triton_launch(fn: Any, grid: Any, *args: Any, **kwargs: Any) -> Launch:
    """The launch `fn[grid](*args, **kwargs)` would make, without making it.

    For a hand-written Triton kernel the call site is the declaration: this
    runs Triton's own argument binding, specialization and compile-cache
    lookup (compiling on a miss, which is `prepare` work), evaluates the
    grid, and packs the parameters the way its launcher does -- every
    argument not specialized to a constant, then the global and profile
    scratch pointers. `@triton.heuristics` values are applied. An autotuned
    kernel must already have been tuned for this key (by running it once,
    outside any capture), and its chosen config is used.
    """
    from triton.runtime import driver
    from triton.runtime.autotuner import Autotuner, Heuristics

    import torch

    kwargs = dict(kwargs)
    while not hasattr(fn, "device_caches"):
        if isinstance(fn, Heuristics):
            named = {**dict(zip(fn.arg_names, args)), **kwargs}
            for name, h in fn.values.items():
                kwargs[name] = h(named)
            fn = fn.fn
        elif isinstance(fn, Autotuner):
            # The key `Autotuner.run` looks its choice up by.
            named = {**dict(zip(fn.arg_names, args)), **kwargs}
            named = {k: v for k, v in named.items() if k in fn.arg_names}
            key = [named[k] for k in fn.keys if k in named]
            key += [str(v.dtype) for v in named.values() if hasattr(v, "dtype")]
            cfg = fn.configs[0] if len(fn.configs) == 1 else fn.cache.get(tuple(key))
            if cfg is None:
                raise Mismatch(
                    f"{fn.fn} is not tuned for {tuple(key)}; prepare must run it"
                )
            kwargs.update(cfg.all_kwargs())
            fn = fn.fn
        else:
            raise Mismatch(f"cannot describe a launch of {type(fn).__name__}")
    device = driver.active.get_current_device()
    kernel = fn.run(*args, grid=grid, warmup=True, **kwargs)
    binder = fn.device_caches[device][4]
    bound, spec, _ = binder(*args, **kwargs)
    kernel._init_handles()
    md = kernel.metadata
    if getattr(md, "launch_pdl", False) or getattr(
        md, "launch_cooperative_grid", False
    ):
        raise Mismatch("PDL / cooperative launches are not declared")
    params: list[Any] = []
    for (name, v), sp in zip(bound.items(), spec):
        if sp[0] == "constexpr":
            continue
        if isinstance(v, tuple) or "tensordesc" in str(sp[0]):
            raise Mismatch(
                f"argument {name}: tuples and tensor descriptors are not declared"
            )
        params.append(v)
    g: Any = grid(bound) if callable(grid) else grid
    g = tuple(g) + (1,) * (3 - len(g))
    num_ctas = getattr(md, "num_ctas", 1)
    owner = []
    for size, align in (
        (md.global_scratch_size, md.global_scratch_align),
        (md.profile_scratch_size, md.profile_scratch_align),
    ):
        if size > 0:
            # Per launch in Triton; here one buffer owned by the launch, so
            # its address is stable for as long as the launch is.
            buf = torch.empty(
                g[0] * g[1] * g[2] * num_ctas * size + align,
                dtype=torch.uint8,
                device="cuda",
            )
            owner.append(buf)
            params.append((buf.data_ptr() + align - 1) // align * align)
        else:
            params.append(0)
    return Launch(
        int(kernel.function),
        (g[0] * num_ctas, g[1], g[2]),
        (32 * md.num_warps, 1, 1),
        md.shared,
        tuple(params),
        (num_ctas, 1, 1) if num_ctas > 1 else None,
        owner or None,
    )
