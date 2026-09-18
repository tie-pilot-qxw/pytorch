"""DynaGraph: reuse one recorded CUDA graph across an interval of shapes.

Today cudagraph_trees keys its function cache on the exact tuple of integer
inputs, which under dynamic shapes *are* the symints, so every distinct shape
records a new graph (see ``cudagraph_trees.deferred_cudagraphify``). This module
supplies the pieces needed to record one graph per shape *interval* instead:

* extract, for every Triton kernel in the compiled wrapper, the symbolic
  expressions that drive its grid and its shape-carrying scalar arguments, plus
  the byte offset of each such argument inside the kernel parameter buffer;
* generate, compile and load a "planner" kernel that recomputes all of the above
  on device from the current symints and re-parameterizes the graph nodes in
  place, via ``cudaGraphKernelNodeSetGridDim`` / ``SetParam`` / ``SetEnabled``.

The planner is injected as the first node of the captured graph, so the update
takes effect within the same launch.

Two properties of the surrounding system shape the design:

* Parameter byte offsets cannot be computed, only queried. PyTorch never builds a
  flat parameter buffer -- every launch path uses the array-of-pointers form of
  ``cuLaunchKernel`` -- so the real layout lives in the cubin and follows the PTX
  ABI: natural C alignment, with int32 arguments *not* padded to 8 bytes. A
  kernel with two int32 arguments packs them at, say, 16 and 20, so an ``8 * i``
  formula is wrong. ``cuFuncGetParamInfo`` is the only reliable source.
* The graph's private memory pool sizes every buffer at capture time, so a graph
  must be recorded at the maximum shape of its interval. Smaller shapes then fit
  inside the same allocations.
"""

from __future__ import annotations

import ast
import ctypes
import logging
import os
import re
import subprocess
import tempfile
from typing import Any

log = logging.getLogger(__name__)

_libcuda: Any = None


def _cuda() -> Any:
    global _libcuda
    if _libcuda is None:
        _libcuda = ctypes.CDLL("libcuda.so.1")
    return _libcuda


def _param_info(func: int, index: int) -> tuple[int, int] | None:
    off, size = ctypes.c_size_t(), ctypes.c_size_t()
    rc = _cuda().cuFuncGetParamInfo(
        ctypes.c_void_p(func), ctypes.c_size_t(index),
        ctypes.byref(off), ctypes.byref(size))
    return None if rc != 0 else (off.value, size.value)


# --------------------------------------------------------------- shape buckets
def bucket_of(int_key: tuple[int, ...] | int | None, ratio: float) -> Any:
    """Map a symint tuple to its bucket id, and to the shape the graph is recorded at.

    Each dimension is bucketed independently on a geometric ladder, so a graph
    recorded at the bucket ceiling serves every shape down to ceiling / ratio.
    """
    if int_key is None:
        return None, None
    keys = (int_key,) if isinstance(int_key, int) else tuple(int_key)
    ids, tops = [], []
    for v in keys:
        if v <= 0:
            ids.append(0)
            tops.append(v)
            continue
        exp = 0
        top = 1
        while top < v:
            top = max(top + 1, int(top * ratio))
            exp += 1
        ids.append(exp)
        tops.append(top)
    bucket = tuple(ids)
    record_at = tops[0] if isinstance(int_key, int) else tuple(tops)
    return bucket, record_at


# --------------------------------------------------------------- extraction
def _split_args(argstr: str) -> list[str]:
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in argstr:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return [a for a in out if a and "=" not in a.split("(")[0]]


def _parse_run_calls(src: str) -> dict[str, list[str]]:
    """Positional arguments of every ``<kernel>.run(...)`` in the wrapper.

    Sizes are not named variables in general; they are inlined at the call site::

        triton_per_..._1.run(buf4, arg1_1, buf0, s77, 256, stream=raw_stream0)
        #                    in_out  in_ptr0 in_ptr1 xnumel r0_numel

    A kernel whose grid_type is FixedGrid additionally passes grid_0/1/2 right
    after its arguments, so the grid is recoverable the same way.
    """
    calls: dict[str, list[str]] = {}
    for m in re.finditer(r"(\w+)\.run\(", src):
        name, i = m.group(1), m.end()
        depth, j = 1, i
        while j < len(src) and depth:
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
            j += 1
        calls[name] = _split_args(src[i : j - 1])
    return calls


def _resolve(expr: str, src: str, symbols: frozenset[str], depth: int = 8) -> str:
    """Expand a named intermediate to its definition, stopping at symbols.

    Both forms occur in the same wrapper: some sizes are inlined at the call site,
    others go through a named variable::

        triton_red_..._sum_2_r0_numel = s77

    Reading only the call site yields the *name*, which contains no symbol and so
    would never be recognized as shape-dependent -- leaving that kernel's
    reduction extent frozen at the shape the graph was recorded with.

    Symbols terminate the expansion. The wrapper also contains ``s77 = arg2_1``,
    i.e. how the symbol is read out of an input tensor; expanding that would
    replace the symbol with a buffer name and lose the dependency entirely.
    """
    for _ in range(depth):
        e = expr.strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", e):
            return expr
        if e in symbols:
            return e
        m = re.search(rf"^\s*{re.escape(e)}\s*=\s*(.+?)\s*$", src, re.M)
        if not m:
            return expr
        expr = m.group(1)
    return expr


def _is_symbolic(expr: Any) -> bool:
    return bool(expr) and bool(re.search(r"\bs\d+\b", str(expr)))


def extract_kernel_table(source_code: str, call_globals: dict[str, Any]) -> Any:
    """Return (kernels, symbols) for the kernels this wrapper actually launches.

    ``call_globals`` is the wrapper module's namespace. Its CachingAutotuner
    values are the kernels that enter the graph; the autotuner *candidates* live
    in other modules and must not be collected.
    """
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    run_args = _parse_run_calls(source_code)
    symbols = frozenset(re.findall(r"\b(s\d+)\b", source_code))

    kernels = []
    for gname, obj in call_globals.items():
        if not isinstance(obj, CachingAutotuner):
            continue
        meta = obj.inductor_meta or {}
        sig = (obj.triton_meta or {}).get("signature", {})
        args = [k for k, v in sig.items() if v != "constexpr"]
        # Only integer scalars are patched. Pointers live at fixed addresses in
        # the graph's private pool; expanding their names would yield an
        # empty_strided_cuda(...) expression that merely looks shape-dependent.
        scalar = {k: isinstance(sig.get(k), str) and not sig[k].startswith("*")
                  for k in args}

        blocks = {}
        for lr in getattr(obj, "launchers", []) or []:
            cfg = getattr(lr, "config", None)
            for bk in ("XBLOCK", "YBLOCK", "R0_BLOCK"):
                if cfg is not None and bk in getattr(cfg, "kwargs", {}):
                    blocks[bk] = cfg.kwargs[bk]

        func = None
        for cr in getattr(obj, "compile_results", []) or []:
            k = getattr(cr, "kernel", None)
            func = getattr(k, "function", None) or next(
                iter(getattr(k, "functions", {}).values()), None)
            if func:
                break
        if not func:
            return None, None  # kernel not statically launched; bail out

        pos = run_args.get(gname) or []
        exprs = {
            nm: _resolve(pos[i], source_code, symbols)
            for i, nm in enumerate(args)
            if scalar.get(nm) and i < len(pos)
        }
        # Pointer arguments are recorded too, by the buffer name they carry. They
        # are what the arena re-layout patches: which buffers share storage is
        # fixed at compile time, but where each one sits is not, once the sizes
        # are only known at replay.
        ptrs = {
            nm: pos[i].strip()
            for i, nm in enumerate(args)
            if not scalar.get(nm) and i < len(pos)
        }
        offsets = {nm: _param_info(func, args.index(nm))
                   for nm in list(exprs) + list(ptrs)}
        if any(v is None for v in offsets.values()):
            return None, None

        grid = None
        if meta.get("grid_type") == "FixedGrid":
            grid = [_resolve(e, source_code, symbols)
                    for e in pos[len(args) : len(args) + 3]]
            if len(grid) != 3:
                return None, None

        kernels.append(dict(gname=gname, name=meta.get("kernel_name", gname),
                            exprs=exprs, ptrs=ptrs, offsets=offsets,
                            blocks=blocks, grid=grid))

    # launch order == order of the .run( call sites in the wrapper
    order = {k["gname"]: source_code.find(f"{k['gname']}.run(") for k in kernels}
    kernels.sort(key=lambda k: order[k["gname"]] if order[k["gname"]] >= 0 else 1 << 30)
    return kernels, sorted(symbols)


# --------------------------------------------------------------- codegen
def _expr_to_c(expr: str, sym_index: dict[str, int]) -> str:
    """Translate a wrapper expression to C.

    Parsed rather than string-substituted: these expressions contain Python's
    ``//``, whose behaviour on negative operands differs from C's ``/``, and
    textual replacement also loses operator precedence.
    """

    def go(n: ast.AST) -> str:
        if isinstance(n, ast.Expression):
            return go(n.body)
        if isinstance(n, ast.Constant):
            return f"(int64_t){int(n.value)}"
        if isinstance(n, ast.Name):
            if n.id not in sym_index:
                raise KeyError(f"unknown symbol {n.id}")
            return f"S({sym_index[n.id]})"
        if isinstance(n, ast.BinOp):
            a, b = go(n.left), go(n.right)
            op = n.op
            if isinstance(op, ast.Add):
                return f"({a} + {b})"
            if isinstance(op, ast.Sub):
                return f"({a} - {b})"
            if isinstance(op, ast.Mult):
                return f"({a} * {b})"
            if isinstance(op, (ast.FloorDiv, ast.Div)):
                return f"dg_floordiv({a}, {b})"
            if isinstance(op, ast.Mod):
                return f"dg_mod({a}, {b})"
            raise NotImplementedError(f"operator {type(op).__name__}")
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub):
                return f"(-{go(n.operand)})"
            if isinstance(n.op, ast.UAdd):
                return go(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id in ("min", "max") and len(n.args) == 2:
                return f"dg_{n.func.id}({go(n.args[0])}, {go(n.args[1])})"
        raise NotImplementedError(f"unsupported expression node {type(n).__name__}")

    return go(ast.parse(expr.strip(), mode="eval"))


_PLANNER_TEMPLATE = r"""
// Generated by torch/_inductor/dynagraph.py -- do not edit.
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ int64_t dg_floordiv(int64_t a, int64_t b) {
  int64_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) --q; return q;
}
__device__ __forceinline__ int64_t dg_mod(int64_t a, int64_t b) {
  int64_t r = a % b; if (r != 0 && ((r < 0) != (b < 0))) r += b; return r;
}
__device__ __forceinline__ int64_t dg_min(int64_t a, int64_t b){ return a<b?a:b; }
__device__ __forceinline__ int64_t dg_max(int64_t a, int64_t b){ return a>b?a:b; }

#define S(i) (ctx[(i)])
__device__ __forceinline__ int64_t dg_eval(int32_t k, const int64_t* __restrict__ ctx) {
  switch (k) {
/*@EXPRS@*/
    default: return 0;
  }
}
#undef S

// One thread per node. Both the grid and the shape-carrying scalar arguments
// must be updated: a stale grid launches the wrong number of blocks, while stale
// arguments leave the kernel's own bounds checks referring to the recorded shape,
// which silently corrupts the tail block.
extern "C" __global__ void dynagraph_planner(
    const cudaGraphDeviceNode_t* __restrict__ handles,
    const int64_t* __restrict__ ctx,
    char* __restrict__ arena,
    const int64_t* __restrict__ slot_off)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= /*@N@*/) return;

  int64_t gx = 1, gy = 1, gz = 1;
  switch (i) {
/*@GRID@*/
    default: break;
  }
  if (gx <= 0 || gy <= 0 || gz <= 0) {
    // A grid dimension of zero is cudaErrorInvalidArgument, so an empty tensor
    // has to be expressed by disabling the node rather than by a zero grid.
    cudaGraphKernelNodeSetEnabled(handles[i], 0);
    return;
  }
  cudaGraphKernelNodeSetEnabled(handles[i], 1);
  cudaGraphKernelNodeSetGridDim(handles[i],
      dim3((unsigned)gx, (unsigned)gy, (unsigned)gz));

  switch (i) {
/*@PARAMS@*/
    default: break;
  }

  // Repoint buffer arguments into the arena. Needed whenever no single shape
  // dominates the interval: with a fixed total split over a varying number of
  // samples, one buffer grows as another shrinks, so recording at a maximum
  // cannot cover both and the layout has to be redone per replay.
  if (arena != nullptr) {
    switch (i) {
/*@PTRS@*/
      default: break;
    }
  }
}
"""


def generate_planner(kernels: list[dict[str, Any]], symbols: list[str],
                     slot_of: dict[str, int] | None = None) -> str:
    sym_index = {s: i for i, s in enumerate(symbols)}
    exprs: list[str] = []

    def idx(e: str) -> int:
        c = _expr_to_c(e, sym_index)
        if c not in exprs:
            exprs.append(c)
        return exprs.index(c)

    grid_cases, param_cases = [], []
    for i, k in enumerate(kernels):
        if k["grid"]:
            gx, gy, gz = (idx(e) for e in k["grid"])
            grid_cases.append(
                f"    case {i}: gx=dg_eval({gx},ctx); gy=dg_eval({gy},ctx); "
                f"gz=dg_eval({gz},ctx); break;  // {k['name']}")
        else:
            xn = k["exprs"].get("xnumel")
            if xn is None:
                raise RuntimeError(f"{k['name']}: neither explicit grid nor xnumel")
            blk = k["blocks"].get("XBLOCK", 1)
            grid_cases.append(
                f"    case {i}: gx=dg_floordiv(dg_eval({idx(xn)},ctx) + {blk} - 1,"
                f" {blk}); break;  // {k['name']}")

        patches = []
        for nm, e in k["exprs"].items():
            if not _is_symbolic(e):
                continue
            off, size = k["offsets"][nm]
            ct = "int32_t" if size == 4 else "int64_t"
            patches.append(
                f"      {{ {ct} v = ({ct})dg_eval({idx(e)},ctx); "
                f"cudaGraphKernelNodeSetParam(handles[i], {off}, &v, {size}); }}"
                f"  // {nm} = {e}")
        if patches:
            param_cases.append(f"    case {i}:\n" + "\n".join(patches) + "\n      break;")

    out = _PLANNER_TEMPLATE
    # Substituted rather than %-formatted: the generated C contains the modulo
    # operator, which collides with %-format placeholders.
    for tag, val in (
        ("/*@EXPRS@*/", "\n".join(f"    case {i}: return {c};"
                                  for i, c in enumerate(exprs))),
        ("/*@N@*/", str(len(kernels))),
        ("/*@GRID@*/", "\n".join(grid_cases)),
        ("/*@PARAMS@*/", "\n".join(param_cases)),
        ("/*@PTRS@*/", generate_pointer_patches(kernels, slot_of or {})),
    ):
        out = out.replace(tag, val)
    return out


def compile_planner(src: str, arch: str | None = None) -> int | None:
    """Compile to a cubin and load it; return the CUfunction, or None on failure.

    ``-rdc`` is deliberately not used. The device-side graph node APIs do not
    require relocatable device code, so a plain cubin can be loaded directly and
    the runtime-linking dance (cuLinkCreate/AddData/Complete against cudadevrt)
    is unnecessary.
    """
    import torch

    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}" + ("a" if (major, minor) >= (9, 0) else "")
    with tempfile.TemporaryDirectory() as wd:
        cu = os.path.join(wd, "planner.cu")
        cubin = os.path.join(wd, "planner.cubin")
        with open(cu, "w") as fh:
            fh.write(src)
        r = subprocess.run(["nvcc", f"-arch={arch}", "-cubin", "-o", cubin, cu],
                           capture_output=True, text=True)
        if r.returncode != 0:
            log.warning("DynaGraph planner failed to compile: %s", r.stderr[-800:])
            return None
        mod = ctypes.c_void_p()
        if _cuda().cuModuleLoad(ctypes.byref(mod), cubin.encode()) != 0:
            log.warning("DynaGraph planner cuModuleLoad failed")
            return None
    fn = ctypes.c_void_p()
    if _cuda().cuModuleGetFunction(
            ctypes.byref(fn), mod, b"dynagraph_planner") != 0:
        log.warning("DynaGraph planner cuModuleGetFunction failed")
        return None
    return fn.value


def launch_planner(func: int, n_nodes: int, handles_ptr: int, ctx_ptr: int,
                   stream: int) -> None:
    a0, a1 = ctypes.c_void_p(handles_ptr), ctypes.c_void_p(ctx_ptr)
    params = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(a0), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(a1), ctypes.c_void_p))
    block = 128
    rc = _cuda().cuLaunchKernel(
        ctypes.c_void_p(func), (n_nodes + block - 1) // block, 1, 1,
        block, 1, 1, 0, ctypes.c_void_p(stream), params, None)
    if rc != 0:
        raise RuntimeError(f"DynaGraph planner launch failed: {rc}")


# --------------------------------------------------------------- safety check
def _eval_int(expr: str, env: dict[str, int]) -> int | None:
    """Evaluate an arithmetic expression over symbol values, or None if it is
    not a plain arithmetic expression."""
    try:
        node = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    def go(n: ast.AST) -> int | None:
        if isinstance(n, ast.Expression):
            return go(n.body)
        if isinstance(n, ast.Constant):
            return int(n.value) if isinstance(n.value, int) else None
        if isinstance(n, ast.Name):
            return env.get(n.id)
        if isinstance(n, ast.BinOp):
            a, b = go(n.left), go(n.right)
            if a is None or b is None:
                return None
            o = n.op
            if isinstance(o, ast.Add):
                return a + b
            if isinstance(o, ast.Sub):
                return a - b
            if isinstance(o, ast.Mult):
                return a * b
            if isinstance(o, (ast.FloorDiv, ast.Div)):
                return a // b if b else None
            if isinstance(o, ast.Mod):
                return a % b if b else None
            return None
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = go(n.operand)
            return None if v is None else -v
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in (
                "min", "max"):
            vs = [go(a) for a in n.args]
            if any(v is None for v in vs):
                return None
            return min(vs) if n.func.id == "min" else max(vs)  # type: ignore[type-var]
        return None

    return go(node)


def _find_allocations(src: str) -> list[tuple[str, list[str], list[str], str]]:
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
        args = _split_args(src[i : j - 1])
        if len(args) < 2:
            continue
        def tup(a: str) -> list[str]:
            a = a.strip()
            return _split_args(a[1:-1]) if a.startswith("(") else []
        dtype = args[2].strip() if len(args) > 2 else "torch.float32"
        out.append((name, tup(args[0]), tup(args[1]), dtype))
    return out


def buffer_dominates(
    source_code: str,
    record_shape: dict[str, int],
    sample_shapes: list[dict[str, int]],
) -> tuple[bool, str]:
    """Check that recording at ``record_shape`` allocates enough for every sample.

    Serving one recorded graph across a set of shapes rests on an assumption that
    is easy to leave unstated: the recording shape's allocation must dominate every
    other shape's, buffer by buffer. If some buffer were larger elsewhere it would
    run past the extent fixed for it at capture time and quietly corrupt its
    neighbour, with no error raised.

    ``sample_shapes`` must describe the shape space as it will actually be used,
    with each symbol given independently. An earlier version of this check varied
    all symbols together, which made it blind to exactly the case it was meant to
    catch: with a fixed total split over a varying number of samples, one symbol
    rises as another falls, so no point on the diagonal exhibits the conflict and
    the check passed on a shape space it should have rejected.

    A negative result is not fatal. The caller can fall back to recording per
    shape, or to laying the buffers out in an arena on each replay.
    """
    allocs = _find_allocations(source_code)
    if not allocs:
        return True, ""

    for name, sizes, strides, _dtype in allocs:
        def span(env: dict[str, int]) -> int | None:
            sz = [_eval_int(e, env) for e in sizes]
            st = [_eval_int(e, env) for e in strides]
            if any(v is None for v in sz) or any(v is None for v in st):
                return None
            return 1 + sum((a - 1) * b for a, b in zip(sz, st))

        at_record = span(record_shape)
        if at_record is None:
            continue  # cannot evaluate; claim nothing about this buffer
        for env in sample_shapes:
            here = span(env)
            if here is not None and here > at_record:
                return False, (
                    f"buffer {name} needs {here} elements at {env} but only "
                    f"{at_record} would be allocated at {record_shape}")
    return True, ""


def buffers_are_monotonic(
    source_code: str, symbols: list[str], record_at: int, probes: int = 24
) -> tuple[bool, str]:
    """Single-symbol convenience wrapper over :func:`buffer_dominates`.

    Only sound when the shape space really is one-dimensional, i.e. every symbol
    moves together. Use :func:`buffer_dominates` with explicit per-symbol samples
    otherwise.
    """
    if record_at <= 1:
        return True, ""
    step = max(1, record_at // probes)
    samples = [dict.fromkeys(symbols, m)
               for m in sorted({record_at, 1, *range(1, record_at, step)})]
    return buffer_dominates(source_code, dict.fromkeys(symbols, record_at), samples)


# --------------------------------------------------------------- arena layout
def plan_slots(
    lifetimes: dict[str, tuple[int, int]], allocated: list[str]
) -> tuple[list[int], int]:
    """Assign each allocated buffer to a slot; buffers sharing a slot never overlap.

    Which buffers may share storage is fixed at compile time, because the
    allocation order and the lifetimes are properties of the schedule and do not
    change with shape. Only the size of each slot varies at runtime. That split is
    what keeps the device side cheap: a slot's size is the maximum over the buffers
    assigned to it, and the offsets are a prefix sum, so the runtime work is one
    O(number of slots) sweep rather than general allocation.

    Lifetimes form an interval graph, for which colouring greedily in order of
    start time is optimal, so this uses the minimum number of slots.

    Returns (slot index per buffer in ``allocated`` order, slot count).
    """
    order = sorted(range(len(allocated)),
                   key=lambda i: lifetimes.get(allocated[i], (0, 0))[0])
    slot_free_at: list[int] = []   # step at which each slot becomes reusable
    assign = [0] * len(allocated)
    for i in order:
        start, end = lifetimes.get(allocated[i], (0, -1))
        if end < 0:
            end = 1 << 30          # graph output: lives past the whole schedule
        placed = False
        for s, free_at in enumerate(slot_free_at):
            if free_at <= start:
                slot_free_at[s] = end
                assign[i] = s
                placed = True
                break
        if not placed:
            slot_free_at.append(end)
            assign[i] = len(slot_free_at) - 1
    return assign, len(slot_free_at)


def buffer_size_exprs(source_code: str) -> dict[str, str]:
    """Symbolic element-span of every ``empty_strided_cuda`` buffer, by name.

    Sizes have to come from the wrapper rather than from Inductor's memory
    planning: ``memory.compute_size_for_scheduler_buffer`` is typed as returning
    ints and does in fact return them, already specialized with the hint value, so
    nothing symbolic survives to that layer. The lifetimes from the same pass are
    still usable, being independent of shape.

    The span is ``1 + sum((size_i - 1) * stride_i)`` rather than the product of the
    sizes, so that a non-contiguous layout is accounted for.
    """
    out: dict[str, tuple[str, int]] = {}
    for name, sizes, strides, dtype in _find_allocations(source_code):
        if not sizes or len(sizes) != len(strides):
            continue
        terms = [f"(({a}) - 1) * ({b})" for a, b in zip(sizes, strides)]
        span = "1 + " + " + ".join(terms) if terms else "1"
        out[name] = (span, _ITEMSIZE.get(dtype, 4))
    return out


# Element sizes for the dtypes Inductor emits in empty_strided_cuda calls.
_ITEMSIZE = {
    "torch.float32": 4, "torch.float": 4, "torch.float64": 8, "torch.double": 8,
    "torch.float16": 2, "torch.half": 2, "torch.bfloat16": 2,
    "torch.int64": 8, "torch.long": 8, "torch.int32": 4, "torch.int": 4,
    "torch.int16": 2, "torch.int8": 1, "torch.uint8": 1, "torch.bool": 1,
    "torch.float8_e4m3fn": 1, "torch.float8_e5m2": 1,
}


_LAYOUT_TEMPLATE = r"""
// Recomputes the arena layout for the current shape. Separate from the planner
// because the offsets are a prefix sum over slots, which does not fit the
// planner's one-thread-per-node shape; both run as the first nodes of the graph.
extern "C" __global__ void dynagraph_layout(
    const int64_t* __restrict__ ctx, int64_t* __restrict__ slot_off)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
#define S(i) (ctx[(i)])
  int64_t sz[/*@NSLOTS@*/];
  for (int i = 0; i < /*@NSLOTS@*/; ++i) sz[i] = 0;

  // A slot must hold the largest of the buffers assigned to it.
/*@SLOTSIZES@*/

  // 256-byte alignment, matching what the caching allocator hands out, so that
  // vectorized accesses and TMA descriptors keep the alignment they were
  // compiled for.
  int64_t acc = 0;
  for (int i = 0; i < /*@NSLOTS@*/; ++i) {
    slot_off[i] = acc;
    acc += (sz[i] + 255) & ~(int64_t)255;
  }
  slot_off[/*@NSLOTS@*/] = acc;   // total, for the caller to check against the arena
#undef S
}
"""


def generate_layout(
    buf_sizes: dict[str, tuple[str, int]],
    slot_of: dict[str, int],
    n_slots: int,
    symbols: list[str],
) -> str:
    sym_index = {s: i for i, s in enumerate(symbols)}
    lines = []
    for name, (span, itemsize) in sorted(buf_sizes.items()):
        if name not in slot_of:
            continue
        c = _expr_to_c(span, sym_index)
        lines.append(
            f"  {{ int64_t b = ({c}) * {itemsize}; "
            f"if (b > sz[{slot_of[name]}]) sz[{slot_of[name]}] = b; }}"
            f"  // {name}")
    out = _LAYOUT_TEMPLATE
    for tag, val in (("/*@NSLOTS@*/", str(n_slots)),
                     ("/*@SLOTSIZES@*/", "\n".join(lines))):
        out = out.replace(tag, val)
    return out


def generate_pointer_patches(
    kernels: list[dict[str, Any]], slot_of: dict[str, int]
) -> str:
    """Per-node ``case`` bodies that repoint each buffer argument into the arena."""
    cases = []
    for i, k in enumerate(kernels):
        body = []
        for nm, buf in (k.get("ptrs") or {}).items():
            if buf not in slot_of:
                continue  # an input or a graph output, not arena-managed
            off, size = k["offsets"][nm]
            if size != 8:
                continue
            body.append(
                f"      {{ char* p = arena + slot_off[{slot_of[buf]}]; "
                f"cudaGraphKernelNodeSetParam(handles[i], {off}, &p, 8); }}"
                f"  // {nm} = {buf}")
        if body:
            cases.append(f"    case {i}:\n" + "\n".join(body) + "\n      break;")
    return "\n".join(cases)
