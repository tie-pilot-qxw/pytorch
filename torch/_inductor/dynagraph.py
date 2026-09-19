"""DynaGraph: serve a whole space of shapes from one recorded CUDA graph.

Today cudagraph_trees keys its function cache on the exact tuple of integer
inputs, which under dynamic shapes *are* the symints, so every distinct shape
records a new graph (see ``cudagraph_trees.deferred_cudagraphify``). This module
records one graph and re-parameterizes it per replay instead:

* extract, for every Triton kernel in the compiled wrapper, the symbolic
  expressions that drive its grid and its shape-carrying scalar arguments, plus
  the byte offset of each such argument inside the kernel parameter buffer;
* generate, compile and load a "planner" kernel that recomputes all of the above
  on device from the current symints and re-parameterizes the graph nodes in
  place, via ``cudaGraphKernelNodeSetGridDim`` / ``SetParam`` / ``SetEnabled``;
* lay every intermediate buffer out in an arena owned here, and repoint the
  kernels' buffer arguments at it, so the sizes are not frozen at capture.

The planner is injected as the first node of the captured graph, so the update
takes effect within the same launch.

Three properties of the surrounding system shape the design:

* Parameter byte offsets cannot be computed, only queried. PyTorch never builds a
  flat parameter buffer -- every launch path uses the array-of-pointers form of
  ``cuLaunchKernel`` -- so the real layout lives in the cubin and follows the PTX
  ABI: natural C alignment, with int32 arguments *not* padded to 8 bytes. A
  kernel with two int32 arguments packs them at, say, 16 and 20, so an ``8 * i``
  formula is wrong. ``cuFuncGetParamInfo`` is the only reliable source.
* The graph's private memory pool sizes every buffer at capture time. Recording
  at the largest shape of an interval is not enough to escape that: when a fixed
  total is split over a varying number of samples one buffer grows as another
  shrinks, so no single shape dominates (:func:`buffer_dominates` is the check
  that says so). Hence the arena and the per-replay re-layout.
* Graph inputs are not in the arena -- their addresses are baked into the nodes
  -- so they are held in oversized storage and a shape past that headroom retires
  the region rather than being served.

Everything here is recovered by reading the generated wrapper, so being wrong is
possible in ways that produce no error, only a tensor whose tail was never
written. Every construct that is not fully understood is refused through
:func:`_fallback`, replays are checked against eager for the first few shapes,
and a mismatch retires the region.
"""

from __future__ import annotations

import ast
import contextlib
import ctypes
import logging
import os
import re
import subprocess
import tempfile
from typing import Any

from torch.utils._ordered_set import OrderedSet


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
        ctypes.c_void_p(func),
        ctypes.c_size_t(index),
        ctypes.byref(off),
        ctypes.byref(size),
    )
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

    Trailing keyword arguments -- ``stream=raw_stream0`` -- are dropped, so that
    the length of what comes back can be compared against the kernel signature.
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
        args = _split_args(src[i : j - 1])
        while args and re.match(r"\s*\w+\s*=[^=]", args[-1]):
            args.pop()
        calls[name] = args
    return calls


def _fallback(tag: str, detail: str = "") -> bool:
    """Record why a graph is not being served, and return False for the caller.

    Every refusal goes through here so that a sweep over many models can count
    the reasons. The tag is a stable short name meant to be grepped and tallied;
    the detail is free text for whoever is reading one log.
    """
    log.info("DynaGraph fallback [%s]%s", tag, f": {detail}" if detail else "")
    return False


class Unsupported(Exception):
    """Something in the wrapper this pass does not model.

    Raised instead of skipping the construct: a kernel argument left unpatched
    keeps pointing at whatever the capture happened to allocate, which produces
    a plausible-looking wrong answer rather than an error.
    """


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
        m = re.search(rf"^\s*{re.escape(e)}\s*=\s*(.+?)\s*$", src, re.MULTILINE)
        if not m:
            return expr
        expr = m.group(1)
    return expr


def _is_symbolic(expr: Any) -> bool:
    return bool(expr) and bool(re.search(r"\bs\d+\b", str(expr)))


# Constexprs the autotuner picks. Unlike a specialized scalar these never reach
# the .run() call site, so they are excluded when lining its positionals up.
_TUNED = ("XBLOCK", "YBLOCK", "ZBLOCK", "R0_BLOCK", "R1_BLOCK", "RBLOCK")


def settled_blocks(obj: Any) -> dict[str, int] | None:
    """Block sizes of the one config a kernel will launch with, or None.

    ``CachingAutotuner.run`` narrows ``launchers`` to a single config on its
    first real call; before that the kernel still carries every candidate, and
    they disagree -- a reduction here offers XBLOCK 1, 8 and 32. Picking the
    wrong one is not a visible error: the grid formula simply launches too few
    blocks and the tail of the output keeps whatever was already in the arena.
    So the answer is only meaningful after the warmup, and a kernel that has not
    settled reports None so the graph can be refused.
    """
    launchers = getattr(obj, "launchers", None) or []
    if len(launchers) != 1:
        return None
    kwargs = getattr(getattr(launchers[0], "config", None), "kwargs", None) or {}
    return {bk: kwargs[bk] for bk in _TUNED if bk in kwargs}


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
        constants = (obj.triton_meta or {}).get("constants", {}) or {}
        # Two different orderings, and conflating them slides every argument
        # after the first specialized one onto the wrong expression.
        #
        # `args` is the cubin's parameter list, which is what cuFuncGetParamInfo
        # indexes: a constexpr is baked into the code and is not a parameter.
        # `call_order` is what the wrapper actually passes to .run(), where a
        # scalar that Inductor specialized to a constant is still present even
        # though the signature now calls it constexpr -- only the autotuned block
        # sizes drop out. Seen in the wild as
        #     signature ... ks0:i64, xnumel:constexpr, r0_numel:i32 ...
        #     .run(arg1_1, buf0, buf2, s77, 1, ..._r0_numel, stream=...)
        # where reading r0_numel off the shorter list yields the literal 1.
        args = [k for k, v in sig.items() if v != "constexpr"]
        call_order = [
            k
            for k, v in sig.items()
            if v != "constexpr" or (k in constants and k not in _TUNED)
        ]
        # Only integer scalars are patched. Pointers live at fixed addresses in
        # the graph's private pool; expanding their names would yield an
        # empty_strided_cuda(...) expression that merely looks shape-dependent.
        scalar = {
            k: isinstance(sig.get(k), str) and not sig[k].startswith("*") for k in args
        }

        blocks = settled_blocks(obj)

        func = None
        for cr in getattr(obj, "compile_results", []) or []:
            k = getattr(cr, "kernel", None)
            func = getattr(k, "function", None) or next(
                iter(getattr(k, "functions", {}).values()), None
            )
            if func:
                break
        if not func:
            return None, None  # kernel not statically launched; bail out

        pos = run_args.get(gname) or []
        n_grid = 3 if meta.get("grid_type") == "FixedGrid" else 0
        if len(call_order) != len(pos) - n_grid:
            # Nothing here can say which positional is which, and reading them
            # by a guessed offset is how a node ends up patched with another
            # argument's value.
            return None, None
        at = {nm: call_order.index(nm) for nm in args if nm in call_order}
        exprs = {
            nm: _resolve(pos[at[nm]], source_code, symbols)
            for nm in args
            if scalar.get(nm) and nm in at
        }
        # Pointer arguments are recorded too, by the buffer name they carry. They
        # are what the arena re-layout patches: which buffers share storage is
        # fixed at compile time, but where each one sits is not, once the sizes
        # are only known at replay.
        ptrs = {
            nm: pos[at[nm]].strip() for nm in args if not scalar.get(nm) and nm in at
        }
        offsets = {
            nm: _param_info(func, args.index(nm)) for nm in list(exprs) + list(ptrs)
        }
        if any(v is None for v in offsets.values()):
            return None, None

        grid = None
        if meta.get("grid_type") == "FixedGrid":
            grid = [
                _resolve(e, source_code, symbols)
                for e in pos[len(call_order) : len(call_order) + 3]
            ]
            if len(grid) != 3:
                return None, None

        kernels.append(
            dict(
                gname=gname,
                name=meta.get("kernel_name", gname),
                exprs=exprs,
                ptrs=ptrs,
                offsets=offsets,
                blocks=blocks,
                grid=grid,
                grid_type=meta.get("grid_type"),
            )
        )

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
            if not isinstance(n.value, int):
                raise Unsupported(f"non-integer constant {n.value!r}")
            return f"(int64_t){n.value}"
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


# Grid shapes that are a ceil-divide per axis. The other GridExpr subclasses in
# triton_heuristics -- cooperative reductions, split scans, combo kernels --
# derive their grid differently and are refused rather than approximated.
_GRID_AXES = {
    "Grid1D": (("xnumel", "XBLOCK"),),
    "Grid2D": (("xnumel", "XBLOCK"), ("ynumel", "YBLOCK")),
    "Grid3D": (("xnumel", "XBLOCK"), ("ynumel", "YBLOCK"), ("znumel", "ZBLOCK")),
}


def generate_planner(
    kernels: list[dict[str, Any]],
    symbols: list[str],
    slot_of: dict[str, int] | None = None,
    alias: dict[str, str] | None = None,
) -> str:
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
                f"gz=dg_eval({gz},ctx); break;  // {k['name']}"
            )
        else:
            axes = _GRID_AXES.get(k.get("grid_type") or "")
            if axes is None:
                raise Unsupported(
                    f"{k['name']}: grid type {k.get('grid_type')} is not modelled"
                )
            blocks = k["blocks"]
            if blocks is None:
                raise Unsupported(f"{k['name']}: launch config has not settled")
            parts = []
            for axis, (numel, bk) in zip(("gx", "gy", "gz"), axes):
                e = k["exprs"].get(numel)
                if e is None or bk not in blocks:
                    raise Unsupported(f"{k['name']}: no {numel}/{bk} to size a grid")
                blk = blocks[bk]
                parts.append(
                    f"{axis}=dg_floordiv(dg_eval({idx(e)},ctx) + {blk} - 1, {blk});"
                )
            grid_cases.append(
                f"    case {i}: " + " ".join(parts) + f" break;  // {k['name']}"
            )

        patches = []
        for nm, e in k["exprs"].items():
            if not _is_symbolic(e):
                continue
            off, size = k["offsets"][nm]
            ct = "int32_t" if size == 4 else "int64_t"
            patches.append(
                f"      {{ {ct} v = ({ct})dg_eval({idx(e)},ctx); "
                f"cudaGraphKernelNodeSetParam(handles[i], {off}, &v, {size}); }}"
                f"  // {nm} = {e}"
            )
        if patches:
            param_cases.append(
                f"    case {i}:\n" + "\n".join(patches) + "\n      break;"
            )

    out = _PLANNER_TEMPLATE
    # Substituted rather than %-formatted: the generated C contains the modulo
    # operator, which collides with %-format placeholders.
    for tag, val in (
        (
            "/*@EXPRS@*/",
            "\n".join(f"    case {i}: return {c};" for i, c in enumerate(exprs)),
        ),
        ("/*@N@*/", str(len(kernels))),
        ("/*@GRID@*/", "\n".join(grid_cases)),
        ("/*@PARAMS@*/", "\n".join(param_cases)),
        ("/*@PTRS@*/", generate_pointer_patches(kernels, slot_of, alias)),
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
        r = subprocess.run(
            ["nvcc", f"-arch={arch}", "-cubin", "-o", cubin, cu],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            log.warning("DynaGraph planner failed to compile: %s", r.stderr[-800:])
            return None
        mod = ctypes.c_void_p()
        if _cuda().cuModuleLoad(ctypes.byref(mod), cubin.encode()) != 0:
            log.warning("DynaGraph planner cuModuleLoad failed")
            return None
    fn = ctypes.c_void_p()
    if _cuda().cuModuleGetFunction(ctypes.byref(fn), mod, b"dynagraph_planner") != 0:
        log.warning("DynaGraph planner cuModuleGetFunction failed")
        return None
    return fn.value


def launch_planner(
    func: int,
    n_nodes: int,
    handles_ptr: int,
    ctx_ptr: int,
    stream: int,
    arena_ptr: int = 0,
    slot_off_ptr: int = 0,
) -> None:
    """Launch the planner, one thread per graph node.

    A null arena means shapes only: the generated code skips the pointer patches
    and every buffer keeps the address the capture gave it.
    """
    _launch(
        func,
        [handles_ptr, ctx_ptr, arena_ptr, slot_off_ptr],
        (n_nodes + 127) // 128,
        128,
        stream,
    )


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
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in ("min", "max")
        ):
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
            sz, st = [], []
            for e, out in ((sizes, sz), (strides, st)):
                for x in e:
                    v = _eval_int(x, env)
                    if v is None:
                        return None
                    out.append(v)
            return 1 + sum((a - 1) * b for a, b in zip(sz, st))

        at_record = span(record_shape)
        if at_record is None:
            continue  # cannot evaluate; claim nothing about this buffer
        for env in sample_shapes:
            here = span(env)
            if here is not None and here > at_record:
                return False, (
                    f"buffer {name} needs {here} elements at {env} but only "
                    f"{at_record} would be allocated at {record_shape}"
                )
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
    samples = [
        dict.fromkeys(symbols, m)
        for m in sorted(OrderedSet([record_at, 1, *range(1, record_at, step)]))
    ]
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
    order = sorted(
        range(len(allocated)), key=lambda i: lifetimes.get(allocated[i], (0, 0))[0]
    )
    slot_free_at: list[int] = []  # step at which each slot becomes reusable
    assign = [0] * len(allocated)
    for i in order:
        start, end = lifetimes.get(allocated[i], (0, -1))
        if end < 0:
            end = 1 << 30  # graph output: lives past the whole schedule
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


def buffer_size_exprs(source_code: str) -> dict[str, tuple[str, int]]:
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


def buffer_layouts(source_code: str) -> dict[str, tuple[list[str], list[str], str]]:
    """Size/stride expressions and dtype of every allocated buffer, by name.

    The element span buffer_size_exprs returns is all the arena layout needs, but
    a buffer the graph returns has to reach the caller with the shape it was
    declared with, which only the allocation call carries.
    """
    return {
        name: (sizes, strides, dtype)
        for name, sizes, strides, dtype in _find_allocations(source_code)
    }


# Element sizes for the dtypes Inductor emits in empty_strided_cuda calls.
_ITEMSIZE = {
    "torch.float32": 4,
    "torch.float": 4,
    "torch.float64": 8,
    "torch.double": 8,
    "torch.float16": 2,
    "torch.half": 2,
    "torch.bfloat16": 2,
    "torch.int64": 8,
    "torch.long": 8,
    "torch.int32": 4,
    "torch.int": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.bool": 1,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e5m2": 1,
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
            f"  // {name}"
        )
    out = _LAYOUT_TEMPLATE
    for tag, val in (
        ("/*@NSLOTS@*/", str(n_slots)),
        ("/*@SLOTSIZES@*/", "\n".join(lines)),
    ):
        out = out.replace(tag, val)
    return out


def generate_pointer_patches(
    kernels: list[dict[str, Any]],
    slot_of: dict[str, int] | None,
    alias: dict[str, str] | None = None,
) -> str:
    """Per-node ``case`` bodies that repoint each buffer argument into the arena.

    ``slot_of`` of None means no arena: the planner is only patching shapes and
    every buffer keeps the address the capture gave it. That is different from an
    empty assignment, which would mean an arena exists but owns nothing.

    Call sites name the buffer Inductor renamed to, not the one that owns the
    allocation -- ``buf2 = buf0  # reuse`` means ``buf2`` never appears in the
    slot assignment -- so the alias map has to be applied before the lookup.
    Anything still unaccounted for is refused: an unpatched pointer keeps the
    address the capture gave it, and the graph would quietly return whichever
    kernel last wrote there.
    """
    if slot_of is None:
        return ""
    alias = alias or {}
    cases = []
    for i, k in enumerate(kernels):
        body = []
        for nm, raw in (k.get("ptrs") or {}).items():
            buf = alias.get(raw, raw)
            if not re.fullmatch(r"buf\d+", buf):
                continue  # a graph input, which keeps its own fixed address
            if buf not in slot_of:
                raise Unsupported(
                    f"{k['name']} argument {nm} is {raw}, which no allocation owns"
                )
            off, size = k["offsets"][nm]
            if size != 8:
                raise Unsupported(f"{k['name']} pointer {nm} is {size} bytes")
            body.append(
                f"      {{ char* p = arena + slot_off[{slot_of[buf]}]; "
                f"cudaGraphKernelNodeSetParam(handles[i], {off}, &p, 8); }}"
                f"  // {nm} = {buf}"
            )
        if body:
            cases.append(f"    case {i}:\n" + "\n".join(body) + "\n      break;")
    return "\n".join(cases)


# --------------------------------------------------------------- runner
def _wrapper_source(model: Any) -> str | None:
    g = getattr(model, "__globals__", None)
    path = (g or {}).get("__file__")
    if not path:
        return None
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def _graph_outputs(src: str) -> list[str]:
    """Buffer names the wrapper returns, in order."""
    m = re.search(r"^\s*return\s*\(([^)]*)\)", src, re.MULTILINE)
    if not m:
        return []
    return [
        a.strip() for a in _split_args(m.group(1)) if re.fullmatch(r"buf\d+", a.strip())
    ]


def _entry_source(src: str) -> str | None:
    """Body of the function cudagraphify is actually handed.

    Under graph partitioning compile_fx cudagraphifies each `partition_N` on its
    own through `recursively_apply_fns`, so the runtime argument order is that
    function's rather than `Runner.call`'s. The two genuinely differ -- call
    takes (weight, bias, symbol, activation) and hands the partition
    (activation, weight, bias, symbol) -- and the partition receives the symbol
    as a named entry instead of through an assignment.

    More than one partition means the kernels read off this file belong to
    several graphs, which nothing here separates, so that bails.
    """
    bodies = re.findall(
        r"^def partition_\d+\(args\):\n(.*?)(?=^\S)", src, re.MULTILINE | re.DOTALL
    )
    if not bodies:
        return src
    if len(bodies) != 1:
        return None
    return bodies[0]


# Ways the wrapper launches GPU work that is not a Triton kernel this pass can
# reach. extern_kernels is cuBLAS/cuDNN, torch.ops and aten calls are fallbacks
# to the dispatcher, and a .item() forces a device-to-host read.
_UNREACHABLE = re.compile(
    r"\bextern_kernels\s*\.|\btorch\s*\.\s*ops\s*\.|(?<![\w.])aten\s*\.\w|\.item\(\)"
)


def unreachable_launch(src: str) -> str | None:
    """A call in the entry function that the planner could never re-parameterize.

    The handle count alone does not catch these, which is easy to get wrong: an
    extern kernel is not a CachingAutotuner, so it never enters the kernel table,
    and it does not go through the static launcher, so it never yields a handle.
    It is missing from both sides at once and the counts still agree. What has
    actually been stopping such graphs is the replay check noticing wrong
    numbers, which is an empirical result and not a guarantee -- a node left at
    the shape it was recorded at is exactly the silent corruption this pass is
    supposed to refuse outright.
    """
    body = _entry_source(src)
    if body is None:
        return None
    for line in body.splitlines():
        code = line.split("#", 1)[0]
        m = _UNREACHABLE.search(code)
        if m:
            return code.strip()[:120]
    return None


def _input_symbol_map(src: str) -> dict[str, int]:
    """Symbol -> index in the argument list.

    A symbol is an argument of its own rather than something read back off a
    tensor, and the unpacking line is where it enters::

        arg3_1, arg0_1, arg1_1, s77 = args  # partition: named directly
        arg0_1, arg1_1, arg2_1, arg3_1 = args  # unpartitioned: via an alias
        s77 = arg2_1

    Reading it here rather than inferring it from a size assertion matters
    because an assertion only constrains a tensor whose shape happens to mention
    the symbol, while this is where the value itself arrives.
    """
    body = _entry_source(src)
    if body is None:
        return {}
    unpack = re.search(r"^\s*((?:\w+\s*,\s*)+\w+)\s*=\s*args\s*$", body, re.MULTILINE)
    if not unpack:
        return {}
    names = [n.strip() for n in unpack.group(1).split(",")]
    if not all(re.fullmatch(r"arg\d+_1|s\d+", n) for n in names):
        return {}
    pos = {n: i for i, n in enumerate(names)}
    out = {n: i for n, i in pos.items() if n.startswith("s")}
    for m in re.finditer(r"^\s*(s\d+)\s*=\s*(arg\d+_1)\s*$", body, re.MULTILINE):
        if m.group(2) in pos:
            out[m.group(1)] = pos[m.group(2)]
    return out


def _buffer_aliases(src: str) -> dict[str, str]:
    """Buffers that are another buffer under a new name.

    Inductor renames a buffer when it reuses the storage::

        buf2 = buf0
        del buf0  # reuse

    so a returned buffer often has no allocation of its own and has to be
    resolved back to the one that does.
    """
    alias: dict[str, str] = {}
    for m in re.finditer(r"^\s*(buf\d+)\s*=\s*(buf\d+)\s*(?:;|$)", src, re.MULTILINE):
        alias[m.group(1)] = m.group(2)
    for k in list(alias):
        seen, v = OrderedSet(), alias[k]
        while v in alias and v not in seen:
            seen.add(v)
            v = alias[v]
        alias[k] = v
    return alias


def lifetimes_from_source(src: str) -> dict[str, tuple[int, int]]:
    """Read buffer lifetimes off the wrapper, using its own del statements.

    Inductor emits `del bufN` at the point a buffer dies, so the source already
    carries the liveness the slot assignment needs, which avoids reaching into
    the compile-time memory planning pass. Buffers the wrapper returns never get
    a del and so live past the schedule, marked here with an end of -1 as
    compute_memory_timeline does.

    A renamed buffer keeps the storage alive under the new name: `buf2 = buf0;
    del buf0  # reuse` deletes buf0 on the spot but buf2 goes on using that
    memory. Lifetimes are keyed on the buffer that owns the allocation, so each
    one has to die no earlier than the last name pointing at it -- otherwise the
    slot gets handed to another buffer while it is still being read.
    """
    steps = [ln.strip() for ln in src.splitlines()]
    born: dict[str, int] = {}
    died: dict[str, int] = {}
    for i, ln in enumerate(steps):
        m = re.match(r"(buf\d+)\s*=\s*empty_strided_cuda\(", ln)
        if m and m.group(1) not in born:
            born[m.group(1)] = i
        for name in re.findall(r"\bdel\s+([\w\s,]+)", ln):
            for nm in (x.strip() for x in name.split(",")):
                if re.fullmatch(r"buf\d+", nm) and nm not in died:
                    died[nm] = i
    alias = _buffer_aliases(src)
    returned = OrderedSet(_graph_outputs(src))
    out: dict[str, tuple[int, int]] = {}
    for nm, b in born.items():
        names = [nm] + [k for k, v in alias.items() if v == nm]
        end = (
            -1
            if any(x in returned for x in names)
            else max(died.get(x, len(steps)) for x in names)
        )
        out[nm] = (b, end)
    return out


class DynaGraphRunner:
    """One captured graph serving a whole shape space, re-laid-out on every call.

    This bypasses CUDAGraphTreeManager rather than extending it. The tree
    machinery exists to share one memory pool between graphs and to checkpoint the
    allocator so that recording can resume after a replay; neither applies when
    there is a single graph whose buffers live in an arena owned here. What is
    given up is pool sharing with other compiled regions.

    Returns None from :meth:`build` whenever anything about the graph is not
    understood, so the caller falls back to the ordinary path.
    """

    def __init__(self, model: Any, src: str, device: Any) -> None:
        self.model = model
        self.src = src
        self.device = device
        self.graph: Any = None
        self.kernels, self.symbols = extract_kernel_table(
            src, getattr(model, "__globals__", {})
        )
        self.sizes = buffer_size_exprs(src)
        self.layouts = buffer_layouts(src)
        self.alias = _buffer_aliases(src)
        # A returned buffer is often a rename of one that owns the allocation.
        self.outputs = [self.alias.get(b, b) for b in _graph_outputs(src)]
        self.outputs = [b for b in self.outputs if b in self.sizes]
        self.sym_from_input = _input_symbol_map(src)
        # Shapes whose replay has already been checked against eager.
        self.verified: OrderedSet[tuple[tuple[str, int], ...]] = OrderedSet()
        self.sym_index = {s: i for i, s in enumerate(self.symbols or [])}

    def unusable_reason(self) -> str | None:
        """The first thing that rules this graph out, or None to go ahead.

        A reason rather than a bool because these four are the commonest way a
        region is turned down, and a sweep over many models wants to know which.
        """
        if not self.kernels:
            return "no-kernels"
        if not self.symbols:
            return "no-symbols"
        if not self.outputs:
            return "no-arena-outputs"
        # Ahead of the symbol check, because a multi-partition wrapper has no
        # single entry function and so reports an empty symbol map -- it would
        # otherwise be counted under a cause that is not its own.
        if _entry_source(self.src) is None:
            return "multi-partition"
        if not self.sym_from_input:
            return "no-symbol-args"
        if unreachable_launch(self.src):
            return "extern-launch"
        return None

    def build(
        self,
        inputs: list[Any],
        lifetimes: dict[str, tuple[int, int]],
        env: dict[str, int],
    ) -> bool:
        """Record the one graph, or return False to leave this region alone.

        The order matters. Warming up has to happen before the planner is
        generated, not merely before the capture: it is what makes each autotuner
        commit to the single config the graph will bake in, and the grid formulas
        are built from that config's block sizes. The capture comes last, and is
        checked against eager before the graph is handed out.
        """
        import torch
        from torch._inductor import config

        headroom = config.triton.dynagraph_headroom

        allocated = sorted(self.sizes)
        assign, n_slots = plan_slots(lifetimes, allocated)
        self.slot_of = dict(zip(allocated, assign))
        self.n_slots = n_slots

        # Inputs must sit at a fixed address for the graph's lifetime, so they are
        # copied into storage owned here and the recorded graph reads only from
        # that. The storage is oversized by the same headroom as the arena: the
        # recorded shape is whatever happened to arrive first, and a later, larger
        # shape has to land at the same address, since only the extents are
        # patched, not the pointer. Past the headroom the region retires.
        self.input_store: list[Any] = []
        self.static_inputs = []
        for x in inputs:
            if not isinstance(x, torch.Tensor):
                self.input_store.append(None)
                self.static_inputs.append(x)
                continue
            if not x.is_contiguous():
                return _fallback("input-not-contiguous")
            n = x.numel()
            store = torch.empty(
                max(int(n * headroom), n), dtype=x.dtype, device=x.device
            )
            store[:n].copy_(x.reshape(-1))
            self.input_store.append(store)
            self.static_inputs.append(store[:n].view(x.shape))

        # The warmup has to come before the planner is generated, not just before
        # the capture: it is what makes each autotuner commit to the single
        # config the graph will bake in, and the grid formula is built from that
        # config's block sizes.
        self._warmup()
        for k in self.kernels:
            k["blocks"] = settled_blocks(self.model.__globals__.get(k["gname"]))
            if k["blocks"] is None:
                return _fallback("unsettled-config", k["name"])

        layout_src = generate_layout(self.sizes, self.slot_of, n_slots, self.symbols)
        try:
            planner_src = generate_planner(
                self.kernels, self.symbols, self.slot_of, self.alias
            )
        except Unsupported as exc:
            return _fallback("unmodelled", str(exc))
        if os.environ.get("TORCHINDUCTOR_DYNAGRAPH_DUMP"):
            with open(os.environ["TORCHINDUCTOR_DYNAGRAPH_DUMP"], "w") as fh:
                fh.write(
                    f"// kernels: {self.kernels}\n// slots: {self.slot_of}\n"
                    f"// alias: {self.alias}\n// sizes: {self.sizes}\n\n"
                )
                fh.write(layout_src + planner_src)
        funcs = _compile_module(
            layout_src + planner_src, ["dynagraph_layout", "dynagraph_planner"]
        )
        if funcs is None:
            return _fallback("planner-build")
        self.f_layout, self.f_planner = funcs

        total = 0
        for span, itemsize in self.sizes.values():
            v = _eval_int(span, env)
            if v is None:
                return _fallback("unevaluable-size", span)
            total += ((v * itemsize + 255) // 256) * 256
        self.arena = torch.empty(
            max(int(total * headroom), 1024), dtype=torch.uint8, device=self.device
        )
        self.slot_off = torch.zeros(n_slots + 1, dtype=torch.int64, device=self.device)
        self.ctx = torch.zeros(
            max(1, len(self.symbols)), dtype=torch.int64, device=self.device
        )
        self.handles = torch.zeros(
            len(self.kernels), dtype=torch.int64, device=self.device
        )

        return self._capture() and self._replays_match(env)

    def _warmup(self) -> None:
        import torch

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.model(list(self.static_inputs))
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    def _replays_match(self, env: dict[str, int]) -> bool:
        """Replay once at the shape that was recorded and check the arena.

        Everything the planner does is derived by reading the wrapper, and a
        formula that is wrong in a way the source does not reveal -- a grid built
        from the wrong autotuner config, say -- produces no error at all, just an
        output whose tail was never written. One replay compared against running
        the same partition eagerly costs nothing next to compiling the graph and
        turns that whole class of mistake into a fallback.
        """

        ref = self.model(list(self.static_inputs))
        got = self(list(self.static_inputs))
        if got is None or not _same_values(got, ref):
            return _fallback("selfcheck-mismatch", f"at {env}")
        return True

    def _capture(self) -> bool:
        import torch

        launcher = torch._C._StaticCudaLauncher
        graph = torch.cuda.CUDAGraph()
        launcher._begin_device_node_collection()
        try:
            with torch.cuda.graph(graph):
                raw = torch.cuda.current_stream().cuda_stream
                _launch(
                    self.f_layout,
                    [self.ctx.data_ptr(), self.slot_off.data_ptr()],
                    1,
                    1,
                    raw,
                )
                launch_planner(
                    self.f_planner,
                    len(self.kernels),
                    self.handles.data_ptr(),
                    self.ctx.data_ptr(),
                    raw,
                    self.arena.data_ptr(),
                    self.slot_off.data_ptr(),
                )
                self.model(list(self.static_inputs))
            handles = launcher._end_device_node_collection()
        except Exception as exc:
            with contextlib.suppress(Exception):
                launcher._end_device_node_collection()
            return _fallback("capture-failed", f"{type(exc).__name__}: {exc}")

        if len(handles) != len(self.kernels):
            # A mismatch means some kernel did not go through the static launcher,
            # so its node has no handle and the planner cannot reach it. Replaying
            # would leave that node at the recorded shape, which is silent
            # corruption rather than an error, so refuse the whole graph.
            #
            # The converse does not hold, which is why unreachable_launch exists:
            # a call that bypasses the static launcher is usually missing from the
            # kernel table too, and the two counts still agree.
            return _fallback(
                "handle-mismatch",
                f"{len(handles)} handles for {len(self.kernels)} kernels",
            )
        self.handles.copy_(torch.tensor(handles, dtype=torch.int64))
        torch.cuda.synchronize()
        self.graph = graph
        return True

    def __call__(self, inputs: list[Any]) -> Any:
        """Serve one call, or return None once this region can no longer be trusted.

        None rather than an exception: the caller still holds `inputs`, which this
        only clears on the way out, so it can just record the shape the ordinary
        way.
        """
        import torch
        from torch._inductor import config

        env = {}
        for sym, i in self.sym_from_input.items():
            v = inputs[i] if i < len(inputs) else None
            if isinstance(v, int):
                env[sym] = v
        key = tuple(sorted(env.items()))

        for j, (store, srcv) in enumerate(zip(self.input_store, inputs)):
            if store is None or not isinstance(srcv, torch.Tensor):
                continue
            n = srcv.numel()
            if n > store.numel():
                _fallback("input-too-large", f"arg {j}: {n} > {store.numel()}")
                return None
            if not srcv.is_contiguous():
                _fallback("input-not-contiguous", f"arg {j}")
                return None
            store[:n].copy_(srcv.reshape(-1))

        # New shapes are run eagerly alongside the replay until enough of them have
        # agreed. The build-time check only covers the shape the graph was recorded
        # at, and a grid formula can be right there and wrong everywhere else --
        # that is exactly how the XBLOCK bug hid. Eager gets `inputs`, which still
        # carries the real shape; the replay reads the fixed-size copies, so the
        # two sides are computed independently.
        ref = None
        if key not in self.verified and len(self.verified) < (
            config.triton.dynagraph_verify_shapes
        ):
            ref = self.model(list(inputs))

        for sym, val in env.items():
            if sym in self.sym_index:
                self.ctx[self.sym_index[sym]] = val

        # The layout has to be known before the replay, not after it. Nothing
        # else bounds it: the arena is sized once at build from the shape that
        # happened to arrive first, and a buffer growing faster than the input
        # outruns the headroom -- by the time the graph's own layout node has run
        # the kernels have already written past the end. Running the same kernel
        # here first is what makes the check preventive, and it costs nothing
        # extra: reading the offsets already forced this sync, it has only moved
        # ahead of the replay. The layout node inside the graph then recomputes
        # the same values from the same ctx.
        _launch(
            self.f_layout,
            [self.ctx.data_ptr(), self.slot_off.data_ptr()],
            1,
            1,
            torch.cuda.current_stream().cuda_stream,
        )
        offsets = self.slot_off.tolist()
        if offsets[self.n_slots] > self.arena.numel():
            _fallback(
                "arena-too-small",
                f"{offsets[self.n_slots]} > {self.arena.numel()} at {env}",
            )
            return None

        self.graph.replay()

        out = []
        for name in self.outputs:
            span, itemsize = self.sizes[name]
            sizes_e, strides_e, dtype_name = self.layouts[name]
            vals = [_eval_int(e, env) for e in (span, *sizes_e, *strides_e)]
            ints = [v for v in vals if v is not None]
            if len(ints) != len(vals):
                raise RuntimeError(f"DynaGraph cannot size output {name}")
            n, rest = ints[0], ints[1:]
            sizes, strides = rest[: len(sizes_e)], rest[len(sizes_e) :]
            base = offsets[self.slot_of[name]]
            # The layout kernel keeps every slot 256-byte aligned, so viewing the
            # byte arena as the buffer dtype is always legal. as_strided keeps the
            # storage offset of the slice it is called on.
            flat = self.arena[base : base + n * itemsize].view(
                getattr(torch, dtype_name.split(".")[-1])
            )
            out.append(flat.as_strided(sizes, strides))

        if ref is not None:
            if not _same_values(out, ref):
                _fallback("runtime-mismatch", f"at {env}")
                return None
            self.verified.add(key)
        inputs.clear()
        return out


def _same_values(got: list[Any], ref: Any) -> bool:
    """Bit-for-bit agreement between a replay's outputs and an eager run's.

    Exact rather than tolerant on purpose: both sides run the same kernels with
    the same settled launch config on the same inputs, so any difference at all
    means the planner changed something it should not have.
    """
    import torch

    flat = list(ref) if isinstance(ref, (list, tuple)) else [ref]
    if len(got) != len(flat):
        return False
    return all(
        isinstance(r, torch.Tensor)
        and g.numel() == r.numel()
        and torch.equal(g.reshape(-1), r.reshape(-1))
        for g, r in zip(got, flat)
    )


def _compile_module(src: str, names: list[str]) -> tuple[int, ...] | None:
    import torch

    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}" + ("a" if (major, minor) >= (9, 0) else "")
    with tempfile.TemporaryDirectory() as wd:
        cu, cubin = os.path.join(wd, "dg.cu"), os.path.join(wd, "dg.cubin")
        with open(cu, "w") as fh:
            fh.write(src)
        r = subprocess.run(
            ["nvcc", f"-arch={arch}", "-cubin", "-o", cubin, cu],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            log.warning("DynaGraph nvcc failed: %s", r.stderr[-600:])
            return None
        mod = ctypes.c_void_p()
        if _cuda().cuModuleLoad(ctypes.byref(mod), cubin.encode()) != 0:
            return None
    got = []
    for nm in names:
        fn = ctypes.c_void_p()
        if _cuda().cuModuleGetFunction(ctypes.byref(fn), mod, nm.encode()) != 0:
            return None
        if fn.value is None:
            return None
        got.append(fn.value)
    return tuple(got)


def _launch(func: int, args: list[int], grid: int, block: int, stream: int) -> None:
    holders = [ctypes.c_void_p(a) for a in args]
    arr = (ctypes.c_void_p * len(holders))(
        *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders]
    )
    rc = _cuda().cuLaunchKernel(
        ctypes.c_void_p(func),
        grid,
        1,
        1,
        block,
        1,
        1,
        0,
        ctypes.c_void_p(stream),
        arr,
        None,
    )
    if rc != 0:
        raise RuntimeError(f"DynaGraph kernel launch failed: {rc}")
