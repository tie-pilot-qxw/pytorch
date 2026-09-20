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
import functools
import hashlib
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


log = logging.getLogger(__name__)

_libcuda: Any = None


def _cuda() -> Any:
    global _libcuda
    if _libcuda is None:
        lib = ctypes.CDLL("libcuda.so.1")
        # The entry points on the per-call path are typed once, so a call is
        # one ctypes dispatch with no argument conversion.
        lib.cuGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cuGraphLaunch.restype = ctypes.c_int
        # The unversioned `cuMemcpyDtoDAsync` in libcuda is the CUDA 3 entry
        # (32-bit pointers); cuda.h maps the name to `_v2`, ctypes does not.
        lib.cuMemcpyDtoDAsync_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.cuMemcpyDtoDAsync_v2.restype = ctypes.c_int
        _libcuda = lib
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


def _parse_run_calls(src: str) -> list[tuple[str, list[str]]]:
    """Every ``<kernel>.run(...)`` in the entry function, in launch order.

    A list rather than a map keyed on the kernel: a model that repeats a block
    launches the same kernel once per block, and each launch is its own graph
    node with its own device-updatable handle. Keying by name kept only the last
    call site, so a twelve-block model built a table of five entries against
    forty-eight nodes and the handle count refused the graph -- correctly, but
    for a reason that made every repeating model look unsupported.

    Sizes are not named variables in general; they are inlined at the call site::

        triton_per_..._1.run(buf4, arg1_1, buf0, s77, 256, stream=raw_stream0)
        #                    in_out  in_ptr0 in_ptr1 xnumel r0_numel

    A kernel whose grid_type is FixedGrid additionally passes grid_0/1/2 right
    after its arguments, so the grid is recoverable the same way.

    Trailing keyword arguments -- ``stream=raw_stream0`` -- are dropped, so that
    the length of what comes back can be compared against the kernel signature.
    """
    body = _entry_source(src)
    if body is None:
        return []
    calls: list[tuple[str, list[str]]] = []
    for m in re.finditer(r"(\w+)\.run\(", body):
        name, i = m.group(1), m.end()
        depth, j = 1, i
        while j < len(body) and depth:
            if body[j] == "(":
                depth += 1
            elif body[j] == ")":
                depth -= 1
            j += 1
        args = _split_args(body[i : j - 1])
        while args and re.match(r"\s*\w+\s*=[^=]", args[-1]):
            args.pop()
        calls.append((name, args))
    return calls


def _fallback(tag: str, detail: str = "") -> bool:
    """Record why a graph is not being served, and return False for the caller.

    Every refusal goes through here so that a sweep over many models can count
    the reasons. The tag is a stable short name meant to be grepped and tallied;
    the detail is free text for whoever is reading one log.
    """
    log.info("DynaGraph fallback [%s]%s", tag, f": {detail}" if detail else "")
    return False


# Returned by DynaGraphRunner.__call__ for a shape it will not serve while the
# region itself stays good: the caller records that one shape the ordinary way
# and keeps asking for the others. Distinct from None, which retires the region.
SKIP_SHAPE = object()

# Returned by DynaGraphRunner.__call__ for a shape the region was built too
# small for: an input past its storage, or a buffer past its arena slot. Both
# are sized from whichever shape arrived at build, times the headroom, and a
# data-dependent length or a buffer quadratic in a symbol walks past that
# easily. The caller records this one shape the ordinary way and builds a new
# runner on it, so the larger shape now sets the sizes -- one recording, which
# is what recording per shape would have charged for it anyway -- instead of
# retiring the region for good on the first shape it did not foresee.
REBUILD = object()


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


# A backed symbol is spelled `s<n>`; an unbacked one (the row count of a
# boolean mask, a nonzero) `u<n>`. Under cudagraph_trees the graph is cut at
# the op that produces an unbacked size, and the value reaches the next
# partition as a plain int argument, so from here on it is a symbol like any
# other: the host knows it before the launch.
_SYMBOL = r"\b[su]\d+\b"


def _is_symbolic(expr: Any) -> bool:
    return bool(expr) and bool(re.search(_SYMBOL, str(expr)))


# Constexprs the autotuner picks. Unlike a specialized scalar these never reach
# the .run() call site, so they are excluded when lining its positionals up.
_TUNED = (
    "XBLOCK",
    "YBLOCK",
    "ZBLOCK",
    "R0_BLOCK",
    "R1_BLOCK",
    "RBLOCK",
    "RSPLIT",
    "RSPLIT_SIZE",
)


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

    run_calls = _parse_run_calls(source_code)
    symbols = frozenset(re.findall(r"\b([su]\d+)\b", source_code))
    autotuners = {
        nm: o for nm, o in call_globals.items() if isinstance(o, CachingAutotuner)
    }

    kernels = []
    for gname, pos in run_calls:
        obj = autotuners.get(gname)
        if obj is None:
            return None, None  # a .run on something not introspectable
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

        # Every compiled variant of this kernel: the autotuner launches the
        # one it settled on, and the host patcher finds the node by function.
        cands: list[int] = []
        for cr in getattr(obj, "compile_results", []) or []:
            k = getattr(cr, "kernel", None)
            f = getattr(k, "function", None)
            if f:
                cands.append(int(f))
            cands += [int(v) for v in (getattr(k, "functions", {}) or {}).values() if v]
        cands = list(dict.fromkeys(cands))
        func = cands[0] if cands else None
        if not func:
            return None, None  # kernel not statically launched; bail out

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
        # A scalar Inductor specialized to a constant is gone from the parameter
        # list but still sits at the call site. It needs no patch -- it is baked
        # into the cubin -- but a grid formula built on it still has to be able
        # to read its value, or an otherwise ordinary kernel gets refused for
        # having "no xnumel".
        consts = {
            nm: pos[i].strip()
            for i, nm in enumerate(call_order)
            if nm not in args and i < len(pos)
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
                func=int(func),
                funcs=cands,
                exprs=exprs,
                consts=consts,
                ptrs=ptrs,
                offsets=offsets,
                blocks=blocks,
                grid=grid,
                grid_type=meta.get("grid_type"),
                # A combo (horizontally fused) kernel's grid is a formula over
                # its sub-kernels' numels; this is the meta that formula reads.
                combo=meta.get("combo_grid_meta"),
                # A split scan's decoupled look-back (atomic_cas, not add, so
                # Inductor's flag misses it) and any atomic add make the
                # result differ bit for bit between two runs of the same
                # kernel on the same input, so a region holding one is
                # checked with a tolerance, not for equality.
                atomic=bool(meta.get("atomic_add_found"))
                or meta.get("grid_type") == "SplitScanGrid",
                # `obj` is the autotuner; `k` above is one compiled variant.
                cooperative=bool((obj.triton_meta or {}).get("launch_cooperative_grid"))
                or meta.get("grid_type") == "CooperativeReductionGrid",
            )
        )

    # Already in launch order: the table is built by walking the call sites.
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
// nvrtc ships no libstdc++ headers and does not pull in the device runtime
// (the device-side graph update calls below) on its own.
#ifdef __CUDACC_RTC__
typedef long long int64_t; typedef int int32_t;
#include <cuda_device_runtime_api.h>
#else
#include <cstdint>
#endif

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

// The host hands the planner its inputs through this: symbol values, the
// "changed" flag, the extern output addresses and the SWITCH body indices, as
// kernel arguments of one launch. A pageable host-to-device copy per value
// (`ctx[i] = v` from Python) cost 15 us each and serialized against the
// stream; a launch carries the values with it and orders like any kernel.
extern "C" __global__ void dynagraph_setctx(int64_t* __restrict__ ctx/*@SETCTX_PARAMS@*/)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
/*@SETCTX_BODY@*/
  // The arena layout for this shape: each slot is the largest buffer assigned
  // to it, offsets are a prefix sum kept 256-byte aligned, the total last.
  // Under a fixed layout these are the constants the build chose.
  {
    int64_t acc = 0, m;
    (void)acc; (void)m;
/*@LAYOUT@*/
  }
}

// The arena base and the slot offsets live in ctx (written by setctx), so a
// grown arena is a new value in the next setctx launch, not a new graph.
#define ARENA ((char*)ctx[/*@ARENA@*/])
#define SLOT_OFF(k) (ctx[/*@OFF0@*/ + (k)])
#define LASTP(j) (ctx[/*@LASTP0@*/ + (j)])

// One thread per node. Both the grid and the shape-carrying scalar arguments
// must be updated: a stale grid launches the wrong number of blocks, while stale
// arguments leave the kernel's own bounds checks referring to the recorded shape,
// which silently corrupts the tail block.
extern "C" __global__ void dynagraph_planner(
    const cudaGraphDeviceNode_t* __restrict__ handles,
    int64_t* __restrict__ ctx)
{
  // Last slot of ctx is the host's answer to "did the shape change since the
  // replay before this one". Patching is idempotent, so repeating it for an
  // unchanged shape is pure cost -- measured at 43.7 us per replay, 14.3% of
  // this graph's device time, spread over several hundred per-node and
  // per-argument update calls with no single dominant one. The comparison is
  // the host's rather than the device's because threads in different blocks
  // are not ordered against each other: one block could read the "already
  // applied" flag after another had written it and skip patches it still owed.
  if (ctx[/*@NSYM@*/] == 0 && ctx[/*@INDIRTY@*/] == 0) return;

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Sites whose extern topology varies are SWITCH nodes: thread 0 selects the
  // body for this shape from the index the host wrote past the extern
  // outputs. A site with a single topology has a zero handle here. Set only
  // when the shape changed (above), so a skipped planner leaves the last
  // selection standing.
  if (i == 0 && ctx[/*@NSYM@*/] != 0) {
    for (int s = 0; s < /*@NSITE@*/; ++s) {
      long long h = ctx[/*@COND0@*/ + s];
      if (h != 0)
        cudaGraphSetConditional((cudaGraphConditionalHandle)h,
                                (unsigned)ctx[/*@BODY0@*/ + s]);
    }
  }
  if (i >= /*@N@*/) return;

  // Inputs read where they are: when one moved, the host wrote the new
  // addresses past the SWITCH handles and raised the flag, and every node
  // repoints the arguments that read them. Before the grid section, so a
  // node disabled at this shape still takes the address for when it is
  // enabled again.
  if (ctx[/*@INDIRTY@*/] != 0) {
    switch (i) {
/*@INPTRS@*/
      default: break;
    }
  }
  if (ctx[/*@NSYM@*/] == 0) return;

  // The cost of this kernel is the number of runtime calls it makes, not any
  // one of them (microbench/planner_batched.cu), so calls that would not
  // change anything are not made. A grid with no symbol in it was fixed at
  // compile time and is never patched; a node's enabled state is kept past the
  // symbols in ctx (one slot per node, owned by this thread) so SetEnabled is
  // only issued on a transition, not on every replay.
  int64_t gx = 1, gy = 1, gz = 1;
  bool static_grid = false;
  switch (i) {
/*@GRID@*/
    default: break;
  }
  if (!static_grid) {
    int64_t* en = ctx + /*@STATE0@*/ + i;
    if (gx <= 0 || gy <= 0 || gz <= 0) {
      // A grid dimension of zero is cudaErrorInvalidArgument, so an empty
      // tensor has to be expressed by disabling the node rather than by a
      // zero grid.
      if (*en) { cudaGraphKernelNodeSetEnabled(handles[i], 0); *en = 0; }
      return;
    }
    if (!*en) { cudaGraphKernelNodeSetEnabled(handles[i], 1); *en = 1; }
    cudaGraphKernelNodeSetGridDim(handles[i],
        dim3((unsigned)gx, (unsigned)gy, (unsigned)gz));
  }

  switch (i) {
/*@PARAMS@*/
    default: break;
  }

  // Repoint buffer arguments into the arena. Needed whenever no single shape
  // dominates the interval: with a fixed total split over a varying number of
  // samples, one buffer grows as another shrinks, so recording at a maximum
  // cannot cover both and the layout is redone per shape. Each argument
  // remembers the address it was last set to (LASTP), and only one that
  // moved costs a runtime call: under a fixed layout that is once after the
  // capture, under the dynamic layout it is the slots the shape shifted. The
  // bisect of the real 48-node planner put these calls at half its cost.
  if (ctx[/*@ARENA@*/] != 0 && (ctx[/*@NSYM@*/] != 0 || ctx[/*@PTRDIRTY@*/] != 0)) {
    switch (i) {
/*@PTRS@*/
      default: break;
    }
  }

  // Buffers an extern call produced itself sit wherever this shape's harvested
  // graph put them; the host writes those addresses past the node states.
  switch (i) {
/*@EXTPTRS@*/
    default: break;
  }

  // Arguments that are a view at an element offset (a cat's slices): the
  // offset can carry a symbol, so the address is recomputed on every run.
  switch (i) {
/*@VIEWPTRS@*/
    default: break;
  }
}
"""


# The grid shapes of triton_heuristics.GridExpr, one entry per launch axis:
# ("cdiv", numel, block) is a ceil-divide of that numel by that block of the
# settled config; ("numel", numel, None) is the numel itself; ("const", None,
# key) is a constant of the config (a cooperative reduction's RSPLIT). What
# is left out -- FixedGrid (grid passed as arguments, handled separately),
# PrecomputedGrid and the combo-kernel grids (per-config tables and
# horizontal fusion, both off by default) -- is refused as `unmodelled`.
# The y dimension of a launch is limited to this, which is why Inductor has a
# grid shape that folds the overflow into z.
_MAX_Y_GRID = 65535

_GRID_AXES = {
    "Grid1D": (("cdiv", "xnumel", "XBLOCK"),),
    "Grid2D": (("cdiv", "xnumel", "XBLOCK"), ("cdiv", "ynumel", "YBLOCK")),
    "Grid3D": (
        ("cdiv", "xnumel", "XBLOCK"),
        ("cdiv", "ynumel", "YBLOCK"),
        ("cdiv", "znumel", "ZBLOCK"),
    ),
    # Inductor's bmm template: batch on z, tiles on y and x.
    "BatchMatmulGrid3D": (
        ("cdiv", "znumel", "ZBLOCK"),
        ("cdiv", "ynumel", "YBLOCK"),
        ("cdiv", "xnumel", "XBLOCK"),
    ),
    "CooperativeReductionGrid": (
        ("const", None, "RSPLIT"),
        ("cdiv", "xnumel", "XBLOCK"),
    ),
    "SplitScanGrid": (("cdiv", "r0_numel", "R0_BLOCK"), ("numel", "xnumel", None)),
    "MixOrderReductionGrid": (("cdiv", "xnumel", "RSPLIT_SIZE"),),
}
# The overflow grid is not in the table (it is a formula, below); these are
# the numels its symbolic-or-not check reads.
_YZ_AXES = (("cdiv", "xnumel", "XBLOCK"), ("cdiv", "ynumel", "YBLOCK"))


def _grid_numel_names(k: dict[str, Any]) -> list[str]:
    """The numel arguments a kernel's grid is a function of.

    Read off the axis table for the plain grids, and off the combo meta for a
    horizontally fused kernel, whose grid sums or maxes its sub-kernels'
    numels. If none of them is symbolic the grid is fixed at capture.
    """
    combo = k.get("combo")
    if combo:
        names = []
        for i in range(int(combo.get("num_kernels", 0))):
            names.append(f"xnumel_{i}")
            if f"ynumel_{i}" in combo:
                names.append(f"ynumel_{i}")
        return names
    axes = _GRID_AXES.get(k.get("grid_type") or "") or _YZ_AXES
    return [n for _, n, _ in axes if n]


def _combo_grid_stmt(k: dict[str, Any], ev: Callable[[str], str]) -> str:
    """C statements setting gx, gy, gz for a combo kernel.

    Mirrors Inductor's SequentialComboKernelGrid (x is the sum of the
    sub-kernels' block counts) and RoundRobinComboKernelGrid (x is the largest
    block count times the kernel count); a sub-kernel with no x dimension
    contributes its numel as is. When any sub-kernel is tiled, y is the largest
    y block count folded into z past the launch limit, as Grid2DWithYZOverflow
    does. `ev` turns a wrapper expression into the C that evaluates it, which
    differs between the host patcher and the planner. The per-sub-kernel block
    variant (SequentialFlattenComboKernelGrid, off by default) is not modelled.
    """
    combo = k["combo"]
    gt = k.get("grid_type") or ""
    blocks = k["blocks"] or {}
    n = int(combo["num_kernels"])

    def numel(name: str) -> str:
        c = combo.get(name)
        if c is not None:
            return str(int(c))  # baked into the meta at codegen: not an argument
        e = k["exprs"].get(name) or k["consts"].get(name)
        if e is None:
            raise Unsupported(f"{k['name']}: no {name} to size a grid")
        return ev(e)

    def block(bk: str) -> int:
        v = blocks.get(bk)
        if v is None and combo.get("default_config"):
            v = combo["default_config"].get(bk)
        if v is None:
            raise Unsupported(f"{k['name']}: no {bk} to size a grid")
        return int(v)

    def cdiv(e: str, b: int) -> str:
        return f"dg_floordiv({e} + {b} - 1, {b})"

    stmts = []
    if gt == "SequentialComboKernelGrid":
        parts = []
        for i in range(n):
            x = numel(f"xnumel_{i}")
            parts.append(
                f"({x})" if combo.get(f"no_x_dim_{i}") else cdiv(x, block("XBLOCK"))
            )
        stmts.append(f"gx = {' + '.join(parts)};")
    elif gt == "RoundRobinComboKernelGrid":
        flat = [numel(f"xnumel_{i}") for i in range(n) if combo.get(f"no_x_dim_{i}")]
        tiled = [
            numel(f"xnumel_{i}") for i in range(n) if not combo.get(f"no_x_dim_{i}")
        ]
        if tiled:
            stmts.append(f"int64_t xm = {tiled[0]};")
            for e in tiled[1:]:
                stmts.append(f"{{ int64_t t = {e}; if (t > xm) xm = t; }}")
            flat.append(cdiv("xm", block("XBLOCK")))
        stmts.append(f"int64_t m = {flat[0]};")
        for e in flat[1:]:
            stmts.append(f"{{ int64_t t = {e}; if (t > m) m = t; }}")
        stmts.append(f"gx = m * {n};")
    else:
        raise Unsupported(f"{k['name']}: grid type {gt} is not modelled")
    if combo.get("min_blocks"):
        stmts.append(
            f"if (gx < {int(combo['min_blocks'])}) gx = {int(combo['min_blocks'])};"
        )
    ys = [numel(f"ynumel_{i}") for i in range(n) if f"ynumel_{i}" in combo]
    if ys:
        yb = block("YBLOCK")
        stmts.append(f"int64_t ym = {ys[0]};")
        for e in ys[1:]:
            stmts.append(f"{{ int64_t t = {e}; if (t > ym) ym = t; }}")
        stmts.append(
            f"int64_t raw = {cdiv('ym', yb)};"
            f" int64_t div = dg_floordiv(raw + {_MAX_Y_GRID} - 1, {_MAX_Y_GRID});"
            " gy = (div == 0) ? 0 : dg_floordiv(raw + div - 1, div); gz = div;"
        )
    else:
        stmts.append("gy = 1; gz = 1;")
    return "{ " + " ".join(stmts) + " }"


def generate_planner(
    kernels: list[dict[str, Any]],
    symbols: list[str],
    slot_of: dict[str, int] | None = None,
    alias: dict[str, str] | None = None,
    input_bufs: Any = (),
    extern_out_of: dict[str, int] | None = None,
    n_sites: int = 0,
    n_ext: int | None = None,
    argv: dict[str, int] | None = None,
    patch_inputs: Any = (),
    views: dict[str, Any] | None = None,
    itemsize_of: dict[str, int] | None = None,
    input_addr: dict[int, int] | None = None,
    sizes: dict[str, tuple[str, int]] | None = None,
    fixed_off: Sequence[int] | None = None,
) -> str:
    """The device-side planner: CUDA source with a kernel that patches the graph.

    One thread per kernel node. From the symbol values the host wrote into ctx
    it recomputes each node\'s grid and shape-carrying scalar arguments,
    repoints buffer arguments into the arena once per exec, and repoints
    extern-output, offset-view and in-place-input arguments per run. The
    generated `case` bodies come from the same kernel table the host patcher
    reads, so the two paths agree by construction.

    `sizes` (element span and item size per buffer) with `slot_of` is what
    the `setctx` kernel lays the arena out from on every shape; `fixed_off`
    instead bakes a layout chosen at build (`dynagraph_layout="fixed"`).
    """
    n_ext = n_sites if n_ext is None else n_ext
    n_in = len(argv) if argv else 0
    n_slots = (max(slot_of.values()) + 1) if slot_of else 0
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
            if not any(_is_symbolic(e) for e in k["grid"]):
                grid_cases.append(
                    f"    case {i}: static_grid = true; break;  // {k['name']}"
                )
                continue
            gx, gy, gz = (idx(e) for e in k["grid"])
            grid_cases.append(
                f"    case {i}: gx=dg_eval({gx},ctx); gy=dg_eval({gy},ctx); "
                f"gz=dg_eval({gz},ctx); break;  // {k['name']}"
            )
        else:
            gt = k.get("grid_type") or ""
            blocks = k["blocks"]
            if blocks is None:
                raise Unsupported(f"{k['name']}: launch config has not settled")

            def extent(numel: str, bk: str) -> tuple[int, int]:
                # exprs holds the patchable scalars; consts holds the ones baked
                # into the cubin, whose axis is simply fixed.
                e = k["exprs"].get(numel) or k["consts"].get(numel)
                if e is None or bk not in blocks:
                    raise Unsupported(f"{k['name']}: no {numel}/{bk} to size a grid")
                return idx(e), blocks[bk]

            def numel_exprs(*names: str) -> list[str]:
                return [
                    e
                    for e in (k["exprs"].get(n) or k["consts"].get(n) for n in names)
                    if e is not None
                ]

            if not any(_is_symbolic(e) for e in numel_exprs(*_grid_numel_names(k))):
                grid_cases.append(
                    f"    case {i}: static_grid = true; break;  // {k['name']}"
                )
                # Still needs its symbolic scalar params and pointer patches below.
                patches_only = True
            else:
                patches_only = False
            if patches_only:
                pass
            elif k.get("combo"):
                grid_cases.append(
                    f"    case {i}: "
                    + _combo_grid_stmt(k, lambda e: f"dg_eval({idx(e)},ctx)")
                    + f" break;  // {k['name']}"
                )
            elif gt == "Grid2DWithYZOverflow":
                # What Inductor actually emits for a tiled pointwise: the y tiles
                # are folded into z once they would pass the 65535 limit on the y
                # dimension of a launch.
                (xe, xb), (ye, yb) = (
                    extent("xnumel", "XBLOCK"),
                    extent("ynumel", "YBLOCK"),
                )
                grid_cases.append(
                    f"    case {i}: {{ int64_t raw = dg_floordiv("
                    f"dg_eval({ye},ctx) + {yb} - 1, {yb});"
                    f" int64_t div = dg_floordiv(raw + {_MAX_Y_GRID} - 1,"
                    f" {_MAX_Y_GRID});"
                    f" gx = dg_floordiv(dg_eval({xe},ctx) + {xb} - 1, {xb});"
                    f" gy = (div == 0) ? 0 : dg_floordiv(raw + div - 1, div);"
                    f" gz = div; }} break;  // {k['name']}"
                )
            else:
                axes = _GRID_AXES.get(gt)
                if axes is None:
                    raise Unsupported(
                        f"{k['name']}: grid type {k.get('grid_type')} is not modelled"
                    )
                parts = []
                for axis, (kind, numel, bk) in zip(("gx", "gy", "gz"), axes):
                    if kind == "cdiv" and numel is not None and bk is not None:
                        e_i, blk = extent(numel, bk)
                        parts.append(
                            f"{axis}=dg_floordiv(dg_eval({e_i},ctx) + {blk} - 1, {blk});"
                        )
                    elif kind == "numel":
                        e = k["exprs"].get(numel) or k["consts"].get(numel)
                        if e is None:
                            raise Unsupported(f"{k['name']}: no {numel} to size a grid")
                        parts.append(f"{axis}=dg_eval({idx(e)},ctx);")
                    else:
                        if bk is None or bk not in blocks:
                            raise Unsupported(f"{k['name']}: no {bk} to size a grid")
                        parts.append(f"{axis}={int(blocks[bk])};")
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

    n_sym = len(symbols)
    ext0 = n_sym + 2 + len(kernels)
    body0 = ext0 + n_ext
    # Past the SWITCH handles: the "inputs moved" flag and one address per
    # argument position (only the patched ones are ever read).
    in0 = body0 + 2 * n_sites
    indirty = in0 + n_in
    # Then the arena base, the slot offsets with the total, and the address
    # each arena pointer argument was last set to (`ctx_layout`).
    arena_i, off0, lastp0 = ctx_layout(
        n_sym, len(kernels), n_ext, n_sites, n_in, n_slots
    )
    v_in = n_sym + 1 + n_ext + n_sites
    n_vals = v_in + 1 + n_in + 1
    # Generated before the EXPRS table is joined: it registers expressions.
    layout_lines = []
    if slot_of:
        if fixed_off is not None:
            layout_lines = [
                f"    ctx[{off0 + slot}] = {int(v)};"
                for slot, v in enumerate(fixed_off)
            ]
        else:
            per_slot: dict[int, list[str]] = {}
            for name, (span, item) in (sizes or {}).items():
                if name in slot_of:
                    per_slot.setdefault(slot_of[name], []).append(
                        f"dg_eval({idx(span)},ctx) * {int(item)}"
                    )
            for slot in range(n_slots):
                terms = per_slot.get(slot, ["0"])
                layout_lines.append(f"    m = {terms[0]};")
                for t in terms[1:]:
                    layout_lines.append(f"    {{ int64_t v = {t}; if (v > m) m = v; }}")
                layout_lines.append(
                    f"    ctx[{off0 + slot}] = acc; acc += (m + 255) & ~(int64_t)255;"
                )
            layout_lines.append(f"    ctx[{off0 + n_slots}] = acc;")
    view_cases = generate_view_pointer_patches(
        kernels,
        slot_of,
        alias,
        views,
        itemsize_of,
        idx,
        input_bufs,
        extern_out_of,
        argv,
        patch_inputs,
        input_addr,
        in0,
    )
    setctx_params = "".join(f", int64_t v{j}" for j in range(n_vals))
    setctx_body = "\n".join(
        [f"  ctx[{j}] = v{j};" for j in range(n_sym + 1)]
        + [f"  ctx[{ext0 + t}] = v{n_sym + 1 + t};" for t in range(n_ext)]
        + [f"  ctx[{body0 + t}] = v{n_sym + 1 + n_ext + t};" for t in range(n_sites)]
        + [f"  ctx[{indirty}] = v{v_in};"]
        + [f"  ctx[{in0 + j}] = v{v_in + 1 + j};" for j in range(n_in)]
        + [f"  ctx[{arena_i}] = v{n_vals - 1};"]
    )
    ptr_cases, _n_ptr = generate_pointer_patches(
        kernels, slot_of, alias, input_bufs, extern_out_of, views
    )
    out = _PLANNER_TEMPLATE
    # Substituted rather than %-formatted: the generated C contains the modulo
    # operator, which collides with %-format placeholders.
    for tag, val in (
        ("/*@SETCTX_PARAMS@*/", setctx_params),
        ("/*@SETCTX_BODY@*/", setctx_body),
        (
            "/*@EXPRS@*/",
            "\n".join(f"    case {i}: return {c};" for i, c in enumerate(exprs)),
        ),
        ("/*@N@*/", str(len(kernels))),
        ("/*@NSYM@*/", str(len(symbols))),
        ("/*@STATE0@*/", str(len(symbols) + 2)),
        ("/*@PTRDIRTY@*/", str(len(symbols) + 1)),
        ("/*@NSITE@*/", str(n_sites)),
        ("/*@BODY0@*/", str(body0)),
        ("/*@COND0@*/", str(body0 + n_sites)),
        ("/*@INDIRTY@*/", str(indirty)),
        ("/*@ARENA@*/", str(arena_i)),
        ("/*@OFF0@*/", str(off0)),
        ("/*@LASTP0@*/", str(lastp0)),
        ("/*@LAYOUT@*/", "\n".join(layout_lines)),
        (
            "/*@INPTRS@*/",
            generate_input_pointer_patches(
                kernels, alias, argv, patch_inputs, in0, views
            ),
        ),
        ("/*@GRID@*/", "\n".join(grid_cases)),
        ("/*@PARAMS@*/", "\n".join(param_cases)),
        ("/*@PTRS@*/", ptr_cases),
        ("/*@VIEWPTRS@*/", view_cases),
        (
            "/*@EXTPTRS@*/",
            generate_extern_pointer_patches(
                kernels, alias, extern_out_of, len(symbols) + 2 + len(kernels), views
            ),
        ),
    ):
        out = out.replace(tag, val)
    return out


_HOST_TEMPLATE = r"""
// Generated by torch/_inductor/dynagraph.py -- do not edit.
// Host-side patcher for one region: given the symbol values, repoint and
// resize every kernel node that depends on them and swap the child graphs,
// all through the driver's exec-update calls, before the launch. Nothing
// runs on the device; already enqueued launches of the exec are not affected
// (CUDA guarantees this for exec updates), so this overlaps with the step
// before.
#include <cuda.h>
#include <cstdint>
#include <cstring>
#include <vector>

#define NK /*@N@*/
#define NSYM /*@NSYM@*/
#define NSITE /*@NSITE@*/
#define NEXT /*@NEXT@*/
#define NIN /*@NIN@*/
#define NSLOT /*@NSLOT@*/

static int g_dg_err = 0;

static inline int64_t dg_floordiv(int64_t a, int64_t b) {
  int64_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) --q; return q;
}
static inline int64_t dg_mod(int64_t a, int64_t b) {
  int64_t r = a % b; if (r != 0 && ((r < 0) != (b < 0))) r += b; return r;
}
static inline int64_t dg_min(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t dg_max(int64_t a, int64_t b) { return a > b ? a : b; }
#define S(i) (syms[(i)])

// The arena layout for this shape, the same arithmetic as the planner's
// setctx and the Python side: each slot is the largest buffer assigned to
// it, offsets are a prefix sum kept 256-byte aligned, the total last. Under
// a fixed layout these are the constants the build chose.
static inline void dg_layout(const int64_t* syms, int64_t* slot_off) {
  int64_t acc = 0, m;
  (void)acc; (void)m; (void)syms; (void)slot_off;
/*@LAYOUT@*/
}

struct Node {
  CUgraphNode node;
  CUDA_KERNEL_NODE_PARAMS p;
  std::vector<char> buf;   // the arguments, packed; patched by byte offset
  std::vector<void*> kp;   // kernelParams: one pointer per argument into buf
  size_t size;
  bool enabled;
};

struct Exec {
  CUgraphExec ex;
  Node nodes[NK > 0 ? NK : 1];
  CUgraphNode cnodes[NSITE > 0 ? NSITE : 1];
  CUgraph child[NSITE > 0 ? NSITE : 1];
  const char* ext[NEXT > 0 ? NEXT : 1];
  // Inputs read through their own address (no copy): the address each node
  // currently holds. Null until the first call, so the first call patches.
  const char* last_in[NIN > 0 ? NIN : 1];
};

// Find the region's kernel nodes in the captured graph by function, in launch
// order; a kernel the wrapper launched some other way (a copy) is skipped.
extern "C" void* dg_init(CUgraph graph, CUgraphExec ex, const CUfunction* funcs,
                         const int* nfuncs, const CUgraphNode* cnodes) {
  size_t n = 0;
  if (cuGraphGetNodes(graph, nullptr, &n) != CUDA_SUCCESS) return nullptr;
  std::vector<CUgraphNode> all(n);
  if (cuGraphGetNodes(graph, all.data(), &n) != CUDA_SUCCESS) return nullptr;
  Exec* e = new Exec();
  e->ex = ex;
  int k = 0, foff = 0;
  for (size_t i = 0; i < n && k < NK; ++i) {
    CUgraphNodeType t;
    if (cuGraphNodeGetType(all[i], &t) != CUDA_SUCCESS) continue;
    if (t != CU_GRAPH_NODE_TYPE_KERNEL) continue;
    CUDA_KERNEL_NODE_PARAMS p;
    if (cuGraphKernelNodeGetParams(all[i], &p) != CUDA_SUCCESS) continue;
    bool mine = false;
    for (int j = 0; j < nfuncs[k]; ++j) if (p.func == funcs[foff + j]) mine = true;
    if (!mine) continue;
    Node& nd = e->nodes[k];
    nd.node = all[i];
    nd.p = p;
    nd.enabled = true;
    size_t total = 0;
    std::vector<std::pair<size_t, size_t>> info;
    for (size_t j = 0;; ++j) {
      size_t off, sz;
      if (cuFuncGetParamInfo(p.func, j, &off, &sz) != CUDA_SUCCESS) break;
      info.push_back({off, sz});
      if (off + sz > total) total = off + sz;
    }
    nd.buf.assign(total, 0);
    nd.size = total;
    if (p.kernelParams) {
      for (size_t j = 0; j < info.size(); ++j)
        memcpy(nd.buf.data() + info[j].first, p.kernelParams[j], info[j].second);
    } else if (p.extra) {
      // Already packed by the launcher: copy the buffer it points at.
      const char* src = nullptr; size_t sz = 0;
      for (int j = 0; p.extra[j] != CU_LAUNCH_PARAM_END; j += 2) {
        if (p.extra[j] == CU_LAUNCH_PARAM_BUFFER_POINTER) src = (const char*)p.extra[j + 1];
        if (p.extra[j] == CU_LAUNCH_PARAM_BUFFER_SIZE) sz = *(size_t*)p.extra[j + 1];
      }
      if (src && sz) memcpy(nd.buf.data(), src, sz < total ? sz : total);
    }
    // kernelParams form, whatever form the launch used: a node captured from
    // a kernelParams launch (Triton's own launcher -- cooperative grids,
    // launch attributes, user kernels) rejects an update in extra form with
    // CUDA_ERROR_INVALID_VALUE, and every node accepts this one.
    nd.kp.resize(info.size());
    for (size_t j = 0; j < info.size(); ++j) nd.kp[j] = nd.buf.data() + info[j].first;
    nd.p.kernelParams = nd.kp.data();
    nd.p.extra = nullptr;
    foff += nfuncs[k];
    ++k;
  }
  if (k != NK) { delete e; return nullptr; }
  for (int s = 0; s < NSITE; ++s) {
    e->cnodes[s] = cnodes ? cnodes[s] : nullptr;
    e->child[s] = nullptr;
  }
  for (int s = 0; s < NEXT; ++s) e->ext[s] = nullptr;
  for (int i = 0; i < NIN; ++i) e->last_in[i] = nullptr;
  return e;
}

extern "C" void dg_free(void* h) { delete (Exec*)h; }

// One call per replay. Returns 0, or 1 + node index / 1000 + site index /
// 2000 for the launch, of the call that failed. Every node compares what it
// holds with what this call asks for and is only updated when something
// differs: a node whose grid and arguments carry no symbol, whose buffers
// the layout did not move and which reads no moved input is not touched
// (`ptr_dirty` forces every node once, after a capture). Inputs are read
// through `in` (one address per argument position: the input's own storage
// when it is read in place, the copy held by the region otherwise; null
// where nothing is known) and patched into the nodes that read them when
// the address moved. With `launch`, the exec is launched on `stream` at the
// end -- the whole step is this one call.
extern "C" int dg_step(void* h, const int64_t* syms, int ptr_dirty, char* arena,
                       const char* const* ext, const CUgraph* child,
                       const char* const* in, void* stream, int launch) {
  Exec* e = (Exec*)h;
  CUresult rc;
  int64_t slot_off[NSLOT + 1];
  dg_layout(syms, slot_off);
  bool in_changed[NIN > 0 ? NIN : 1];
  for (int i = 0; i < NIN; ++i) {
    in_changed[i] = in != nullptr && in[i] != nullptr && in[i] != e->last_in[i];
    if (in_changed[i]) e->last_in[i] = in[i];
  }
  (void)in_changed;
  // `ptr_dirty` also forces the children: after a re-harvest the new graph
  // may have been given the handle value the destroyed one had.
  for (int s = 0; s < NSITE; ++s) {
    if (child && child[s] && (ptr_dirty || child[s] != e->child[s]) && e->cnodes[s]) {
      rc = cuGraphExecChildGraphNodeSetParams(e->ex, e->cnodes[s], child[s]);
      if (rc != CUDA_SUCCESS) { g_dg_err = rc; return 1000 + s; }
      e->child[s] = child[s];
    }
  }
  bool ext_changed[NEXT > 0 ? NEXT : 1];
  for (int s = 0; s < NEXT; ++s) {
    ext_changed[s] = ext != nullptr && ext[s] != e->ext[s];
    if (ext_changed[s]) e->ext[s] = ext[s];
  }
  (void)ext_changed; (void)arena; (void)slot_off; (void)syms;
  int64_t gx, gy, gz;
  bool touched, zero;
  Node* nd;
/*@NODES@*/
  if (launch) {
    rc = cuGraphLaunch(e->ex, (CUstream)stream);
    if (rc != CUDA_SUCCESS) { g_dg_err = rc; return 2000; }
  }
  return 0;
}

// The CUDA error behind the last nonzero `dg_step`, for the fallback message.
extern "C" int dg_last_error(void) { return g_dg_err; }
"""


def generate_host_patcher(
    kernels: list[dict[str, Any]],
    symbols: list[str],
    slot_of: dict[str, int] | None,
    alias: dict[str, str] | None,
    input_bufs: Any,
    extern_out_of: dict[str, int] | None,
    n_sites: int,
    argv: dict[str, int] | None = None,
    patch_inputs: Any = (),
    n_ext: int | None = None,
    views: dict[str, Any] | None = None,
    itemsize_of: dict[str, int] | None = None,
    input_addr: dict[int, int] | None = None,
    sizes: dict[str, tuple[str, int]] | None = None,
    fixed_off: Sequence[int] | None = None,
) -> str:
    """The host-side counterpart of `generate_planner`: same tables, C++ on the
    host through `cuGraphExecKernelNodeSetParams` instead of a kernel on the
    device through the device graph update API. `patch_inputs` are the
    argument positions whose readers take the address from the per-call
    table. `sizes` with `slot_of` gives the per-shape arena layout;
    `fixed_off` bakes one chosen at build."""
    sym_index = {s: i for i, s in enumerate(symbols)}
    n_slots = (max(slot_of.values()) + 1) if slot_of else 0
    layout_lines = []
    if slot_of:
        if fixed_off is not None:
            layout_lines = [
                f"  slot_off[{slot}] = {int(v)};" for slot, v in enumerate(fixed_off)
            ]
        else:
            per_slot: dict[int, list[str]] = {}
            for name, (span, item) in (sizes or {}).items():
                if name in slot_of:
                    per_slot.setdefault(slot_of[name], []).append(
                        f"({_expr_to_c(span, sym_index)}) * {int(item)}"
                    )
            for slot in range(n_slots):
                terms = per_slot.get(slot, ["0"])
                layout_lines.append(f"  m = {terms[0]};")
                for t in terms[1:]:
                    layout_lines.append(f"  {{ int64_t v = {t}; if (v > m) m = v; }}")
                layout_lines.append(
                    f"  slot_off[{slot}] = acc; acc += (m + 255) & ~(int64_t)255;"
                )
            layout_lines.append(f"  slot_off[{n_slots}] = acc;")
    alias = alias or {}
    extern_out_of = extern_out_of or {}
    argv = argv or {}
    patch_inputs = OrderedSet(patch_inputs)
    blocks_of: list[str] = []
    for i, k in enumerate(kernels):
        lines = [
            f"  // ---- node {i}: {k['name']}",
            f"  nd = &e->nodes[{i}]; touched = ptr_dirty != 0; zero = false;",
        ]
        grid_stmt = None
        if k["grid"]:
            if any(_is_symbolic(e) for e in k["grid"]):
                gx, gy, gz = (_expr_to_c(e, sym_index) for e in k["grid"])
                grid_stmt = f"gx = {gx}; gy = {gy}; gz = {gz};"
        else:
            gt = k.get("grid_type") or ""
            blocks = k["blocks"]
            if blocks is None:
                raise Unsupported(f"{k['name']}: launch config has not settled")

            def extent(numel: str, bk: str) -> tuple[str, int]:
                e = k["exprs"].get(numel) or k["consts"].get(numel)
                if e is None or bk not in blocks:
                    raise Unsupported(f"{k['name']}: no {numel}/{bk} to size a grid")
                return _expr_to_c(e, sym_index), blocks[bk]

            numels = [
                e
                for e in (
                    k["exprs"].get(n) or k["consts"].get(n)
                    for n in _grid_numel_names(k)
                )
                if e is not None
            ]
            if any(_is_symbolic(e) for e in numels):
                if k.get("combo"):
                    grid_stmt = _combo_grid_stmt(k, lambda e: _expr_to_c(e, sym_index))
                elif gt == "Grid2DWithYZOverflow":
                    (xe, xb), (ye, yb) = (
                        extent("xnumel", "XBLOCK"),
                        extent("ynumel", "YBLOCK"),
                    )
                    grid_stmt = (
                        f"{{ int64_t raw = dg_floordiv({ye} + {yb} - 1, {yb});"
                        f" int64_t div = dg_floordiv(raw + {_MAX_Y_GRID} - 1, {_MAX_Y_GRID});"
                        f" gx = dg_floordiv({xe} + {xb} - 1, {xb});"
                        f" gy = (div == 0) ? 0 : dg_floordiv(raw + div - 1, div); gz = div; }}"
                    )
                else:
                    if _GRID_AXES.get(gt) is None:
                        raise Unsupported(
                            f"{k['name']}: grid type {gt} is not modelled"
                        )
                    parts = ["gx = 1; gy = 1; gz = 1;"]
                    for axis, (kind, numel, bk) in zip(
                        ("gx", "gy", "gz"), _GRID_AXES[gt]
                    ):
                        if kind == "cdiv" and numel is not None and bk is not None:
                            ce, blk = extent(numel, bk)
                            parts.append(
                                f"{axis} = dg_floordiv({ce} + {blk} - 1, {blk});"
                            )
                        elif kind == "numel":
                            e = k["exprs"].get(numel) or k["consts"].get(numel)
                            if e is None:
                                raise Unsupported(
                                    f"{k['name']}: no {numel} to size a grid"
                                )
                            parts.append(f"{axis} = {_expr_to_c(e, sym_index)};")
                        else:
                            if bk is None or bk not in blocks:
                                raise Unsupported(
                                    f"{k['name']}: no {bk} to size a grid"
                                )
                            parts.append(f"{axis} = {int(blocks[bk])};")
                    grid_stmt = " ".join(parts)
        if grid_stmt is not None:
            lines.append(f"  {grid_stmt}")
            lines.append(
                "  if (gx <= 0 || gy <= 0 || gz <= 0) {\n"
                "    zero = true;\n"
                "    if (nd->enabled) { cuGraphNodeSetEnabled(e->ex, nd->node, 0); nd->enabled = false; }\n"
                "  } else {\n"
                "    if (!nd->enabled) { cuGraphNodeSetEnabled(e->ex, nd->node, 1); nd->enabled = true; }\n"
                "    if (nd->p.gridDimX != (unsigned)gx || nd->p.gridDimY != (unsigned)gy || nd->p.gridDimZ != (unsigned)gz) {\n"
                "      nd->p.gridDimX = (unsigned)gx; nd->p.gridDimY = (unsigned)gy; nd->p.gridDimZ = (unsigned)gz; touched = true;\n"
                "    }\n"
                "  }"
            )
        for nm, e in k["exprs"].items():
            if not _is_symbolic(e):
                continue
            off, size = k["offsets"][nm]
            ct = "int32_t" if size == 4 else "int64_t"
            lines.append(
                f"  {{ {ct} v = ({ct})({_expr_to_c(e, sym_index)}); {ct}* at = ({ct}*)(nd->buf.data() + {off});"
                f" if (*at != v) {{ *at = v; touched = true; }} }}  // {nm} = {e}"
            )
        for nm, raw in (k.get("ptrs") or {}).items():
            buf, voff = _view_of(raw, alias, views or {})
            off, size = k["offsets"][nm]
            if size != 8:
                raise Unsupported(f"{k['name']} pointer {nm} is {size} bytes")
            if not _off_is_zero(voff):
                # A view at an element offset (a cat's slice), possibly
                # symbolic: base plus offset, compared and set per call.
                if buf in extern_out_of:
                    raise Unsupported(
                        f"{k['name']} argument {nm} is a view at an offset of extern output {buf}"
                    )
                item = (itemsize_of or {}).get(buf)
                if item is None:
                    raise Unsupported(
                        f"{k['name']} argument {nm}: no item size for {buf}"
                    )
                if buf in argv:
                    j = argv[buf]
                    lit = (input_addr or {}).get(j)
                    if lit is None:
                        raise Unsupported(
                            f"{k['name']} argument {nm}: no address for input {buf}"
                        )
                    base = f"(e->last_in[{j}] ? e->last_in[{j}] : (const char*)0x{lit:x}ULL)"
                else:
                    if (
                        not re.fullmatch(r"buf\d+", buf)
                        or slot_of is None
                        or buf not in slot_of
                        or buf in input_bufs
                    ):
                        raise Unsupported(
                            f"{k['name']} argument {nm} is {raw}, which no allocation owns"
                        )
                    base = f"(const char*)(arena + slot_off[{slot_of[buf]}])"
                lines.append(
                    f"  {{ const char* p = {base} + (int64_t)({_expr_to_c(voff, sym_index)}) * {item};"
                    f" const char** at = (const char**)(nd->buf.data() + {off});"
                    f" if (*at != p) {{ *at = p; touched = true; }} }}  // {nm} = {buf} + ({voff}) elements"
                )
                continue
            if buf in extern_out_of:
                site = extern_out_of[buf]
                lines.append(
                    f"  if (ext_changed[{site}]) {{ *(const char**)(nd->buf.data() + {off}) = e->ext[{site}]; touched = true; }}"
                    f"  // {nm} = {buf} (extern output)"
                )
                continue
            if buf in argv and argv[buf] in patch_inputs:
                idx = argv[buf]
                lines.append(
                    f"  if (in_changed[{idx}]) {{ *(const char**)(nd->buf.data() + {off}) = e->last_in[{idx}]; touched = true; }}"
                    f"  // {nm} = {buf} (input {idx}, by address)"
                )
                continue
            if not re.fullmatch(r"buf\d+", buf) or buf in input_bufs or slot_of is None:
                continue  # a graph input: keeps the address the capture gave it
            if buf not in slot_of:
                raise Unsupported(
                    f"{k['name']} argument {nm} is {raw}, which no allocation owns"
                )
            lines.append(
                f"  {{ char* p = arena + slot_off[{slot_of[buf]}]; char** at = (char**)(nd->buf.data() + {off});"
                f" if (*at != p) {{ *at = p; touched = true; }} }}  // {nm} = {buf}"
            )
        lines.append(
            "  if (touched && !zero) { rc = cuGraphExecKernelNodeSetParams(e->ex, nd->node, &nd->p);"
            f" if (rc != CUDA_SUCCESS) {{ g_dg_err = rc; return {1 + i}; }} }}"
        )
        # A disabled node still takes its pointer patches so it is right when
        # it comes back; the grid it keeps is the last positive one.
        lines.append(
            "  if (touched && zero) { rc = cuGraphExecKernelNodeSetParams(e->ex, nd->node, &nd->p);"
            f" if (rc != CUDA_SUCCESS) {{ g_dg_err = rc; return {1 + i}; }} }}"
        )
        blocks_of.append("\n".join(lines))
    out = _HOST_TEMPLATE
    for tag, val in (
        ("/*@N@*/", str(len(kernels))),
        ("/*@NSYM@*/", str(len(symbols))),
        ("/*@NSITE@*/", str(n_sites)),
        ("/*@NEXT@*/", str(n_sites if n_ext is None else n_ext)),
        ("/*@NIN@*/", str(len(argv))),
        ("/*@NSLOT@*/", str(n_slots)),
        ("/*@LAYOUT@*/", "\n".join(layout_lines)),
        ("/*@NODES@*/", "\n".join(blocks_of)),
    ):
        out = out.replace(tag, val)
    return out


_host_lib_cache: dict[str, Any] = {}


def _compile_host(src: str) -> Any:
    """Build the region's host patcher into a shared object and load it.

    g++ once per distinct source per machine; the object is kept under
    Inductor's cache dir like the planner cubin. libcuda is resolved from
    the process (loaded RTLD_GLOBAL here), not linked.
    """
    lib = _host_lib_cache.get(src)
    if lib is not None:
        return lib
    from torch._inductor import config
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    key = hashlib.sha256(src.encode()).hexdigest()[:32]
    base = tempfile.gettempdir() if config.force_disable_caches else cache_dir()
    d = os.path.join(base, "dynagraph")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"host_{key}.so")
    if not os.path.exists(path):
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        cpp = os.path.join(d, f"host_{key}.{os.getpid()}.cpp")
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(cpp, "w") as fh:
            fh.write(src)
        r = subprocess.run(
            [
                "g++",
                "-O2",
                "-shared",
                "-fPIC",
                "-std=c++17",
                f"-I{cuda_home}/include",
                "-o",
                tmp,
                cpp,
            ],
            capture_output=True,
            text=True,
        )
        with contextlib.suppress(OSError):
            os.remove(cpp)
        if r.returncode != 0:
            log.warning("DynaGraph host patcher g++ failed: %s", r.stderr[-800:])
            return None
        os.replace(tmp, path)
    ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(path)
    lib.dg_init.restype = ctypes.c_void_p
    lib.dg_init.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.dg_free.restype = None
    lib.dg_free.argtypes = [ctypes.c_void_p]
    lib.dg_step.restype = ctypes.c_int
    lib.dg_step.argtypes = [
        ctypes.c_void_p,  # exec state
        ctypes.c_void_p,  # symbol values
        ctypes.c_int,  # pointers dirty
        ctypes.c_void_p,  # arena base
        ctypes.c_void_p,  # extern slot addresses
        ctypes.c_void_p,  # child graphs
        ctypes.c_void_p,  # input addresses
        ctypes.c_void_p,  # stream
        ctypes.c_int,  # launch
    ]
    _host_lib_cache[src] = lib
    return lib


def compile_planner(src: str, arch: str | None = None) -> int | None:
    """Compile and load a planner source on its own; the CUfunction, or None.

    The runner goes through `_compile_module`; this is for the tests that
    build a planner by hand. Same compiler path (nvrtc, nvcc as fallback,
    cubin cached on disk), so what they measure is what the runner does.
    """
    import torch

    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}" + ("a" if (major, minor) >= (9, 0) else "")
    got = _compile_module_uncached(src, ["dynagraph_planner"], arch)
    return got[0] if got else None


def ctx_layout(
    n_sym: int, n_k: int, n_ext: int, n_sites: int, n_in: int, n_slots: int
) -> tuple[int, int, int]:
    """Where the arena base, the slot offsets and the last-set pointer
    addresses sit in a planner's ctx: past the symbols, the two flags, the
    node states, the extern slots, the SWITCH bodies and handles, the input
    addresses and the "inputs moved" flag. Returns (arena, off0, lastp0);
    `lastp0` is followed by one slot per arena pointer argument."""
    arena_i = n_sym + 2 + n_k + n_ext + 2 * n_sites + n_in + 1
    off0 = arena_i + 1
    return arena_i, off0, off0 + n_slots + 1


def launch_planner(
    func: int,
    n_nodes: int,
    handles_ptr: int,
    ctx_ptr: int,
    stream: int,
) -> None:
    """Launch the planner, one thread per graph node.

    The arena base and layout are in ctx (a zero base means shapes only: the
    generated code skips the pointer patches and every buffer keeps the
    address the capture gave it).
    """
    _launch(func, [handles_ptr, ctx_ptr], (n_nodes + 127) // 128, 128, stream)


# --------------------------------------------------------------- safety check
@functools.lru_cache(maxsize=4096)
def _parse_expr(expr: str) -> Any:
    """Parsed form of a wrapper expression, cached.

    These strings are fixed when the graph is compiled, but they are evaluated
    once per buffer per replay; re-parsing them there measured as most of the
    per-call cost in a launch-bound region, which is the one place this pass is
    supposed to be saving time.
    """
    try:
        return ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None


def _eval_int(expr: str, env: dict[str, int]) -> int | None:
    """Evaluate an arithmetic expression over symbol values, or None if it is
    not a plain arithmetic expression."""
    node = _parse_expr(expr)
    if node is None:
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

        def tup(a: str) -> list[str] | None:
            # None means "not a tuple literal, so this allocation was not
            # understood", which is different from `()` -- a 0-d tensor, whose
            # size and stride tuples are legitimately empty and which holds one
            # element. Conflating the two dropped scalar buffers from the size
            # table, and a kernel writing to one was then refused as belonging to
            # no allocation.
            a = a.strip()
            if not (a.startswith("(") and a.endswith(")")):
                return None
            return _split_args(a[1:-1])

        sizes, strides = tup(args[0]), tup(args[1])
        if sizes is None or strides is None:
            continue
        dtype = args[2].strip() if len(args) > 2 else "torch.float32"
        out.append((name, sizes, strides, dtype))
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
        if len(sizes) != len(strides):
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


def slot_sizes(
    sizes: dict[str, tuple[str, int]],
    slot_of: dict[str, int],
    n_slots: int,
    env: dict[str, int],
) -> list[int] | None:
    """Bytes each slot needs at this shape: the largest buffer assigned to it."""
    sz = [0] * n_slots
    for name, (span, itemsize) in sizes.items():
        if name not in slot_of:
            continue
        v = _eval_int(span, env)
        if v is None:
            return None
        nbytes = v * itemsize
        i = slot_of[name]
        if nbytes > sz[i]:
            sz[i] = nbytes
    return sz


def fixed_slot_offsets(
    sizes: list[int], headroom: float, per_slot: Sequence[float] | None = None
) -> tuple[list[int], list[int]]:
    """(per-slot capacity, offsets with the total last) for fixed slots.

    `per_slot` overrides the headroom of individual slots (an unbacked size
    gets more room, see `dynagraph_unbacked_headroom`)."""
    hr = list(per_slot) if per_slot is not None else [headroom] * len(sizes)
    caps = [max(int(v * h), v, 256) for v, h in zip(sizes, hr)]
    off, acc = [], 0
    for nbytes in caps:
        off.append(acc)
        acc += (nbytes + 255) & ~255
    off.append(acc)
    return caps, off


def generate_view_pointer_patches(
    kernels: list[dict[str, Any]],
    slot_of: dict[str, int] | None,
    alias: dict[str, str] | None,
    views: dict[str, Any] | None,
    itemsize_of: dict[str, int] | None,
    expr_idx: Any,
    input_bufs: Any = (),
    extern_out_of: dict[str, int] | None = None,
    argv: dict[str, int] | None = None,
    patch_inputs: Any = (),
    input_addr: dict[int, int] | None = None,
    in0: int = 0,
) -> str:
    """Per-node `case` bodies for arguments that are a view at an element offset.

    A `torch.cat` is codegened as each producer writing into
    `reinterpret_tensor(joined, ..., offset)`; the offset may carry a symbol
    (a channel offset times a spatial size), so these are recomputed on every
    planner run rather than written once: base of the owning slot, or of the
    input (its ctx address when read in place, the fixed storage otherwise),
    plus the offset in elements times the item size.
    """
    alias = alias or {}
    views = views or {}
    itemsize_of = itemsize_of or {}
    argv = argv or {}
    patch = OrderedSet(patch_inputs)
    input_addr = input_addr or {}
    cases = []
    for i, k in enumerate(kernels):
        body = []
        for nm, raw in (k.get("ptrs") or {}).items():
            buf, voff = _view_of(raw, alias, views)
            if _off_is_zero(voff):
                continue
            off, size = k["offsets"][nm]
            if size != 8:
                raise Unsupported(f"{k['name']} pointer {nm} is {size} bytes")
            if buf in (extern_out_of or {}):
                raise Unsupported(
                    f"{k['name']} argument {nm} is a view at an offset of extern output {buf}"
                )
            item = itemsize_of.get(buf)
            if item is None:
                raise Unsupported(f"{k['name']} argument {nm}: no item size for {buf}")
            if buf in argv:
                j = argv[buf]
                if j in patch:
                    base = f"(char*)ctx[{in0 + j}]"
                elif j in input_addr:
                    base = f"(char*)0x{input_addr[j]:x}ULL"
                else:
                    raise Unsupported(
                        f"{k['name']} argument {nm}: no address for input {buf}"
                    )
            else:
                if slot_of is None or buf not in slot_of or buf in input_bufs:
                    raise Unsupported(
                        f"{k['name']} argument {nm} is {raw}, which no allocation owns"
                    )
                base = f"ARENA + SLOT_OFF({slot_of[buf]})"
            body.append(
                f"      {{ char* p = {base} + (int64_t)dg_eval({expr_idx(voff)},ctx) * {item}; "
                f"cudaGraphKernelNodeSetParam(handles[i], {off}, &p, 8); }}"
                f"  // {nm} = {buf} + ({voff}) elements"
            )
        if body:
            cases.append(f"    case {i}:\n" + "\n".join(body) + "\n      break;")
    return "\n".join(cases)


def generate_input_pointer_patches(
    kernels: list[dict[str, Any]],
    alias: dict[str, str] | None,
    argv: dict[str, int] | None,
    patch_inputs: Any,
    in0: int,
    views: dict[str, Any] | None = None,
) -> str:
    """Per-node `case` bodies for arguments that are an input read in place.

    The address is the caller's tensor, per call: the host writes it into
    ctx at `in0 + position` when it moved, and this repoints the argument.
    """
    if not argv or not patch_inputs:
        return ""
    alias = alias or {}
    views = views or {}
    patch = OrderedSet(patch_inputs)
    cases = []
    for i, k in enumerate(kernels):
        body = []
        for nm, raw in (k.get("ptrs") or {}).items():
            buf, voff = _view_of(raw, alias, views)
            j = argv.get(buf)
            if j is None or j not in patch or not _off_is_zero(voff):
                continue
            off, size = k["offsets"][nm]
            if size != 8:
                raise Unsupported(f"{k['name']} pointer {nm} is {size} bytes")
            body.append(
                f"      {{ char* p = (char*)ctx[{in0 + j}]; "
                f"cudaGraphKernelNodeSetParam(handles[i], {off}, &p, 8); }}"
                f"  // {nm} = {buf} (input, read in place)"
            )
        if body:
            cases.append(f"    case {i}:\n" + "\n".join(body) + "\n      break;")
    return "\n".join(cases)


def generate_extern_pointer_patches(
    kernels: list[dict[str, Any]],
    alias: dict[str, str] | None,
    extern_out_of: dict[str, int] | None,
    ext0: int,
    views: dict[str, Any] | None = None,
) -> str:
    """Per-node `case` bodies for arguments that are an extern call's own output.

    Such a buffer lives wherever the harvested graph for the current shape put
    it, so its address is per shape: the host writes it into ctx past the node
    states (`ext0 + site`) when the shape changes, and this repoints the
    argument on every planner run rather than once.
    """
    if not extern_out_of:
        return ""
    alias = alias or {}
    views = views or {}
    cases = []
    for i, k in enumerate(kernels):
        body = []
        for nm, raw in (k.get("ptrs") or {}).items():
            buf, voff = _view_of(raw, alias, views)
            if buf not in extern_out_of:
                continue
            if not _off_is_zero(voff):
                raise Unsupported(
                    f"{k['name']} argument {nm} is a view at an offset of extern output {buf}"
                )
            off, size = k["offsets"][nm]
            if size != 8:
                raise Unsupported(f"{k['name']} pointer {nm} is {size} bytes")
            body.append(
                f"      {{ char* p = (char*)ctx[{ext0 + extern_out_of[buf]}]; "
                f"cudaGraphKernelNodeSetParam(handles[i], {off}, &p, 8); }}"
                f"  // {nm} = {buf} (extern output)"
            )
        if body:
            cases.append(f"    case {i}:\n" + "\n".join(body) + "\n      break;")
    return "\n".join(cases)


def generate_pointer_patches(
    kernels: list[dict[str, Any]],
    slot_of: dict[str, int] | None,
    alias: dict[str, str] | None = None,
    input_bufs: Any = (),
    extern_out_of: dict[str, int] | None = None,
    views: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Per-node ``case`` bodies that repoint each buffer argument into the arena.

    ``slot_of`` of None means no arena: the planner is only patching shapes and
    every buffer keeps the address the capture gave it. That is different from an
    empty assignment, which would mean an arena exists but owns nothing.

    Each argument compares the address for this layout with the one it was
    last set to (its LASTP slot in ctx, zero after a capture) and issues the
    runtime call only when they differ. Returns the cases and how many LASTP
    slots they use.

    A buffer the wrapper allocated but that is missing from the slot assignment
    is refused rather than skipped. Skipping would leave the argument at the
    address the capture gave it, and the graph would quietly return whichever
    kernel last wrote there.
    """
    if slot_of is None:
        return "", 0
    alias = alias or {}
    views = views or {}
    cases = []
    n_ptr = 0
    for i, k in enumerate(kernels):
        body = []
        for nm, raw in (k.get("ptrs") or {}).items():
            buf, voff = _view_of(raw, alias, views)
            if not re.fullmatch(r"buf\d+", buf):
                continue  # a graph input, patched from its own address
            if not _off_is_zero(voff):
                continue  # a view at an offset: generate_view_pointer_patches
            if buf in input_bufs:
                # Produced by an earlier partition and handed to this one as an
                # argument. It is an input like any other, just named for the
                # buffer it carries, so its address is patched the same way.
                continue
            if buf in (extern_out_of or {}):
                continue  # patched from ctx, see generate_extern_pointer_patches
            if buf not in slot_of:
                raise Unsupported(
                    f"{k['name']} argument {nm} is {raw}, which no allocation owns"
                )
            off, size = k["offsets"][nm]
            if size != 8:
                raise Unsupported(f"{k['name']} pointer {nm} is {size} bytes")
            body.append(
                f"      {{ char* p = ARENA + SLOT_OFF({slot_of[buf]}); "
                f"if ((int64_t)p != LASTP({n_ptr})) {{ "
                f"cudaGraphKernelNodeSetParam(handles[i], {off}, &p, 8); "
                f"LASTP({n_ptr}) = (int64_t)p; }} }}"
                f"  // {nm} = {buf}"
            )
            n_ptr += 1
        if body:
            cases.append(f"    case {i}:\n" + "\n".join(body) + "\n      break;")
    return "\n".join(cases), n_ptr


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
    """Everything the wrapper returns, in order, buffers and inputs alike.

    A partition commonly hands an input straight back -- `return (buf0, buf3,
    arg1_1)` -- because a later partition needs it. Dropping those silently
    returned a shorter tuple than the caller unpacks, which the replay check
    caught as a mismatch rather than as the structural thing it is.
    """
    m = re.search(r"^\s*return\s*\(([^)]*)\)", src, re.MULTILINE)
    if not m:
        return []
    return [a.strip() for a in _split_args(m.group(1)) if a.strip()]


def _input_names(src: str) -> dict[str, int]:
    """Name -> index for the entry function's arguments.

    Unlike :func:`_input_symbol_map` this does not insist the names look like
    `argN_1`: a partition receives buffers produced by an earlier one under
    their buffer names, so `arg1_1, buf0, buf6, s77 = args` is ordinary.
    """
    m = re.search(r"^\s*((?:\w+\s*,\s*)+\w+)\s*=\s*args\s*$", src, re.MULTILINE)
    if not m:
        return {}
    return {n.strip(): i for i, n in enumerate(m.group(1).split(","))}


def _entry_source(src: str, entry: str | None = None) -> str | None:
    """Body of the function cudagraphify is actually handed.

    Under graph partitioning compile_fx cudagraphifies each `partition_N` on its
    own through `recursively_apply_fns`, so the runtime argument order is that
    function's rather than `Runner.call`'s. The two genuinely differ -- call
    takes (weight, bias, symbol, activation) and hands the partition
    (activation, weight, bias, symbol) -- and the partition receives the symbol
    as a named entry instead of through an assignment.

    `entry` is the callable's `__name__`, which names the partition it is. That
    is what makes several partitions tractable: each one is cudagraphified
    separately and so gets its own runner, arena and graph, and the only thing
    ever needed was to read the right section of the shared file. Since any
    graph break -- `.item()`, a `.cpu()` round trip, a data-dependent branch --
    creates partitions, this is the common shape rather than an exotic one.

    Passing an already-scoped body back in is a no-op, because a partition body
    is indented and so matches nothing here.
    """
    bodies = dict(
        re.findall(
            r"^def (partition_\d+)\(args\):\n(.*?)(?=^\S)",
            src,
            re.MULTILINE | re.DOTALL,
        )
    )
    if not bodies:
        return src
    if entry in bodies:
        return bodies[entry]
    if len(bodies) == 1:
        return next(iter(bodies.values()))
    return None


# Ways the wrapper launches GPU work that is not a Triton kernel this pass can
# reach. extern_kernels is cuBLAS/cuDNN, torch.ops and aten calls are fallbacks
# to the dispatcher, and a .item() forces a device-to-host read.
_UNREACHABLE = re.compile(
    r"\bextern_kernels\s*\.|\btorch\s*\.\s*ops\s*\.|(?<![\w.])aten\s*\.\w|\.item\(\)"
)


# With the child route on, `torch.ops.<ns>.<op>.<overload>(` and
# `aten.<op>.<overload>(` calls are sites too (collectives, aten fallbacks,
# random draws), so only other uses of `aten` and `.item()` remain
# unreachable.
_UNREACHABLE_NO_EXTERN = re.compile(
    r"(?<![\w.])aten\s*\.\s*\w+\b(?!\s*\.\s*\w+\s*\()|\.item\(\)"
)

_SITE_CALL = re.compile(
    r"(?:(?P<out>buf\d+)\s*=\s*)?"
    r"(?:extern_kernels\s*\.\s*(?P<ek>\w+)"
    r"|torch\s*\.\s*ops\s*\.\s*(?P<ops>\w+\s*\.\s*\w+\s*\.\s*\w+)"
    r"|(?<![\w.])aten\s*\.\s*(?P<aten>\w+\s*\.\s*\w+))\s*\("
)

# torch.ops calls that launch nothing and only order streams. Skipped in a
# harvest and at capture alike: the child node before them is already the
# whole of what the next node depends on, and run eagerly after the collective
# was captured they would wait on an event recorded inside a capture that has
# ended, which is cudaErrorInvalidValue.
_NOOP_OPS = OrderedSet(["ops:_c10d_functional.wait_tensor.default"])
# Ops run on the host before every replay instead of being captured: their
# value must differ per call, so a node replaying them would be wrong on the
# second call. Random draws -- `aten.randint.low_out` is how Inductor seeds
# its Triton philox, and every `fallback_random` op is one of these. With
# `out=` they write the arena slot the graph reads; without, the draw is
# copied into the storage the harvest captured. One small launch per call.
_EAGER_OP_NAMES = OrderedSet(
    [
        "rand",
        "randn",
        "randint",
        "randperm",
        "bernoulli",
        "normal",
        "uniform",
        "exponential",
        "multinomial",
        "poisson",
        "random",
        "cauchy",
        "log_normal",
        "geometric",
        "native_dropout",
        "rrelu_with_noise",
    ]
)


def _is_eager_site(name: str) -> bool:
    if not name.startswith("ops:aten."):
        return False
    op = name[len("ops:aten.") :].split(".", 1)[0]
    op = op.removesuffix("_like")
    return op.rstrip("_") in _EAGER_OP_NAMES


# Topologies one extern site may take before further ones are simply recorded
# per shape upstream. cuDNN conv shows three tiers by batch; cuBLAS two.
# These are budgets of this module, not driver limits: how many bodies a
# SWITCH site may grow, how many re-captures a SWITCH runner may do, and how
# many graphs a host-selected region keeps before the least recently used
# one is dropped (`config.triton.dynagraph_max_graphs`).


def _max_graphs() -> int:
    from torch._inductor import config

    return max(1, int(config.triton.dynagraph_max_graphs))


def _site_calls(src: str, entry: str | None) -> list[tuple[str, str | None]]:
    """(site name, buffer it assigns or None) per call in the entry, in order.

    `extern_kernels.mm` is named `mm`; `torch.ops._c10d_functional.all_reduce_
    .default` is named `ops:_c10d_functional.all_reduce_.default`, and so is
    `aten.randint.low_out` (`ops:aten.randint.low_out`), the spelling the
    wrapper uses through its own `aten` global. Order is what
    identifies a site: the wrapper is straight-line, so the k-th call at run
    time is the k-th one in the source.
    """
    body = _entry_source(src, entry)
    if body is None:
        return []
    out = []
    for line in body.splitlines():
        code = line.split("#", 1)[0]
        for m in _SITE_CALL.finditer(code):
            if m.group("ek"):
                name = m.group("ek")
            elif m.group("ops"):
                name = "ops:" + re.sub(r"\s+", "", m.group("ops"))
            else:
                name = "ops:aten." + re.sub(r"\s+", "", m.group("aten"))
            out.append((name, m.group("out")))
    return out


def extern_sites(src: str, entry: str | None = None) -> list[str]:
    """Names of the extern call sites in the entry function, in order."""
    return [name for name, _ in _site_calls(src, entry)]


def extern_site_outputs(src: str, entry: str | None = None) -> list[str | None]:
    """The wrapper buffer each site assigns its result to, or None.

    A call with `out=` writes into a buffer the wrapper allocated, which the
    arena owns. A call without one -- convolution, any aten fallback -- hands
    back storage of its own, and that buffer is wherever the harvested graph
    for the current shape put it.
    """
    return [out for _, out in _site_calls(src, entry)]


def _resolve_op(path: str) -> Any:
    """`torch.ops.<ns>.<op>.<overload>` for a dotted `ns.op.overload`."""
    import torch

    obj: Any = torch.ops
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


class _OpsLevel:
    """One level of `torch.ops.<...>`, with chosen leaves swapped for wrappers."""

    def __init__(self, real: Any, calls: dict[str, Any], path: str) -> None:
        self._real, self._calls, self._path = real, calls, path

    def __getattr__(self, name: str) -> Any:
        path = f"{self._path}.{name}" if self._path else name
        if path in self._calls:
            return self._calls[path]
        return _OpsLevel(getattr(self._real, name), self._calls, path)

    def __call__(self, *a: Any, **kw: Any) -> Any:
        return self._real(*a, **kw)


class _TorchProxy:
    """`torch` as the wrapper sees it: everything real except the chosen ops.

    The wrapper reaches collectives and fallbacks as `torch.ops.<ns>.<op>
    .<overload>(...)` on its own module global, so that global is the one place
    they can be intercepted without touching `torch.ops` for everyone.
    """

    def __init__(self, real: Any, calls: dict[str, Any]) -> None:
        self._real, self._calls = real, calls

    def __getattr__(self, name: str) -> Any:
        if name == "ops":
            return _OpsLevel(self._real.ops, self._calls, "")
        return getattr(self._real, name)


def unreachable_launch(src: str, allow_extern: bool = False) -> str | None:
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
    pat = _UNREACHABLE_NO_EXTERN if allow_extern else _UNREACHABLE
    for line in body.splitlines():
        code = line.split("#", 1)[0]
        m = pat.search(code)
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
        primals_4, primals_1, mul_3, gt, s33 = args  # training forward/backward

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
    # Any identifier: inference wrappers unpack `arg3_1`, partitions add
    # `buf6`, and a training forward or backward unpacks `primals_4`,
    # `tangents_1` and saved activations under their own names (`mul_3`,
    # `gt`). Only the symbols have a fixed spelling.
    if not all(re.fullmatch(r"[A-Za-z_]\w*", n) for n in names):
        return {}
    pos = {n: i for i, n in enumerate(names)}
    out = {n: i for n, i in pos.items() if re.fullmatch(r"[su]\d+", n)}
    for m in re.finditer(r"^\s*([su]\d+)\s*=\s*([A-Za-z_]\w*)\s*$", body, re.MULTILINE):
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
    for m in re.finditer(
        r"^\s*(buf\d+)\s*=\s*(?:reinterpret_tensor\(\s*)?(buf\d+)\s*(?:;|$|,)",
        src,
        re.MULTILINE,
    ):
        alias[m.group(1)] = m.group(2)
    for k in list(alias):
        seen, v = OrderedSet(), alias[k]
        while v in alias and v not in seen:
            seen.add(v)
            v = alias[v]
        alias[k] = v
    return alias


def _view_of(raw: str, alias: dict[str, str], views: dict[str, Any]) -> tuple[str, str]:
    """The buffer owning `raw`'s storage, and `raw`'s element offset into it.

    `raw` is a call-site argument as written: a buffer name (a rename, or a
    `reinterpret_tensor` view assigned earlier, whose composed offset
    `_buffer_views` recorded), or an inline
    `reinterpret_tensor(buf, sizes, strides, offset)`. A `torch.cat` is
    codegened as its producers writing into such views of the joined buffer,
    so the offset is what keeps a kernel's output in its own slice.
    """
    raw = raw.strip()
    m = re.match(r"reinterpret_tensor\((.*)\)\s*$", raw, re.DOTALL)
    if m:
        args = [a.strip() for a in _split_args(m.group(1))]
        base = args[0] if args else raw
        off = args[3] if len(args) > 3 else "0"
        inner = views.get(base)
        if inner is not None:
            off = f"({inner[2]}) + ({off})"
        return alias.get(base, base), off
    v = views.get(raw)
    return alias.get(raw, raw), (v[2] if v is not None else "0")


def _off_is_zero(off: str) -> bool:
    return _eval_int(off, {}) == 0


def _buffer_views(src: str) -> dict[str, tuple[list[str], list[str], str]]:
    """Geometry of every `bufA = reinterpret_tensor(bufB, sizes, strides, offset)`.

    The alias table resolves such a name to the buffer owning the storage; this
    is the shape it is returned with, which is not that buffer's. A plain
    rename of a view keeps the view's geometry; a view of a view adds offsets.
    """
    views: dict[str, tuple[list[str], list[str], str]] = {}
    for ln in src.splitlines():
        m = re.match(r"\s*(buf\d+)\s*=\s*reinterpret_tensor\(", ln)
        if m:
            i = m.end()
            depth, j = 1, i
            while j < len(ln) and depth:
                if ln[j] == "(":
                    depth += 1
                elif ln[j] == ")":
                    depth -= 1
                j += 1
            args = [a.strip() for a in _split_args(ln[i : j - 1])]
            if len(args) < 4 or not (
                args[1].startswith("(") and args[2].startswith("(")
            ):
                continue
            sizes = [a.strip() for a in _split_args(args[1][1:-1]) if a.strip()]
            strides = [a.strip() for a in _split_args(args[2][1:-1]) if a.strip()]
            off = args[3]
            inner = views.get(args[0])
            if inner is not None:
                off = f"({inner[2]}) + ({off})"
            views[m.group(1)] = (sizes, strides, off)
            continue
        m = re.match(r"\s*(buf\d+)\s*=\s*(buf\d+)\s*(?:;|$)", ln)
        if m and m.group(2) in views:
            views[m.group(1)] = views[m.group(2)]
    return views


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


def _extent(t: Any) -> int:
    """Elements from the start of `t` to one past its last: what a copy of
    `t` that keeps its strides occupies. numel() for a contiguous or a
    permuted tensor; more for a view with gaps."""
    if t.numel() == 0:
        return 0
    return 1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()) if n > 1)


def _store_view(store: Any, t: Any) -> Any:
    """`t`'s geometry over the front of `store`."""
    if t.is_contiguous():
        return store[: t.numel()].view(t.shape)
    return store[: _extent(t)].as_strided(t.shape, t.stride())


class _Exec:
    """One instantiated main graph and the state that is its alone.

    Under host-side topology selection a region holds one per combination of
    extern topologies it has met (`DynaGraphRunner.execs`), picked by shape
    before the launch; under device-side SWITCH it holds one, replaced on
    re-capture. Fresh state means: every node enabled, pointers to be
    written, no shape applied, no child swapped, no shape trusted.
    """

    def __init__(self, runner: Any) -> None:
        import torch

        n_sym, n_k = len(runner.symbols), len(runner.kernels)
        n_s = len(runner.extern_sites)
        self.graph: Any = None
        self.handles = torch.zeros(n_k, dtype=torch.int64, device=runner.device)
        # Past the symbols: the host's "the shape changed" flag, then the
        # "pointers dirty" flag (1 from capture until the first replay has
        # written the buffer pointers), then one slot per node holding whether
        # it is currently enabled -- every node is, right after capture -- so
        # SetEnabled is only issued on a transition. Past the node states,
        # per extern site: the address of its output at the current shape
        # (`generate_extern_pointer_patches`), then the SWITCH body selected
        # for this shape, then the conditional handle (zero while the site is
        # a plain child node).
        # Then (`ctx_layout`): the arena base, the slot offsets and the
        # address each arena pointer argument was last set to.
        _arena_i, _off0, lastp0 = ctx_layout(
            n_sym, n_k, len(runner.ext_slots), n_s, len(runner.argv), runner.n_slots
        )
        self.ctx = torch.zeros(
            lastp0 + max(int(getattr(runner, "n_ptr", 0)), 0),
            dtype=torch.int64,
            device=runner.device,
        )
        self.ctx[n_sym + 2 : n_sym + 2 + n_k] = 1
        self.ptr_dirty = True
        # When this graph was last picked to serve a call (`_harvest` drops
        # the one used longest ago once over budget).
        self.used = 0
        # Symbol values the graph was last patched for. None until the first
        # replay, so the first one never takes the early-out.
        self.applied: Any = None
        # Whether the device-side "changed" flag is currently 1.
        self.flag_on = False
        self.child_applied: Any = None
        # Device path: the prepared arguments of the last `setctx` launch
        # (`DynaGraphRunner.ctx_args`), so an unchanged shape can re-send
        # them with the flag down.
        self.last_args: Any = None
        # Device path: the address each input read in place was last patched
        # to in this exec's nodes (0 until the first call).
        self.last_in: list[int] = [0] * len(runner.argv)
        self.child_nodes: list[Any] = []
        self.site_body_nodes: list[list[Any]] = []
        self.site_cond: list[int] = []
        self.site_held: list[list[int | None]] = []
        self.site_applied_raw: list[list[int | None]] = []
        # Shapes whose replay has already been checked against eager.
        self.verified: OrderedSet[Any] = OrderedSet()
        # Host-side path: the patcher's per-exec state (`dg_init`), and the
        # library that owns it.
        self.host: int | None = None
        self.host_lib: Any = None
        self.cnodes_host: Any = None
        # The instantiated graph's handle: launched with `cuGraphLaunch`
        # directly, not through torch's `replay()` (see `__call__`).
        self.exec_h: int = 0

    def __del__(self) -> None:
        try:
            if self.host and self.host_lib is not None:
                self.host_lib.dg_free(self.host)
        except Exception:
            pass


class _LazyAllocs:
    """Sequence that performs the wrapper's real allocation on demand.

    `_run_intercepted` iterates it in call order; each item is created with
    the arguments of that call, so the graph pool sees exactly the
    allocations it would have made.
    """

    def __init__(self, real: Any, n: int) -> None:
        self.real, self.n = real, n
        self.pending: Any = None

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        a, kw = self.pending
        self.pending = None
        return self.real(*a, **kw)


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
        import torch
        from torch._inductor import config

        self.model = model
        self.src = src
        self.device = device
        dev = torch.device(device)
        self.device_index = (
            dev.index if dev.index is not None else torch.cuda.current_device()
        )
        # The per-call tables (`_hot_tables`), made on the first call once the
        # build has settled everything they derive from; and which exec serves
        # each shape, cleared whenever a graph is captured.
        self._hot: Any = None
        self._ex_of: dict[Any, Any] = {}
        # Device path: per shape, the `setctx` launch arguments prepared once
        # (flag up, flag down), so a call is one `cuLaunchKernel`.
        self.ctx_args: dict[Any, Any] = {}
        # The graph being served, and every graph this region holds, by the
        # combination of extern topologies it was captured for (`()` with no
        # extern sites, and always under device-side SWITCH).
        self.ex: Any = None
        self.execs: dict[tuple[int, ...], Any] = {}
        self.host_mode = config.triton.dynagraph_topology != "switch"
        # Who patches the graph for a new shape: a planner kernel inside the
        # graph ("device") or generated C++ on the host before the launch
        # ("host"). "auto" is settled in `build` (`_pick_update`).
        self.update: str = config.triton.dynagraph_update
        self.host_lib: Any = None
        # Everything below reads `body`, not `src`. A partitioned wrapper holds
        # several graphs in one file and only this one's buffers belong in the
        # arena -- taking allocations, returns or lifetimes from the whole file
        # would lay out another partition's buffers alongside ours and read the
        # wrong `return` as our outputs.
        self.entry = getattr(model, "__name__", None)
        body = _entry_source(src, self.entry)
        self.body = body
        if body is None:
            self.kernels, self.symbols = [], []
            self.sizes, self.layouts, self.alias = {}, {}, {}
            self.outputs, self.out_order, self.sym_from_input = [], [], {}
            self.output_views = []
            self.argv, self.input_bufs = {}, OrderedSet()
            self.in_ptrs = (ctypes.c_void_p * 1)()
            self.extern_sites, self.alloc_order = [], []
            self.eager_sites = []
            self.ext_slots = []
            self.exact = True
            self.child_graphs, self.child_holds = {}, {}
            self.harvests = 0
            self.tick = 0
            self.n_slots, self.n_ptr = 0, 0
            self.inplace: OrderedSet[int] = OrderedSet()
            self.extern_read: OrderedSet[int] = OrderedSet()
            self.fixed_off: list[int] | None = None
            self.fixed_size: list[int] | None = None
            self.skip_keys = OrderedSet()
            self.plans = {}
            self.sym_index = {}
            return
        kernels, symbols = extract_kernel_table(body, getattr(model, "__globals__", {}))
        self.kernels: list[dict[str, Any]] = kernels or []
        self.symbols: list[str] = symbols or []
        self.exact = not any(k.get("atomic") for k in self.kernels)
        self.sizes = buffer_size_exprs(body)
        self.layouts = buffer_layouts(body)
        self.alias = _buffer_aliases(body)
        # Every view assigned in the wrapper, with its composed element
        # offset: a call-site argument may be one (`_view_of`).
        self.views = _buffer_views(body)
        # A returned buffer is often a rename of one that owns the allocation.
        # What comes back is a mix: arena buffers, and inputs handed straight
        # through. `out_order` keeps the caller's order across both, since the
        # caller unpacks the tuple positionally.
        argv = self.argv = _input_names(body)
        self.input_bufs = OrderedSet(n for n in argv if re.fullmatch(r"buf\d+", n))
        self.outputs: list[str] = []
        # Per arena output: its own geometry when it is a `reinterpret_tensor`
        # of the owning buffer, else None for the allocation's.
        self.output_views: list[Any] = []
        views = _buffer_views(body)
        self.out_order: list[tuple[bool, Any]] = []
        # Buffers that are an extern call's own output, by site.
        site_out = extern_site_outputs(src, self.entry)
        self.extern_outs_of = {out: i for i, out in enumerate(site_out) if out}
        # One address slot per site, then one per element of a tuple-valued
        # site output the wrapper picks out (`buf14 = buf13[0]`: efficient
        # attention returns several tensors). Each is an address of its own
        # at every shape, patched into the nodes that read it as a site's is.
        self.ext_slots: list[tuple[int, int | None]] = [
            (i, None) for i in range(len(site_out))
        ]
        for m in re.finditer(
            r"^\s*(buf\d+)\s*=\s*(buf\d+)\[(\d+)\]\s*$", body, re.MULTILINE
        ):
            name, owner, elem = m.group(1), m.group(2), int(m.group(3))
            if owner in self.extern_outs_of and name not in self.extern_outs_of:
                self.ext_slots.append((self.extern_outs_of[owner], elem))
                self.extern_outs_of[name] = len(self.ext_slots) - 1
        for name in _graph_outputs(body):
            owner = self.alias.get(name, name)
            if owner in self.sizes:
                self.out_order.append((True, len(self.outputs)))
                self.outputs.append(owner)
                self.output_views.append(views.get(name))
            elif name in argv or owner in argv:
                self.out_order.append((False, argv.get(name, argv.get(owner))))
            elif owner in self.extern_outs_of:
                # Handed back from the harvest for the current shape.
                self.out_order.append(("extern", self.extern_outs_of[owner]))
            else:
                # Neither ours to lay out nor ours to pass on -- refuse rather
                # than return a tuple of the wrong length.
                self.outputs, self.out_order, self.output_views = [], [], []
                break
        self.sym_from_input = _input_symbol_map(body)
        # extern_kernels.* call sites, in order. Each becomes a child-graph
        # node; see `_harvest` and `_capture`.
        self.extern_sites = extern_sites(src, self.entry)
        # Device path: where in the `setctx` values the "inputs moved" flag
        # sits (the input addresses follow it, one per argument position).
        self._ctx_in0 = (
            len(self.symbols) + 1 + len(self.ext_slots) + len(self.extern_sites)
        )
        self.eager_sites = [
            i for i, n in enumerate(self.extern_sites) if _is_eager_site(n)
        ]
        # Allocation order, so a harvest can hand each empty_strided_cuda call
        # its arena view by position.
        self.alloc_order = [n for n, _, _, _ in _find_allocations(body)]
        # key -> raw child graph per site, plus the CUDAGraph objects that own
        # them; the raw handles are only valid while those live.
        self.child_graphs: dict[Any, list[int]] = {}
        self.child_holds: dict[Any, list[Any]] = {}
        # key -> what each site returned when harvested at that shape. For a
        # site without `out=` that is the storage its output lives in.
        self.extern_outs: dict[Any, list[Any]] = {}
        # key -> site -> (fn, args, kwargs) of the ops run on the host before
        # each replay (`_EAGER_OP_NAMES`), as the harvest at that shape called
        # them.
        self.eager_calls: dict[Any, dict[int, Any]] = {}
        # Per site: the topologies seen (`_node_sig`), one body each, with
        # the first graph seen for each and the object owning it; per key,
        # which body each site's graph belongs to. Filled by `_harvest`,
        # turned into nodes by `_capture`.
        self.site_topos: list[list[Any]] = [[] for _ in self.extern_sites]
        self.site_graphs: list[list[int | None]] = [[] for _ in self.extern_sites]
        self.site_holds: list[list[Any]] = [[] for _ in self.extern_sites]
        self.key_bodies: dict[Any, list[int]] = {}
        self.recaptures = 0
        # One memory pool for every harvested graph of this region. Each
        # capture otherwise gets a private pool of its own, and a pool
        # reserves whole segments, so a long-tailed shape stream ran a
        # 12-block GEMM model out of 80 GB on the bench. Graphs sharing a
        # pool may only alias memory that is dead between them: the
        # workspaces the libraries allocate inside a capture are, and the
        # outputs a call allocates for itself are held (`extern_outs`) so
        # they are never handed out twice.
        self.harvest_pool: Any = None
        self.harvests = 0
        # Call counter, stamped on the graph that served each call (LRU).
        self.tick = 0
        # Shapes whose extern topology differs from the captured one; served
        # upstream, per shape, without retiring the region.
        self.skip_keys: OrderedSet[Any] = OrderedSet()
        # Arena offsets and output geometry per shape. Recomputing them every
        # replay is pure overhead once a shape repeats, and shapes repeat.
        self.plans: dict[tuple[tuple[str, int], ...], Any] = {}
        # Host path: per shape, the ctypes arrays `dg_step` takes; and the
        # per-call addresses of the inputs read in place (0 elsewhere).
        self.host_args: dict[Any, Any] = {}
        self.in_ptrs = (ctypes.c_void_p * max(len(self.argv), 1))()
        # Per shape: the output tensors that do not depend on the call.
        self.out_cache: dict[Any, list[Any]] = {}
        self.sym_index = {s: i for i, s in enumerate(self.symbols or [])}

    def unusable_reason(self) -> str | None:
        """The first thing that rules this graph out, or None to go ahead.

        A reason rather than a bool because these four are the commonest way a
        region is turned down, and a sweep over many models wants to know which.
        """
        # First, because a wrapper whose entry cannot be identified reports
        # empty everything else and would be counted under a cause that is not
        # its own. It now takes a partitioned wrapper whose callable does not
        # name its partition, which nothing observed does.
        if self.body is None:
            return "multi-partition"
        if not self.kernels:
            return "no-kernels"
        if not self.symbols:
            return "no-symbols"
        # `out_order`, not `outputs`: a partition can legitimately return only
        # buffers it was handed, owning no allocation of its own, and that is a
        # graph worth serving -- it still has grids and scalars to patch.
        if not self.out_order:
            return "no-outputs"
        if not self.sym_from_input:
            return "no-symbol-args"
        from torch._inductor import config

        if unreachable_launch(self.body, config.triton.dynagraph_extern_child):
            return "extern-launch"
        return None

    def build(
        self,
        inputs: list[Any],
        lifetimes: dict[str, tuple[int, int]],
        env: dict[str, int],
        static_input_idxs: Any = (),
        mutated_input_idxs: Any = (),
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
        from torch._inductor.utils import remove_unaligned_input_idxs

        headroom = config.triton.dynagraph_headroom
        # "dynamic": the arena is laid out per shape (slot sizes from the
        # symbol values, a prefix sum) and grows to the largest total seen;
        # memory is the maximum over shapes of the sum. "fixed": slot offsets
        # are chosen at build with headroom, so the device path never patches
        # a pointer twice; memory is the sum of per-slot maxima.
        self.fixed = config.triton.dynagraph_layout == "fixed"

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
        #
        # Parameters are exempt, and that exemption is most of the runtime cost
        # of this pass: a twelve-block model arrives with forty-nine tensors, of
        # which forty-eight are weights that never move. Copying all of them was
        # 452 of the 489 microseconds spent in a replay -- an order of magnitude
        # more than the replay itself. cudagraph_trees already names them through
        # `static_input_idxs`, and that is a promise about the *address*, not the
        # contents, so a captured node may read the parameter in place and still
        # see an optimizer step. The alignment filter is upstream's: an unaligned
        # input has to be copied into aligned storage anyway, because Inductor
        # specialized its kernels on the example's alignment.
        static = OrderedSet(remove_unaligned_input_idxs(inputs, static_input_idxs))
        self.input_store: list[Any] = []
        self.static_inputs = []
        # Parallel to `inputs`, as `_tensors_data_ptrs_at_indices_equal` wants it.
        self.static_ptrs: list[int | None] = []
        self.static_idxs: list[int] = []
        for i, x in enumerate(inputs):
            if not isinstance(x, torch.Tensor):
                self.input_store.append(None)
                self.static_inputs.append(x)
                self.static_ptrs.append(None)
                continue
            if i in static:
                self.input_store.append(None)
                self.static_inputs.append(x)
                self.static_ptrs.append(x.data_ptr())
                self.static_idxs.append(i)
                continue
            # Sized and viewed by the input's own strides: a transposed or
            # sliced input keeps its geometry in the copy, since the kernels
            # were specialized on that geometry and read through it. Exactly
            # this call's size: a larger one later replaces the storage and
            # the readers are repointed (`_grow_store`).
            n = _extent(x)
            store = torch.empty(max(n, 1), dtype=x.dtype, device=x.device)
            view = _store_view(store, x)
            view.copy_(x)
            self.input_store.append(store)
            self.static_inputs.append(view)
            self.static_ptrs.append(None)

        # A buffer handed in by an earlier partition can be written in place --
        # `buf8 = buf0` then `...run(buf8, ...)` with buf8 as in_out_ptr0. The
        # kernel writes into the copy held here, so the caller's tensor would be
        # left stale unless the result goes back. Detected from the kernel
        # parameter name, which is the only place read-write is stated.
        self.patch_inputs: OrderedSet[int] = OrderedSet()
        self.mutated_inputs = []
        # Every input some node writes, static or copied. The extra eager runs
        # (warmup, self-check, verification, harvest) must leave these as they
        # found them, or a BatchNorm's running stats would take several steps
        # per build. Inductor's own list covers extern calls, which the kernel
        # scan cannot see; the scan covers what the list does not name.
        self.mutated_all: list[int] = []
        written = [
            self.argv.get(_view_of(raw, self.alias, self.views)[0])
            for k in self.kernels
            for nm, raw in (k.get("ptrs") or {}).items()
            if nm.startswith(("in_out_ptr", "out_ptr"))
        ]
        written += [int(i) for i in mutated_input_idxs]
        for i in written:
            if i is None or i >= len(inputs) or not isinstance(inputs[i], torch.Tensor):
                continue
            if i not in self.mutated_all:
                self.mutated_all.append(i)
            if self.input_store[i] is not None and i not in self.mutated_inputs:
                self.mutated_inputs.append(i)

        # The warmup has to come before the planner is generated, not just before
        # the capture: it is what makes each autotuner commit to the single
        # config the graph will bake in, and the grid formula is built from that
        # config's block sizes.
        self._warmup()
        for k in self.kernels:
            k["blocks"] = settled_blocks(self.model.__globals__.get(k["gname"]))
            if k["blocks"] is None:
                return _fallback("unsettled-config", k["name"])

        sz = self._slot_sizes(env)
        if sz is None:
            return _fallback("unevaluable-size", "at build")
        self.fixed_size: list[int] | None = None
        self.fixed_off: list[int] | None = None
        if self.fixed:
            # Each slot sized for the recorded shape times the headroom; one
            # holding a buffer whose size carries an unbacked symbol gets the
            # larger headroom, since its value has no largest-first order.
            ub = float(config.triton.dynagraph_unbacked_headroom)
            per_slot = [headroom] * len(sz)
            for name, (span, _item) in self.sizes.items():
                slot = self.slot_of.get(name)
                if (
                    slot is not None
                    and slot < len(per_slot)
                    and re.search(r"\bu\d+\b", str(span))
                ):
                    per_slot[slot] = max(per_slot[slot], ub)
            self.fixed_size, self.fixed_off = fixed_slot_offsets(sz, headroom, per_slot)
        if self.update == "auto":
            self.update = self._pick_update()
        # Item sizes of everything a view can be taken of, for the pointer
        # patches of offset views.
        self.itemsize_of = {
            name: int(item) for name, (_span, item) in self.sizes.items()
        }
        for name, j in self.argv.items():
            if j < len(inputs) and isinstance(inputs[j], torch.Tensor):
                self.itemsize_of[name] = inputs[j].element_size()
        # Every tensor input is read through the per-call address table
        # (`in_ptrs`): its own storage when read in place, the copy held here
        # otherwise, the tensor itself when static. So nothing bakes an
        # input address into the graph, and a copy that has to grow, or a
        # static input that moves, is a new entry in the table.
        self.patch_inputs = OrderedSet(
            j
            for j in self.argv.values()
            if j < len(inputs) and isinstance(inputs[j], torch.Tensor)
        )
        # Inputs an extern child reads: the harvested graph holds their
        # address, so one of those moving means harvesting again.
        self.extern_read = self._extern_read_positions()
        for j in self.patch_inputs:
            store = self.input_store[j]
            self.in_ptrs[j] = (
                store.data_ptr() if store is not None else inputs[j].data_ptr()
            )
        # Of those, the ones read where they are, no copy (`__call__`).
        self.inplace: OrderedSet[int] = OrderedSet()
        # How many arena pointer arguments the planner tracks (LASTP slots).
        self.n_ptr = generate_pointer_patches(
            self.kernels,
            self.slot_of,
            self.alias,
            self.input_bufs,
            self.extern_outs_of,
            self.views,
        )[1]
        host_src: str | None = None
        planner_src: str | None = None
        try:
            if self.update == "host":
                self.inplace = self._inputs_by_address(inputs, static)
                host_src = generate_host_patcher(
                    self.kernels,
                    self.symbols,
                    self.slot_of,
                    self.alias,
                    self.input_bufs,
                    self.extern_outs_of,
                    len(self.extern_sites),
                    self.argv,
                    self.patch_inputs,
                    n_ext=len(self.ext_slots),
                    views=self.views,
                    itemsize_of=self.itemsize_of,
                    sizes=self.sizes,
                    fixed_off=self.fixed_off,
                )
                planner_src = None
            else:
                # Read in place only when the copy would cost more than a
                # planner run: large activations, a KV cache. Static inputs
                # are read where they are and checked; a moved one is
                # repointed like any other.
                thr = int(config.triton.dynagraph_patch_bytes)
                self.inplace = OrderedSet(
                    i
                    for i in self._inputs_by_address(inputs, static)
                    if thr > 0
                    and i not in static
                    and _extent(inputs[i]) * inputs[i].element_size() >= thr
                )
                planner_src = generate_planner(
                    self.kernels,
                    self.symbols,
                    self.slot_of,
                    self.alias,
                    self.input_bufs,
                    self.extern_outs_of,
                    len(self.extern_sites),
                    len(self.ext_slots),
                    argv=self.argv,
                    patch_inputs=self.patch_inputs,
                    views=self.views,
                    itemsize_of=self.itemsize_of,
                    sizes=self.sizes,
                    fixed_off=self.fixed_off,
                )
        except Unsupported as exc:
            return _fallback("unmodelled", str(exc))
        if os.environ.get("TORCHINDUCTOR_DYNAGRAPH_DUMP"):
            with open(os.environ["TORCHINDUCTOR_DYNAGRAPH_DUMP"], "w") as fh:
                fh.write(
                    f"// kernels: {self.kernels}\n// slots: {self.slot_of}\n"
                    f"// alias: {self.alias}\n// sizes: {self.sizes}\n\n"
                )
                fh.write(planner_src if planner_src is not None else (host_src or ""))
        # The source depends on the wrapper alone, not on the shape this build
        # saw: the slot offsets live in `slot_off`, written once below. So a
        # rebuild on a larger shape, and every later process compiling the
        # same region, hit a cache instead of the compiler.
        if self.update == "host":
            self.host_lib = _compile_host(host_src or "")
            if self.host_lib is None:
                return _fallback("planner-build", "host patcher")
            self.f_planner = None
            # What the patcher needs on the host: the kernel functions in
            # launch order.
            flat = [f for k in self.kernels for f in k["funcs"]]
            self.funcs_host = (ctypes.c_void_p * max(len(flat), 1))(*flat)
            self.nfuncs_host = (ctypes.c_int * max(len(self.kernels), 1))(
                *[len(k["funcs"]) for k in self.kernels]
            )
        else:
            funcs = _compile_module(
                planner_src or "", ["dynagraph_planner", "dynagraph_setctx"]
            )
            if funcs is None:
                return _fallback("planner-build")
            self.f_planner, self.f_setctx = funcs

        offsets = self.slot_offsets(env)
        if offsets is None:
            return _fallback("unevaluable-size", "layout at build")
        # The arena: what this shape needs (times the headroom under a fixed
        # layout, where that is the slots' own room). Under the dynamic layout
        # a later shape needing more gets a larger one (`_grow_arena`).
        self.arena = torch.empty(
            max(offsets[-1], 1024), dtype=torch.uint8, device=self.device
        )
        if self.extern_sites:
            key = tuple(sorted(env.items()))
            if not self._harvest(env, key, inputs):
                return False

        return self._capture() and self._replays_match(env)

    # ------------------------------------------------------------ extern
    def _extern_read_positions(self) -> OrderedSet[int]:
        """Argument positions an extern call line mentions: a harvested child
        graph holds their address, so a copy of theirs cannot move without a
        new harvest."""
        out: OrderedSet[int] = OrderedSet()
        if self.extern_sites and self.body:
            for line in self.body.splitlines():
                code = line.split("#", 1)[0]
                if _SITE_CALL.search(code):
                    for nm in re.findall(r"\b[A-Za-z_]\w*\b", code):
                        if nm in self.argv:
                            out.add(self.argv[nm])
        return out

    def _invalidate_harvests(self) -> None:
        """Forget every harvested extern graph: the addresses they hold are
        stale (the arena or a copied input moved). Shapes are harvested again
        as they recur; the graphs keep their nodes and get the new children
        swapped in."""
        self.child_graphs.clear()
        self.child_holds.clear()
        self.extern_outs.clear()
        self.key_bodies.clear()
        self.eager_calls.clear()
        self.host_args.clear()
        self.ctx_args.clear()
        self._ex_of.clear()
        self.out_cache.clear()
        for ex in self.execs.values():
            # Every child is swapped again, whatever its handle value: the
            # driver may give a new graph the handle a destroyed one had,
            # so "same handle" no longer means "same graph".
            ex.child_applied = None
            ex.applied = None
            ex.ptr_dirty = True
            ex.site_applied_raw = [[None] * len(h) for h in ex.site_held]

    def _grow_arena(self, total: int) -> None:
        """A shape needs more arena than there is: replace it with a larger
        one. Pointers are laid out per shape anyway, so every graph takes
        the new base through its next update (`ptr_dirty`); only harvested
        extern graphs, which hold absolute addresses, are redone."""
        import torch
        from torch._inductor import config

        grow = max(1.0, float(config.triton.dynagraph_grow))
        new = torch.empty(
            max(int(total * grow), total, 1024), dtype=torch.uint8, device=self.device
        )
        log.info("DynaGraph arena %d -> %d bytes", self.arena.numel(), new.numel())
        self.arena = new
        self.plans.clear()
        self.out_cache.clear()
        self.ctx_args.clear()
        self.host_args.clear()
        for ex in self.execs.values():
            ex.ptr_dirty = True
            ex.applied = None
        if self.extern_sites:
            self._invalidate_harvests()

    def _grow_store(self, j: int, x: Any, n: int) -> None:
        """The copy held for input `j` is too small for this call: replace
        it. Its readers take the new address from the table on this call;
        an extern child that read the old copy is harvested again."""
        import torch

        store = torch.empty(max(n, 1), dtype=x.dtype, device=x.device)
        log.info(
            "DynaGraph input %d storage %d -> %d elements",
            j,
            self.input_store[j].numel(),
            store.numel(),
        )
        self.input_store[j] = store
        self.static_inputs[j] = _store_view(store, x)
        self.in_ptrs[j] = store.data_ptr()
        if j in self.extern_read:
            self._invalidate_harvests()

    def _inputs_by_address(self, inputs: list[Any], static: Any) -> OrderedSet[int]:
        """Argument positions to read through their own address, no copy.

        An input that an extern child reads cannot be: the harvested graph
        holds the address it was captured with. One that very many nodes read
        is copied too, since each reader is an update call when the address
        moves (every call). Everything else is patched: it rides on the
        `SetParams` the reading node needs for the new shape anyway, no copy
        is issued, and the input's size no longer bounds anything.

        A static input (a parameter, or an activation the cudagraph tree
        keeps at one address) is in the set too: it is still read where it
        is and never touched per call, but if it does move -- a forward
        rebuilt on a larger shape hands its backward saved activations from
        a new arena -- the readers are repointed (`_rebind_static`) rather
        than the region retiring.
        """
        import torch

        extern_read = self._extern_read_positions()
        readers: dict[int, int] = {}
        for k in self.kernels:
            for raw in (k.get("ptrs") or {}).values():
                buf = _view_of(raw, self.alias, self.views)[0]
                if buf in self.argv:
                    readers[self.argv[buf]] = readers.get(self.argv[buf], 0) + 1
        out: OrderedSet[int] = OrderedSet()
        for i in self.argv.values():
            if i >= len(inputs) or not isinstance(inputs[i], torch.Tensor):
                continue
            if i in extern_read:
                continue
            if readers.get(i, 0) > 16:
                continue
            out.add(i)
        return out

    def _rebind_static(self, inputs: list[Any]) -> bool:
        """Static inputs moved: take the new addresses if every one can be.

        A reader of a moved input is repointed by the next update on either
        path (every input is in the address table). One read by an extern
        child means harvesting again; one now unaligned cannot be read in
        place at all (Inductor specialized on 16-byte alignment), so the
        caller rebuilds on this call's inputs.
        """
        import torch

        moved = [
            i
            for i in self.static_idxs
            if isinstance(inputs[i], torch.Tensor)
            and inputs[i].data_ptr() != self.static_ptrs[i]
        ]
        for i in moved:
            if inputs[i].data_ptr() % 16:
                return False
        for i in moved:
            ptr = inputs[i].data_ptr()
            self.static_ptrs[i] = ptr
            self.static_inputs[i] = inputs[i]
            self.in_ptrs[i] = ptr
        if any(i in self.extern_read for i in moved):
            self._invalidate_harvests()
        log.info("DynaGraph static inputs moved, repointed: %s", moved)
        return True

    def _shape_inputs(self, inputs: list[Any]) -> list[Any]:
        """The argument list a harvest runs the wrapper on.

        An input read in place (static, or `inplace`) is the caller's tensor
        itself; a copied one is its storage here, viewed with this call's
        shape. Symbols pass through as ints. An extern call never reads an
        in-place input (`_inputs_by_address`), so the addresses a harvest
        captures are all storage held here.
        """
        import torch

        out = []
        for i, x in enumerate(inputs):
            store = self.input_store[i] if i < len(self.input_store) else None
            if store is None or i in self.inplace or not isinstance(x, torch.Tensor):
                out.append(x)
            else:
                out.append(_store_view(store, x))
        return out

    @contextlib.contextmanager
    def _rng_kept(self) -> Any:
        """Put the CUDA random state back once the block is done.

        For a region with a seed op (`_EAGER_OP_NAMES`): the draw that counts
        is the one made on the host right before the replay, and neither the
        reference run nor a harvest may move the generator past it, or the
        two sides of the check draw different numbers.
        """
        import torch

        rng = torch.cuda.get_rng_state() if self.eager_sites else None
        try:
            yield
        finally:
            if rng is not None:
                torch.cuda.set_rng_state(rng)

    def _reference(self, inputs: list[Any]) -> Any:
        """Eager on `inputs` for a value to check the replay against.

        Written tensors are copied first (`_eager_args`), and the random state
        is put back afterwards (`_rng_kept`).
        """
        with self._rng_kept():
            return self.model(self._eager_args(inputs))

    def _eager_args(self, inputs: list[Any]) -> list[Any]:
        """`inputs` with every written tensor replaced by a copy.

        For the eager runs whose only job is a reference value: they must not
        write the caller's tensors, and the reference may alias the copy.
        """
        import torch

        out = list(inputs)
        for i in self.mutated_all:
            if i < len(out) and isinstance(out[i], torch.Tensor):
                out[i] = out[i].clone()
        return out

    @contextlib.contextmanager
    def _unwritten(self, inputs: list[Any]) -> Any:
        """Put every written tensor in `inputs` back once the block is done.

        For the runs that have to see the real storage -- a harvest captures
        its addresses -- but whose writes are not this call's.
        """
        import torch

        saved = [
            (inputs[i], inputs[i].clone())
            for i in self.mutated_all
            if i < len(inputs) and isinstance(inputs[i], torch.Tensor)
        ]
        try:
            yield
        finally:
            for t, c in saved:
                t.copy_(c)

    def _arena_views(self, env: dict[str, int]) -> list[Any] | None:
        """One tensor per allocation, in source order, at this shape's layout."""
        import torch

        offsets = self.slot_offsets(env)
        if offsets is None or offsets[self.n_slots] > self.arena.numel():
            return None
        views = []
        for name in self.alloc_order:
            span, itemsize = self.sizes[name]
            sizes_e, strides_e, dtype_name = self.layouts[name]
            vals = [_eval_int(e, env) for e in (span, *sizes_e, *strides_e)]
            if any(v is None for v in vals):
                return None
            ints = [v for v in vals if v is not None]
            n, rest = ints[0], ints[1:]
            base = offsets[self.slot_of[name]]
            dtype = getattr(torch, dtype_name.split(".")[-1])
            flat = self.arena[base : base + n * itemsize].view(dtype)
            views.append(flat.as_strided(rest[: len(sizes_e)], rest[len(sizes_e) :]))
        return views

    def _run_intercepted(self, args: list[Any], views: Any, on_extern: Any) -> Any:
        """Run the wrapper once with its allocations and extern calls redirected.

        `empty_strided_cuda` hands out `views` by position; every
        `extern_kernels.<name>` goes through `on_extern(site, fn, a, kw)`.
        The names are swapped on the wrapper's own module globals and on the
        shared `extern_kernels` namespace, and restored on the way out.
        """
        from torch._inductor.select_algorithm import extern_kernels

        g = self.model.__globals__
        it = iter(views)
        n_alloc = {"n": 0}

        def alloc(*a: Any, **kw: Any) -> Any:
            n_alloc["n"] += 1
            if isinstance(views, _LazyAllocs):
                views.pending = (a, kw)
            return next(it)

        site = {"i": 0}
        saved = {}

        def wrap(fn: Any) -> Any:
            def call(*a: Any, **kw: Any) -> Any:
                i = site["i"]
                site["i"] += 1
                return on_extern(i, fn, a, kw)

            return call

        old_alloc = g.get("empty_strided_cuda")
        g["empty_strided_cuda"] = alloc
        old_torch = g.get("torch")
        old_aten = g.get("aten")
        ops_calls = {}
        try:
            for name in OrderedSet(self.extern_sites):
                if name.startswith("ops:"):
                    ops_calls[name[4:]] = wrap(_resolve_op(name[4:]))
                    continue
                saved[name] = getattr(extern_kernels, name)
                setattr(extern_kernels, name, wrap(saved[name]))
            if ops_calls:
                g["torch"] = _TorchProxy(old_torch, ops_calls)
                if old_aten is not None:
                    g["aten"] = _OpsLevel(old_aten, ops_calls, "aten")
            out = self.model(list(args))
        finally:
            for name, fn in saved.items():
                setattr(extern_kernels, name, fn)
            if old_alloc is not None:
                g["empty_strided_cuda"] = old_alloc
            if ops_calls:
                g["torch"] = old_torch
                if old_aten is not None:
                    g["aten"] = old_aten
        if n_alloc["n"] != len(views) or site["i"] != len(self.extern_sites):
            raise Unsupported(
                f"wrapper made {n_alloc['n']} allocations and {site['i']} extern "
                f"calls; parsed {len(views)} and {len(self.extern_sites)}"
            )
        return out

    def _combo(self, key: Any) -> tuple[int, ...]:
        """Which graph serves `key`: the topology combination under host-side
        selection, the one and only graph otherwise."""
        if self.host_mode and self.extern_sites and key in self.key_bodies:
            return tuple(self.key_bodies[key])
        return ()

    def _harvest(self, env: dict[str, int], key: Any, inputs: list[Any]) -> bool:
        """Capture every extern call at this shape into its own small graph.

        The wrapper is run eagerly with the arena laid out for `env`, so the
        pointers the extern call receives are the ones the graph will use at
        that shape. Each call is warmed up once (cuBLAS wants its workspace
        before a capture) and then captured on a side stream. A node count
        not seen at a site before is a new topology, which means capturing
        the main graph again with a body for it.
        """
        import torch

        views = self._arena_views(env)
        if views is None:
            return _fallback("unevaluable-size", f"harvest at {env}")
        graphs: list[Any] = []
        results: list[Any] = []
        eager: dict[int, Any] = {}

        def on_extern(i: int, fn: Any, a: Any, kw: Any) -> Any:
            if self.extern_sites[i] in _NOOP_OPS:
                r = a[0] if a else None
                graphs.append(None)
                results.append(r)
                return r
            if _is_eager_site(self.extern_sites[i]):
                # Run, not captured: replayed on the host before every
                # launch with these very arguments (arena views and static
                # inputs, whose addresses hold).
                r = fn(*a, **kw)
                eager[i] = (fn, a, kw)
                graphs.append(None)
                results.append(kw.get("out", r))
                return results[-1]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            gen = torch.cuda.default_generators[self.device_index]
            offset = gen.get_offset()
            with torch.cuda.stream(s):
                r = fn(*a, **kw)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            if gen.get_offset() != offset:
                # The warm-up drew from the CUDA generator: a random op the
                # name list did not know (a custom op, a new overload).
                # Captured, it would draw the same numbers on every launch,
                # since the graph is launched without torch's replay
                # prologue, which is what moves the generator. So it joins
                # the seed ops, run on the host before every launch.
                if i not in self.eager_sites:
                    self.eager_sites.append(i)
                eager[i] = (fn, a, kw)
                graphs.append(None)
                results.append(kw.get("out", r))
                return results[-1]
            if (
                self.harvest_pool is None
                and os.environ.get("TORCHINDUCTOR_DYNAGRAPH_HARVEST_POOL", "1") != "0"
            ):
                self.harvest_pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(g, stream=s, pool=self.harvest_pool):
                r = fn(*a, **kw)
            torch.cuda.synchronize()
            graphs.append(g)
            results.append(r)
            return r

        shaped = self._shape_inputs(inputs)
        try:
            with self._unwritten(shaped), self._rng_kept():
                self._run_intercepted(shaped, views, on_extern)
        except Unsupported as exc:
            return _fallback("extern-harvest", str(exc))
        raws = [g.raw_cuda_graph() if g is not None else None for g in graphs]
        # Which body of each site this shape's graphs belong to. A node count
        # not seen at a site before is a new topology -- the library chose a
        # different node structure for this shape -- and gets a body of its
        # own, which means capturing the main graph again with a SWITCH at
        # that site: once per topology, not per shape (cuDNN conv has three
        # tiers by batch, cuBLAS fp32 addmm two by M).
        counts = [self._node_sig(raw) for raw in raws]
        fresh = any(n not in self.site_topos[i] for i, n in enumerate(counts))
        if (
            not self.host_mode
            and fresh
            and self.execs
            and self.recaptures >= _max_graphs()
        ):
            self.skip_keys.add(key)
            _fallback("extern-topology", f"recapture budget spent, {counts} at {env}")
            return False
        bodies = []
        for i, n in enumerate(counts):
            raw = raws[i]
            topos = self.site_topos[i]
            if n not in topos:
                if len(topos) >= _max_graphs():
                    self.skip_keys.add(key)
                    _fallback(
                        "extern-topology",
                        f"site {i} already has {len(topos)} bodies, {n} nodes at {env}",
                    )
                    return False
                topos.append(n)
                self.site_graphs[i].append(raw)
                self.site_holds[i].append(graphs[i])
                log.info("DynaGraph site %d: new topology %s at %s", i, n, env)
            bodies.append(topos.index(n))
        if self.host_mode:
            # A graph per combination of topologies, picked by shape before
            # the launch: no SWITCH node, so nothing per replay. The
            # combinations seen track the largest site's tiers, not their
            # product -- the sites step together with the shape.
            fresh = tuple(bodies) not in self.execs
            if fresh and self.execs and len(self.execs) >= _max_graphs():
                # Over budget: drop the graph used longest ago. A shape that
                # needs it again captures again (milliseconds), so no shape
                # is ever handed back to per-shape recording for this.
                combo = min(self.execs, key=lambda c: self.execs[c].used)
                gone = self.execs.pop(combo)
                for k2 in [k2 for k2, e2 in self._ex_of.items() if e2 is gone]:
                    del self._ex_of[k2]
                self.host_args.clear()
                log.info(
                    "DynaGraph dropped graph %s (%d kept) for %s at %s",
                    combo,
                    len(self.execs),
                    counts,
                    env,
                )
        # Bounded like `plans`: the shape space is long-tailed. The dropped
        # harvests hold child handles and output addresses a re-harvested
        # shape must not find (and a handle may be reused by the driver).
        if len(self.child_graphs) >= 1024:
            self._invalidate_harvests()
        self.child_graphs[key] = raws
        self.child_holds[key] = graphs
        self.extern_outs[key] = results
        self.key_bodies[key] = bodies
        self.eager_calls[key] = eager
        self.harvests += 1
        if fresh and self.execs and not self._recapture(key, inputs):
            return False
        return True

    def _recapture(self, key: Any, inputs: list[Any]) -> bool:
        """Capture another main graph: this shape's extern topologies are new.

        Under host-side selection the graph is added beside the others;
        under SWITCH it replaces the one graph with a body more. Either way
        it is a fresh `_Exec`: the next few shapes are checked against eager
        again, as after a build.
        """
        # On this shape's inputs, not the build's: the extern outputs handed
        # back to the wrapper are the ones harvested at `key`, and the wrapper
        # asserts their metadata against the shape it is running at.
        if not self._capture(key, self._shape_inputs(inputs)):
            return False
        self.recaptures += 1
        return True

    def _ext_value(self, key: Any, slot: int) -> Any:
        """What extern slot `slot` holds at `key`: a site's result, or one
        element of a tuple-valued one."""
        site, elem = self.ext_slots[slot]
        t = self.extern_outs[key][site]
        if elem is not None and isinstance(t, (tuple, list)):
            return t[elem] if elem < len(t) else None
        return t

    def _ext_ptrs(self, key: Any) -> list[int]:
        """The address in every extern slot at `key` (0 where it is not a tensor)."""
        import torch

        out = []
        for slot in range(len(self.ext_slots)):
            t = self._ext_value(key, slot)
            out.append(t.data_ptr() if isinstance(t, torch.Tensor) else 0)
        return out

    def _harvest_result(self, key: Any, i: int, a: Any, kw: Any) -> Any:
        """What the wrapper goes on using after site `i` at capture.

        With `out=` that is the buffer the call was handed, which is what the
        child node will write. Without it, the storage the harvest at this
        shape captured the call writing into.
        """
        if "out" in kw:
            return kw["out"]
        return self.extern_outs[key][i]

    def _pool_allocs(self) -> Any:
        """Deferred real allocations for the main capture, one per call site."""
        return _LazyAllocs(
            self.model.__globals__["empty_strided_cuda"], len(self.alloc_order)
        )

    @staticmethod
    def _node_sig(raw: int | None) -> tuple[Any, ...]:
        """A child graph's topology as far as an exec-level swap can follow it.

        Per node its type and, for a kernel node, its cluster dimensions and
        cooperative flag. `cudaGraphExecChildGraphNodeSetParams` carries the
        new child's functions and parameters over but keeps the launch
        configuration the exec was instantiated with: swapping in a child
        whose kernel has other cluster dimensions launches it under the old
        ones, which is an illegal instruction (cuBLAS sm90 fp32 GEMM at
        M=947 is a (1,2,1)-cluster kernel, at M=150 a (5,1,1) one). So two
        children are the same topology only if these agree; a different
        cluster shape is another body, and another graph under host-side
        selection. Two node counts equal, functions differ: still one body,
        which is the case that makes the child route cheap.
        """
        from cuda.bindings import driver as cu, runtime as cr

        from torch.cuda._utils import _check_cuda_bindings as ck

        if raw is None:
            return ()
        n = int(ck(cr.cudaGraphGetNodes(raw))[1])
        nodes = ck(cr.cudaGraphGetNodes(raw, n))[0]
        kernel = cr.cudaGraphNodeType.cudaGraphNodeTypeKernel
        sig: list[tuple[int, ...]] = []
        for nd in nodes:
            ty = ck(cr.cudaGraphNodeGetType(nd))
            if ty != kernel:
                sig.append((int(ty),))
                continue
            node = cu.CUgraphNode(int(nd))
            # Copied out before the next query: the value object is a view
            # of one union buffer, and the next attribute read overwrites it.
            cl = ck(
                cu.cuGraphKernelNodeGetAttribute(
                    node, cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                )
            ).clusterDim
            dims = (int(cl.x), int(cl.y), int(cl.z))
            coop = int(
                ck(
                    cu.cuGraphKernelNodeGetAttribute(
                        node, cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
                    )
                ).cooperative
            )
            sig.append((int(ty), *dims, coop))
        return tuple(sig)

    def _swap_children(self, key: Any) -> bool:
        """Point every child-graph node at this shape's harvested graph."""
        from cuda.bindings import runtime as cr

        from torch.cuda._utils import _check_cuda_bindings as ck

        bodies = self.key_bodies[key]
        if key == self.ex.child_applied:
            return True
        ex = self.ex.graph.raw_cuda_graph_exec()
        try:
            for i, (raw, j) in enumerate(zip(self.child_graphs[key], bodies)):
                if self.host_mode:
                    j = 0
                node = self.ex.site_body_nodes[i][j]
                if node is None or self.ex.site_applied_raw[i][j] == raw:
                    continue
                ck(cr.cudaGraphExecChildGraphNodeSetParams(ex, node, raw))
                self.ex.site_applied_raw[i][j] = raw
        except RuntimeError as exc:
            return _fallback("extern-swap", f"{exc} at {key}")
        self.ex.child_applied = key
        return True

    def slot_offsets(self, env: dict[str, int]) -> list[int] | None:
        """Byte offset of every slot, plus the total, computed on the host.

        The same prefix sum the generated code does per shape (setctx on the
        device, `dg_layout` on the host): slot size is the largest buffer
        assigned, offsets are kept 256-byte aligned. Reading it back from the
        device would cost a synchronize on every call, so the arithmetic is
        duplicated deliberately and must agree. Under a fixed layout these
        are the build's constants, and None for a shape that outgrows one of
        its slots (the caller rebuilds).
        """
        sz = self._slot_sizes(env)
        if sz is None:
            return None
        if self.fixed_off is not None:
            if self.fixed_size is not None and any(
                v > cap for v, cap in zip(sz, self.fixed_size)
            ):
                return None
            return list(self.fixed_off)
        off, acc = [], 0
        for nbytes in sz:
            off.append(acc)
            acc += (nbytes + 255) & ~255
        off.append(acc)
        return off

    def _slot_sizes(self, env: dict[str, int]) -> list[int] | None:
        return slot_sizes(self.sizes, self.slot_of, self.n_slots, env)

    def _make_plan(self, env: dict[str, int]) -> Any:
        """Where each output lands in the arena at this shape, with the
        arena bytes the shape needs; None to retire, REBUILD when a fixed
        layout cannot hold it."""
        import torch

        sz = self._slot_sizes(env)
        if sz is None:
            _fallback("unevaluable-size", f"at {env}")
            return None
        if self.fixed_size is not None:
            for i, (need, cap) in enumerate(zip(sz, self.fixed_size)):
                if need > cap:
                    _fallback(
                        "arena-too-small", f"slot {i} needs {need} > {cap} at {env}"
                    )
                    return REBUILD
        offsets = self.slot_offsets(env)
        if offsets is None:
            _fallback("unevaluable-size", f"layout at {env}")
            return None

        plan = []
        for name, view in zip(self.outputs, self.output_views):
            span, itemsize = self.sizes[name]
            sizes_e, strides_e, dtype_name = self.layouts[name]
            off_e = "0"
            if view is not None:
                sizes_e, strides_e, off_e = view
            vals = [_eval_int(e, env) for e in (span, off_e, *sizes_e, *strides_e)]
            if any(v is None for v in vals):
                _fallback("unevaluable-size", f"output {name} at {env}")
                return None
            ints = [v for v in vals if v is not None]
            n, off, rest = ints[0], ints[1], ints[2:]
            plan.append(
                (
                    offsets[self.slot_of[name]] + off * itemsize,
                    (n - off) * itemsize,
                    getattr(torch, dtype_name.split(".")[-1]),
                    rest[: len(sizes_e)],
                    rest[len(sizes_e) :],
                )
            )
        return plan, offsets[-1]

    def _host_step(self, env: dict[str, int], key: Any, stream: int) -> bool:
        """Patch the exec on the host for this call and launch it: one C++ call.

        The call touches only what differs from what the exec holds (per
        node: grid, symbolic scalars, the addresses of inputs read in place),
        swaps the children that changed, writes buffer pointers once per exec,
        and launches. Nothing here waits on the device.
        """

        ex = self.ex
        args = self.host_args.get(key)
        if args is None:
            syms = (ctypes.c_int64 * max(len(self.symbols), 1))(
                *[int(env[s]) for s in self.symbols]
            )
            n_s = len(self.extern_sites)
            if n_s:
                ext = (ctypes.c_void_p * max(len(self.ext_slots), 1))(
                    *self._ext_ptrs(key)
                )
                child = (ctypes.c_void_p * n_s)(
                    *[(g or 0) for g in self.child_graphs[key]]
                )
            else:
                ext = child = None
            args = (syms, ext, child)
            if len(self.host_args) >= 4096:
                self.host_args.clear()
            self.host_args[key] = args
        syms, ext, child = args
        rc = ex.host_lib.dg_step(
            ex.host,
            syms,
            int(ex.ptr_dirty),
            self.arena.data_ptr(),
            ext,
            child,
            self.in_ptrs,
            stream,
            1,
        )
        if rc != 0:
            return _fallback(
                "host-update",
                f"dg_step returned {rc} (CUDA error {ex.host_lib.dg_last_error()}) at {env}",
            )
        ex.applied = key
        ex.ptr_dirty = False
        ex.child_applied = key
        return True

    def _write_ctx(
        self, env: dict[str, int], key: Any, stream: int, in_changed: Sequence[int] = ()
    ) -> None:
        """Hand the planner its inputs for this shape: one `setctx` launch.

        The values ride as kernel arguments (symbols, the "changed" flag, the
        extern output addresses and SWITCH body indices for `key`, then the
        "inputs moved" flag and the addresses of the inputs read in place),
        so there is no host-to-device copy and nothing that waits. The flags
        are what let the planner skip a replay it already patched for -- that
        measured at 43.7 us of device time per replay -- and are compared
        here rather than on the device because blocks are not ordered against
        each other. A call that changes nothing only lowers the flags, once.
        """
        ex = self.ex
        same = key == ex.applied and not ex.ptr_dirty
        d = self._ctx_in0
        if same and not in_changed:
            if ex.flag_on and ex.last_args is not None:
                down = ex.last_args[1]
                if self.patch_inputs:
                    down[0][1 + d].value = 0
                _launch_prepared(self.f_setctx, ex.ctx.data_ptr(), down, stream)
                ex.flag_on = False
            return
        args = self.ctx_args.get(key)
        if args is None:
            args = self._make_ctx_args(env, key)
        up, down = args
        if self.patch_inputs:
            # The current addresses, into both arrays (the prepared values of
            # a key are per shape; the addresses are per call).
            flag = 1 if in_changed else 0
            up[0][1 + d].value = flag
            down[0][1 + d].value = flag
            in_ptrs = self.in_ptrs
            for j in self.patch_inputs:
                v = in_ptrs[j] or 0
                up[0][2 + d + j].value = v
                down[0][2 + d + j].value = v
        _launch_prepared(self.f_setctx, ex.ctx.data_ptr(), down if same else up, stream)
        ex.last_args = args
        ex.flag_on = True
        ex.applied = key

    def _make_ctx_args(self, env: dict[str, int], key: Any) -> Any:
        """The prepared `setctx` arguments for `key`: flags up, and flags down."""
        n_sym = len(self.symbols)
        vals = [int(env[sym]) for sym in self.symbols] + [1]
        if self.extern_sites and key in self.extern_outs:
            vals += self._ext_ptrs(key)
            vals += list(self.key_bodies[key])
        else:
            vals += [0] * (len(self.ext_slots) + len(self.extern_sites))
        vals += [0] * (1 + len(self.argv))
        vals.append(self.arena.data_ptr())
        down = list(vals)
        down[n_sym] = 0
        args = (_prep_vals(vals), _prep_vals(down))
        if len(self.ctx_args) >= 4096:
            self.ctx_args.clear()
        self.ctx_args[key] = args
        return args

    def _pick_update(self) -> str:
        """Host or device patching for this region, by what each costs a step.

        The host path patches every node whose grid or scalars changed before
        the launch (about 0.5 us a node in C++, `microbench/host_update.cu`),
        which the GPU never sees when it is still busy with the previous
        step; the device path costs the host one small launch and the GPU a
        planner node (7-17 us) in every step whose shape changed. So the
        host path wins when the region's own GPU time covers its patching,
        and loses by that patching when the GPU is waiting on the host. The
        region's GPU time is measured at the recorded shape from a plain
        capture, replayed; the margin stands in for the per-step Python
        around the region, which is what the GPU is waiting on when the
        step is launch-bound. A region with extern child sites is host-side
        regardless: swapping a child is 0.44 us in C++ and 2.4 us in Python.
        """
        if self.extern_sites or any(k.get("cooperative") for k in self.kernels):
            # A cooperative launch does not go through the static launcher,
            # so its node has no device handle for a planner to reach.
            return "host"
        gpu_us = self._gpu_time_us()
        host_us = 0.5 * len(self.kernels) + 60.0
        pick = "host" if gpu_us is not None and gpu_us >= host_us else "device"
        log.info(
            "DynaGraph update auto: %s (gpu %s us at the recorded shape, host patching ~%.0f us)",
            pick,
            "n/a" if gpu_us is None else f"{gpu_us:.0f}",
            host_us,
        )
        return pick

    def _gpu_time_us(self) -> float | None:
        """One replay of the region at the recorded shape, in microseconds.

        A throwaway plain capture (no planner, no patcher), timed with events
        over three replays; the median. Written inputs are put back and the
        random state kept, as for every extra run at build.
        """
        import torch

        try:
            args = self._eager_args(self.static_inputs)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with self._rng_kept():
                with torch.cuda.stream(s):
                    self.model(list(args))
                with torch.cuda.graph(g, stream=s):
                    self.model(list(args))
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            times = []
            for _ in range(3):
                e0, e1 = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                e0.record()
                g.replay()
                e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1) * 1e3)
            times.sort()
            return times[1]
        except Exception as exc:
            log.info(
                "DynaGraph update auto: timing capture failed, %s: %s",
                type(exc).__name__,
                exc,
            )
            return None

    def _warmup(self) -> None:
        import torch

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.model(self._eager_args(self.static_inputs))
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

        ref = self._reference(self.static_inputs)
        with self._unwritten(self.static_inputs):
            got = self(list(self.static_inputs))
            ok = got is not None and _same_values(got, ref, self.exact)
        if not ok:
            if got is not None:
                log.info("DynaGraph selfcheck: %s", _mismatch_report(got, ref))
            if os.environ.get("TORCHINDUCTOR_DYNAGRAPH_DEBUG"):
                self._debug_mismatch(env)
            return _fallback("selfcheck-mismatch", f"at {env}")
        return True

    def _debug_mismatch(self, env: dict[str, int]) -> None:
        """After a failed self-check: every node's grid inputs, and for a region
        without extern sites, the first arena buffer the replay got wrong.

        The replay's buffers are still in the arena; the wrapper is then run
        eagerly into the same slots and the two are compared in allocation
        order, so the first differing buffer names the first wrong node.
        Debug only (`TORCHINDUCTOR_DYNAGRAPH_DEBUG`).
        """
        import torch

        for k in self.kernels:
            vals = {n: _eval_int(e, env) for n, e in k["exprs"].items()}
            log.info(
                "DynaGraph debug kernel %s: %s blocks=%s exprs=%s consts=%s grid=%s",
                k["name"],
                k.get("grid_type"),
                k.get("blocks"),
                vals,
                k.get("consts"),
                k.get("grid"),
            )
        if self.extern_sites:
            return
        views = self._arena_views(env)
        if views is None:
            return
        replay = [v.detach().clone() for v in views]
        shaped = self._shape_inputs(self.static_inputs)
        try:
            with self._unwritten(shaped), self._rng_kept():
                self._run_intercepted(shaped, views, None)
            torch.cuda.synchronize()
        except Exception as exc:
            log.info(
                "DynaGraph debug: eager-into-arena failed, %s: %s",
                type(exc).__name__,
                exc,
            )
            return
        for name, r, e in zip(self.alloc_order, replay, views):
            if r.numel() == 0:
                continue
            d = (r.float() - e.float()).abs().reshape(-1)
            n_bad = int((d > 0).sum())
            if n_bad:
                j = int(d.argmax())
                log.info(
                    "DynaGraph debug buffer %s%s: %d/%d differ, max %.3e at flat %d (replay %.6g, eager %.6g)",
                    name,
                    tuple(r.shape),
                    n_bad,
                    d.numel(),
                    float(d[j]),
                    j,
                    float(r.reshape(-1)[j]),
                    float(e.reshape(-1)[j]),
                )
            else:
                log.info("DynaGraph debug buffer %s%s: same", name, tuple(r.shape))

    def _capture(self, key: Any = None, args: list[Any] | None = None) -> bool:
        import torch

        if args is None:
            args = list(self.static_inputs)

        launcher = torch._C._StaticCudaLauncher
        # With child-graph nodes the cudaGraph_t has to outlive instantiation:
        # cudaGraphExecChildGraphNodeSetParams finds the node in the exec by
        # the handle taken at capture, which dangles once the graph is freed
        # (it fails with invalid argument, whatever the topology).
        with_children = bool(self.extern_sites)
        graph = torch.cuda.CUDAGraph(keep_graph=with_children)
        if with_children:
            build_key = key if key is not None else next(iter(self.child_graphs))
        else:
            build_key = None
        self.ex = ex = _Exec(self)

        def on_extern(i: int, fn: Any, a: Any, kw: Any) -> Any:
            # Not run: a child-graph node holding the harvested capture is
            # added to the graph being captured, and the stream's capture
            # dependencies are moved onto it so what follows depends on it.
            from cuda.bindings import runtime as cr

            from torch.cuda._utils import _check_cuda_bindings as ck

            graphs_i = self.site_graphs[i]
            if self.child_graphs[build_key][i] is None:
                # Nothing to launch (a stream-ordering op): the wrapper just
                # goes on, and the next node already depends on the child
                # node before this one.
                self.ex.child_nodes.append(None)
                self.ex.site_body_nodes.append([None])
                self.ex.site_cond.append(0)
                self.ex.site_held.append([None])
                return self._harvest_result(build_key, i, a, kw)
            st = torch.cuda.current_stream().cuda_stream
            info = ck(cr.cudaStreamGetCaptureInfo(st))
            cap_graph, deps = info[2], list(info[3] or [])
            # What each body holds at capture: this shape's own graph in the
            # body it belongs to, the first graph seen for every other
            # topology. So the capture already stands at `build_key` and no
            # host-side swap is needed before its first launch.
            own = self.key_bodies[build_key][i]
            if self.host_mode:
                # This graph serves one combination: one body, this shape's.
                held = [self.child_graphs[build_key][i]]
            else:
                held = list(graphs_i)
                held[own] = self.child_graphs[build_key][i]
            self.ex.site_held.append(held)
            if len(held) == 1:
                node = ck(
                    cr.cudaGraphAddChildGraphNode(cap_graph, deps, len(deps), held[0])
                )
                self.ex.site_body_nodes.append([node])
                self.ex.site_cond.append(0)
            else:
                # One SWITCH, one body per topology seen, a child node in
                # each. The planner selects the body; the child inside is
                # swapped per shape like a plain child node
                # (probe_switch_py.py).
                handle = ck(cr.cudaGraphConditionalHandleCreate(cap_graph, 0, 0))
                params = cr.cudaGraphNodeParams()
                params.type = cr.cudaGraphNodeType.cudaGraphNodeTypeConditional
                params.conditional.handle = handle
                params.conditional.type = (
                    cr.cudaGraphConditionalNodeType.cudaGraphCondTypeSwitch
                )
                params.conditional.size = len(held)
                node = ck(cr.cudaGraphAddNode(cap_graph, deps, None, len(deps), params))
                bodies = params.conditional.phGraph_out
                self.ex.site_body_nodes.append(
                    [
                        ck(cr.cudaGraphAddChildGraphNode(bodies[j], None, 0, g))
                        for j, g in enumerate(held)
                    ]
                )
                self.ex.site_cond.append(int(handle))
            ck(
                cr.cudaStreamUpdateCaptureDependencies(
                    st,
                    [node],
                    None,
                    1,
                    cr.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
                )
            )
            self.ex.child_nodes.append(self.ex.site_body_nodes[-1][0])
            return self._harvest_result(build_key, i, a, kw)

        host = self.update == "host"
        if host:
            # The host patcher finds the nodes in the graph itself; the graph
            # must outlive instantiation for that, as with children.
            graph = torch.cuda.CUDAGraph(keep_graph=True)
        if not host:
            launcher._begin_device_node_collection()
        try:
            with torch.cuda.graph(graph):
                raw = torch.cuda.current_stream().cuda_stream
                if not host and self.f_planner is not None:
                    launch_planner(
                        self.f_planner,
                        len(self.kernels),
                        self.ex.handles.data_ptr(),
                        self.ex.ctx.data_ptr(),
                        raw,
                    )
                if with_children:
                    # Allocations still come from the graph pool here; only
                    # the extern calls are redirected. Their harvested graphs
                    # already carry arena addresses, and the Triton nodes are
                    # repointed into the arena by the planner as always.
                    self._run_intercepted(list(args), self._pool_allocs(), on_extern)
                else:
                    self.model(list(args))
            if with_children or host:
                graph.instantiate()
            handles = [] if host else launcher._end_device_node_collection()
        except Exception as exc:
            if not host:
                with contextlib.suppress(Exception):
                    launcher._end_device_node_collection()
            return _fallback("capture-failed", f"{type(exc).__name__}: {exc}")
        if with_children:
            self.ex.child_applied = build_key
            # What each body holds right now, and the conditional handles the
            # planner selects with (written outside the capture: a ctx write
            # is a kernel, and inside it would have been captured).
            self.ex.site_applied_raw = [list(h) for h in self.ex.site_held]
            base = (
                len(self.symbols) + 2 + len(self.kernels) + 2 * len(self.extern_sites)
            )
            self.ex.ctx[base : base + len(self.ex.site_cond)].copy_(
                torch.tensor(self.ex.site_cond, dtype=torch.int64)
            )

        if host:
            n_s = len(self.extern_sites)
            cnodes = (ctypes.c_void_p * max(n_s, 1))(
                *[int(b[0]) if b[0] is not None else 0 for b in self.ex.site_body_nodes]
                + [0] * (max(n_s, 1) - len(self.ex.site_body_nodes))
            )
            h = self.host_lib.dg_init(
                graph.raw_cuda_graph(),
                graph.raw_cuda_graph_exec(),
                self.funcs_host,
                self.nfuncs_host,
                cnodes,
            )
            if not h:
                return _fallback(
                    "handle-mismatch",
                    f"host patcher found fewer than {len(self.kernels)} kernel nodes",
                )
            self.ex.host = h
            self.ex.host_lib = self.host_lib
            self.ex.cnodes_host = cnodes
        elif len(handles) != len(self.kernels):
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
        if not host:
            self.ex.handles.copy_(torch.tensor(handles, dtype=torch.int64))
        torch.cuda.synchronize()
        self.ex.graph = graph
        ex.exec_h = graph.raw_cuda_graph_exec()
        self.execs[self._combo(build_key)] = ex
        self._ex_of.clear()
        log.info(
            "DynaGraph captured graph %d (%s update, key %s, %d kernel nodes, %d child sites)",
            len(self.execs),
            self.update,
            build_key,
            len(self.kernels),
            len(self.extern_sites),
        )
        # Upload the exec now rather than letting its first launch do it: an
        # exec with conditional nodes or external event nodes (a SWITCH, an
        # NCCL child) is uploaded lazily at first launch, and that upload
        # overwrites the device-side updates the planner makes in the same
        # launch, so the first replay after a capture ran at the captured
        # parameters (probe_cond_semantics.py, P1/P2/P3).
        try:
            from cuda.bindings import runtime as cr

            from torch.cuda._utils import _check_cuda_bindings as ck

            ck(
                cr.cudaGraphUpload(
                    graph.raw_cuda_graph_exec(), torch.cuda.current_stream().cuda_stream
                )
            )
            torch.cuda.synchronize()
        except Exception as exc:
            return _fallback("upload-failed", f"{type(exc).__name__}: {exc}")
        return True

    def _hot_tables(self) -> tuple[Any, ...]:
        """The per-call tables, derived once from what the build settled."""
        from torch._inductor import config

        sym_order = sorted(self.sym_from_input.items())
        held_idxs = [i for ok, i in self.out_order if not ok]
        held_idxs += [i for i in self.mutated_inputs if i not in held_idxs]
        out_pass = [(pos, v) for pos, (ok, v) in enumerate(self.out_order) if not ok]
        mut_copy = [i for i in self.mutated_inputs if i not in self.inplace]
        mut_patch = [i for i in self.mutated_inputs if i in self.inplace]
        return (
            sym_order,
            held_idxs,
            out_pass,
            mut_copy,
            mut_patch,
            int(config.triton.dynagraph_verify_shapes),
        )

    @staticmethod
    def _write_back(t: Any, store: Any, cu: Any, stream: int) -> None:
        """Copy what the graph wrote in `store` back into the caller's `t`."""
        if t.is_contiguous() and t.dtype is store.dtype:
            n = t.numel()
            if n:
                rc = cu.cuMemcpyDtoDAsync_v2(
                    t.data_ptr(), store.data_ptr(), n * t.element_size(), stream
                )
                if rc != 0:
                    raise RuntimeError(
                        f"DynaGraph write-back failed: cuMemcpyDtoDAsync {rc}"
                    )
            return
        t.copy_(_store_view(store, t))

    def __call__(self, inputs: list[Any]) -> Any:
        """Serve one call, or return None once this region can no longer be trusted.

        None rather than an exception: the caller still holds `inputs`, which this
        only clears on the way out, so it can just record the shape the ordinary
        way.
        """
        import torch

        hot = self._hot
        if hot is None:
            hot = self._hot = self._hot_tables()
        sym_order, held_idxs, out_pass, mut_copy, mut_patch, verify_shapes = hot
        n_in = len(inputs)
        # The symbol values in name order (`sym_order` was sorted once), so the
        # key is the same tuple as a sort of the pairs would give.
        key = tuple(
            (sym, v)
            for sym, i in sym_order
            if i < n_in and isinstance((v := inputs[i]), int)
        )
        env = dict(key)

        # Held aside because `inputs` is cleared on the way out, and these are
        # the tensors that come back out or get written back into.
        held = {i: inputs[i] for i in held_idxs}
        cu = _cuda()
        stream = torch._C._cuda_getCurrentRawStream(self.device_index)

        # Which inputs went through the store on this call (the rest, under
        # the host path, are read where they are).
        copied: list[int] = []
        in_ptrs = self.in_ptrs
        inplace = self.inplace
        for j, (store, srcv) in enumerate(zip(self.input_store, inputs)):
            if store is None or not isinstance(srcv, torch.Tensor):
                continue
            if j in inplace:
                ptr = srcv.data_ptr()
                if ptr % 16 == 0:
                    # Read in place: the nodes that read it are repointed.
                    # Inductor specialized on 16-byte alignment, so an
                    # unaligned one takes the copy below instead.
                    in_ptrs[j] = ptr
                    continue
            if srcv.is_contiguous() and srcv.dtype is store.dtype:
                n = srcv.numel()
                if n > store.numel():
                    self._grow_store(j, srcv, n)
                    store = self.input_store[j]
                # One driver call on the current stream: no aten dispatch
                # and no view built, which is what `copy_` cost.
                if n:
                    rc = cu.cuMemcpyDtoDAsync_v2(
                        store.data_ptr(),
                        srcv.data_ptr(),
                        n * srcv.element_size(),
                        stream,
                    )
                    if rc != 0:
                        _fallback("copy-failed", f"arg {j}: cuMemcpyDtoDAsync {rc}")
                        return None
            else:
                n = _extent(srcv)
                if n > store.numel():
                    self._grow_store(j, srcv, n)
                    store = self.input_store[j]
                _store_view(store, srcv).copy_(srcv)
            copied.append(j)
            if j in inplace:
                in_ptrs[j] = store.data_ptr()

        # The nodes hold these addresses, so a parameter that moved would be read
        # at its old one -- stale weights, not an error. This is the same check
        # upstream runs per call, and the same one-shot C++ helper, so it costs a
        # single call rather than one `data_ptr()` per parameter.
        if self.static_idxs and not torch._C._tensors_data_ptrs_at_indices_equal(
            inputs, self.static_ptrs, self.static_idxs
        ):
            if not self._rebind_static(inputs):
                # An unaligned one: rebuilt on this call's inputs (bounded
                # by `dynagraph_rebuilds`), since the graph would otherwise
                # read the old address, stale weights rather than an error.
                _fallback(
                    "static-input-moved", f"{len(self.static_idxs)} static inputs"
                )
                return REBUILD

        # New shapes are run eagerly alongside the replay until enough of them have
        # agreed. The build-time check only covers the shape the graph was recorded
        # at, and a grid formula can be right there and wrong everywhere else --
        # that is exactly how the XBLOCK bug hid. Eager gets `inputs`, which still
        # carries the real shape; the replay reads the fixed-size copies, so the
        # two sides are computed independently.
        # Computed on the host, which also makes the bound check preventive:
        # a shape the arena cannot hold is turned away before anything runs.
        planned = self.plans.get(key)
        if planned is None:
            planned = self._make_plan(env)
            if planned is None or planned is REBUILD:
                return planned
            # Capped: the workloads this targets have long-tailed shape
            # distributions, so the number of distinct shapes is not bounded by
            # anything. Dropping the table is fine -- it is a cache, and the
            # shapes that recur will refill it.
            if len(self.plans) >= 4096:
                self.plans.clear()
            self.plans[key] = planned
        plan, total = planned
        if total > self.arena.numel():
            # Only under the dynamic layout (a fixed one refused above).
            self._grow_arena(total)

        ex = self.ex
        if self.extern_sites:
            if key in self.skip_keys:
                return SKIP_SHAPE
            ex = self._ex_of.get(key)
            if ex is None:
                if key not in self.child_graphs and not self._harvest(env, key, inputs):
                    return SKIP_SHAPE if key in self.skip_keys else None
                # The graph for this shape's topologies; the harvest just
                # made it if it was new.
                ex = self.execs[self._combo(key)]
                if len(self._ex_of) >= 4096:
                    self._ex_of.clear()
                self._ex_of[key] = ex
            self.ex = ex
        self.tick += 1
        ex.used = self.tick

        ref = None
        if key not in ex.verified and len(ex.verified) < verify_shapes:
            ref = self._reference(inputs)

        host = self.update == "host"
        in_changed: list[int] = []
        if not host:
            if self.patch_inputs:
                last = ex.last_in
                for j in self.patch_inputs:
                    if (in_ptrs[j] or 0) != last[j]:
                        in_changed.append(j)
            self._write_ctx(env, key, stream, in_changed)
            if self.extern_sites and not self._swap_children(key):
                return None

        if self.extern_sites:
            calls = self.eager_calls.get(key)
            if calls:
                outs = self.extern_outs[key]
                for i, (fn, a, kw) in calls.items():
                    r = fn(*a, **kw)
                    if "out" not in kw:
                        outs[i].copy_(r)

        if host:
            # Patch what moved and launch, one call into the region's C++.
            if not self._host_step(env, key, stream):
                return None
        else:
            # Launched directly: torch's `replay()` also moves the CUDA
            # generator for the philox ops it captured and checks pool
            # liveness, neither of which this graph has (random ops run on
            # the host, `_harvest` checks; the pool is held here).
            rc = cu.cuGraphLaunch(ex.exec_h, stream)
            if rc != 0:
                _fallback("launch-failed", f"cuGraphLaunch returned {rc}")
                return None
            for j in in_changed:
                ex.last_in[j] = in_ptrs[j] or 0
            if ex.ptr_dirty:
                # This replay's planner wrote the buffer pointers; from here
                # on only grids and scalars are patched.
                ex.ctx[len(self.symbols) + 1] = 0
                ex.ptr_dirty = False

        # A buffer written in place lives in the storage held here, not in the
        # caller's tensor, so the caller's copy has to be caught up before it is
        # read again by whatever comes after this partition. One read by
        # address was written where it is, unless this call copied it.
        for i in mut_copy:
            self._write_back(held[i], self.input_store[i], cu, stream)
        if copied:
            for i in mut_patch:
                if i in copied:
                    self._write_back(held[i], self.input_store[i], cu, stream)

        # The arena views and extern outputs are the same tensors every call
        # at this shape (fixed slots, per-shape harvest), so they are built
        # once per shape; only an input handed back changes per call.
        fixed = self.out_cache.get(key)
        if fixed is None:
            fixed = []
            for is_arena, v in self.out_order:
                if is_arena == "extern":
                    fixed.append(self._ext_value(key, v))
                elif not is_arena:
                    fixed.append(None)
                else:
                    base, nbytes, dtype, sizes, strides = plan[v]
                    # The layout keeps every slot 256-byte aligned, so viewing
                    # the byte arena as the buffer dtype is always legal.
                    flat = self.arena[base : base + nbytes].view(dtype)
                    fixed.append(flat.as_strided(sizes, strides))
            if len(self.out_cache) >= 4096:
                self.out_cache.clear()
            self.out_cache[key] = fixed
        out = list(fixed)
        for pos, v in out_pass:
            t = held[v]
            store = self.input_store[v]
            # A static input is read in place, so it is its own answer; so is
            # one read where it is on this call. A copied one was read -- and
            # possibly written -- in the storage here, viewed with this
            # call's shape.
            if store is None or v not in copied:
                out[pos] = t
            else:
                out[pos] = _store_view(store, t)

        if ref is not None:
            if not _same_values(out, ref, self.exact):
                log.info("DynaGraph mismatch: %s", _mismatch_report(out, ref))
                _fallback("runtime-mismatch", f"at {env}")
                return None
            ex.verified.add(key)
        inputs.clear()
        return out


def _mismatch_report(got: list[Any], ref: Any) -> str:
    """Per output: the largest difference and where, for a mismatch log line."""
    import torch

    flat = list(ref) if isinstance(ref, (list, tuple)) else [ref]
    if len(got) != len(flat):
        return f"{len(got)} outputs vs {len(flat)} expected"
    parts = []
    for i, (g, r) in enumerate(zip(got, flat)):
        if not isinstance(r, torch.Tensor) or not isinstance(g, torch.Tensor):
            parts.append(f"out{i}: not a tensor")
            continue
        if g.shape != r.shape:
            parts.append(f"out{i}: shape {tuple(g.shape)} vs {tuple(r.shape)}")
            continue
        d = (g.float() - r.float()).abs().reshape(-1)
        if d.numel() == 0 or not bool((d > 0).any()):
            continue
        j = int(d.argmax())
        n_bad = int((d > 0).sum())
        parts.append(
            f"out{i}{tuple(g.shape)}: {n_bad}/{d.numel()} differ, max {float(d[j]):.3e} at flat {j}"
            f" (got {float(g.reshape(-1)[j]):.6g}, ref {float(r.reshape(-1)[j]):.6g})"
        )
    return "; ".join(parts) or "no difference found"


def _same_values(got: list[Any], ref: Any, exact: bool = True) -> bool:
    """Agreement between a replay's outputs and an eager run's.

    Bit for bit by default, on purpose: both sides run the same kernels with
    the same settled launch config on the same inputs, so any difference at
    all means the planner changed something it should not have. A region
    with a kernel that is not repeatable itself (atomics, a split scan's
    look-back) is compared with a tolerance instead.
    """
    import torch

    flat = list(ref) if isinstance(ref, (list, tuple)) else [ref]
    if len(got) != len(flat):
        return False
    for g, r in zip(got, flat):
        if not isinstance(r, torch.Tensor) or g.numel() != r.numel():
            return False
        a, b = g.reshape(-1), r.reshape(-1)
        if exact:
            if not torch.equal(a, b):
                return False
        else:
            # Run-to-run noise of such a kernel is absolute and shows up at
            # the zero crossings, so the bound is relative to the largest
            # magnitude in the reference, not element-wise.
            af, bf = a.float(), b.float()
            if bf.numel():
                tol = 1e-3 * max(1.0, float(bf.abs().max()))
                close = ((af - bf).abs() <= tol) | (torch.isnan(af) & torch.isnan(bf))
                if not bool(close.all()):
                    return False
    return True


# Loaded modules by (source, arch). A region that Dynamo recompiles on every
# call -- a Python-side step counter in the module is enough -- rebuilds its
# runner each time, and the wrapper source, hence the planner source, is the
# same every time; without this that is one nvcc run per call.
_module_cache: dict[tuple[str, str], tuple[int, ...]] = {}


def _compile_module(src: str, names: list[str]) -> tuple[int, ...] | None:
    import torch

    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}" + ("a" if (major, minor) >= (9, 0) else "")
    cached = _module_cache.get((src, arch))
    if cached is not None:
        return cached
    got = _compile_module_uncached(src, names, arch)
    if got is not None:
        _module_cache[(src, arch)] = got
    return got


def _cubin_path(src: str, arch: str) -> str | None:
    """Where this source's cubin is kept across processes; None if caching is off."""
    from torch._inductor import config
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    if config.force_disable_caches:
        return None
    key = hashlib.sha256(f"{arch}\n{src}".encode()).hexdigest()[:32]
    return os.path.join(cache_dir(), "dynagraph", f"{key}.cubin")


def _nvrtc_cubin(src: str, arch: str) -> bytes | None:
    """Compile in-process: about 30 ms, against nvcc's second (a process, cicc, ptxas)."""
    try:
        from cuda.bindings import nvrtc
    except ImportError:
        return None
    ok = nvrtc.nvrtcResult.NVRTC_SUCCESS
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    opts = [
        f"-arch={arch}".encode(),
        b"-default-device",
        b"-std=c++17",
        f"-I{cuda_home}/include".encode(),
    ]
    try:
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"dynagraph.cu", 0, [], [])
        if err != ok:
            return None
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != ok:
            err, n = nvrtc.nvrtcGetProgramLogSize(prog)
            buf = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, buf)
            log.warning(
                "DynaGraph nvrtc failed: %s", buf.decode(errors="replace")[-600:]
            )
            return None
        err, n = nvrtc.nvrtcGetCUBINSize(prog)
        if err != ok or n == 0:
            return None
        cubin = b" " * n
        (err,) = nvrtc.nvrtcGetCUBIN(prog, cubin)
        with contextlib.suppress(Exception):
            nvrtc.nvrtcDestroyProgram(prog)
        return cubin if err == ok else None
    except Exception as exc:
        log.warning("DynaGraph nvrtc failed: %s", exc)
        return None


def _nvcc_cubin(src: str, arch: str) -> bytes | None:
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
        with open(cubin, "rb") as fh:
            return fh.read()


def _compile_module_uncached(
    src: str, names: list[str], arch: str
) -> tuple[int, ...] | None:
    path = _cubin_path(src, arch)
    image = None
    if path is not None and os.path.exists(path):
        with open(path, "rb") as fh:
            image = fh.read()
    if image is None:
        image = _nvrtc_cubin(src, arch) or _nvcc_cubin(src, arch)
        if image is None:
            return None
        if path is not None:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "wb") as fh:
                    fh.write(image)
                os.replace(tmp, path)
            except OSError as exc:
                log.warning("DynaGraph cubin cache write failed: %s", exc)
    mod = ctypes.c_void_p()
    if _cuda().cuModuleLoadData(ctypes.byref(mod), image) != 0:
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


def _prep_vals(vals: list[int]) -> tuple[Any, Any]:
    """The `cuLaunchKernel` argument array for `func(ptr, vals...)`, built once.

    The holders are returned with the array because the array only points
    at them; the first holder is the pointer, set at launch.
    """
    holders: list[Any] = [ctypes.c_void_p(0)] + [ctypes.c_int64(int(v)) for v in vals]
    arr = (ctypes.c_void_p * len(holders))(
        *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders]
    )
    return holders, arr


def _launch_prepared(
    func: int, ptr: int, prepared: tuple[Any, Any], stream: int
) -> None:
    """One block, one thread: `func(ptr, vals...)` from `_prep_vals` arguments."""
    holders, arr = prepared
    holders[0].value = ptr
    rc = _cuda().cuLaunchKernel(
        ctypes.c_void_p(func), 1, 1, 1, 1, 1, 1, 0, ctypes.c_void_p(stream), arr, None
    )
    if rc != 0:
        raise RuntimeError(f"DynaGraph setctx launch failed: {rc}")


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
