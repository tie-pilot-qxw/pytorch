#!/usr/bin/env python3
"""
Code generation for the planner kernel: turn Inductor's sympy size expressions into device-side CUDA.

Background
----------
`survey/dump_grid_exprs.py` has already verified on real generated code: every Triton kernel's numel
is a closed-form arithmetic expression of symints, and the symints themselves enter the graph as plain int arguments:

    triton_poi_fused_addmm_relu_0_xnumel = 512*s77
    s77 = arg2_1

So the planner can be generated mechanically: give it this step's symint values, and it computes each node's
gridDim on the device and rewrites it in place -- this is how "one capture covers the whole range" is implemented.

Design points
-------------
**No switch-case.** Writing 3000 nodes as a switch produces a huge jump table that the instruction cache cannot hold.
The vast majority of expressions are **affine** (`c*s`, `c*s+d`), so they go through a compact table:

    struct NodeDesc { int64_t coef, konst; int32_t sym, block, param_off, flags; };

One thread handles one node and computes numel by table lookup. Only **non-affine** expressions fall into the generated
`switch` branches (a few cases: products of several symbols, FloorDiv, Max/Min, etc.).

This way the planner's code size is independent of the node count and depends only on "how many kinds of non-affine expressions there are".
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field


@dataclass
class Node:
    """A kernel node to be patched."""
    name: str                 # for debugging
    numel_expr: object        # sympy expression, or already an int
    block: int                # XBLOCK; grid = ceil(numel / block)
    param_offset: int         # byte offset of the numel argument in the kernel parameter buffer
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- sympy -> C
def sympy_to_c(expr, sym_index: dict[str, int]) -> str:
    """Convert a sympy expression into a C expression, with symbols taken from the `S(i)` macro (i.e. ctx[i]).

    Covers only the operators Inductor actually produces. On anything unrecognized it **raises instead of guessing** --
    the cost of a wrong guess is a silently wrong grid, and from that a silently wrong result, the hardest kind of bug to track down.
    """
    import sympy
    from torch.utils._sympy.functions import FloorDiv, CeilDiv, ModularIndexing

    def go(e):
        if isinstance(e, (sympy.Integer, int)):
            return f"(int64_t){int(e)}"
        if isinstance(e, sympy.Symbol):
            name = e.name
            if name not in sym_index:
                raise KeyError(f"unregistered symbol {name} in expression; "
                               f"known symbols {sorted(sym_index)}")
            return f"S({sym_index[name]})"
        if isinstance(e, sympy.Add):
            return "(" + " + ".join(go(a) for a in e.args) + ")"
        if isinstance(e, sympy.Mul):
            return "(" + " * ".join(go(a) for a in e.args) + ")"
        if isinstance(e, FloorDiv):
            return f"floor_div({go(e.args[0])}, {go(e.args[1])})"
        if isinstance(e, CeilDiv):
            return f"ceil_div({go(e.args[0])}, {go(e.args[1])})"
        if isinstance(e, ModularIndexing):
            b, d, m = e.args
            return f"(floor_div({go(b)}, {go(d)}) % {go(m)})"
        if isinstance(e, sympy.Mod):
            return f"({go(e.args[0])} % {go(e.args[1])})"
        if isinstance(e, sympy.Max):
            return "max_i64(" + ", ".join(go(a) for a in e.args) + ")"
        if isinstance(e, sympy.Min):
            return "min_i64(" + ", ".join(go(a) for a in e.args) + ")"
        if isinstance(e, sympy.Pow):
            b, p = e.args
            if isinstance(p, sympy.Integer) and 0 < int(p) <= 4:
                return "(" + " * ".join([go(b)] * int(p)) + ")"
            raise NotImplementedError(f"unsupported power: {e}")
        raise NotImplementedError(f"planner does not recognize this sympy structure: {type(e).__name__} in {e}")

    return go(expr)


def as_affine(expr, sym_index: dict[str, int]):
    """Return (coef, sym_idx, konst) if the expression can be written as coef*sym + konst, otherwise None.

    A pure constant also counts as affine, with sym_idx = -1 meaning "depends on no symbol".
    """
    import sympy
    if isinstance(expr, (int, sympy.Integer)):
        return (0, -1, int(expr))
    syms = list(expr.free_symbols) if hasattr(expr, "free_symbols") else []
    if len(syms) != 1:
        return None
    s = syms[0]
    try:
        poly = sympy.Poly(expr, s)
    except Exception:
        return None
    if poly.degree() != 1:
        return None
    coef = poly.coeff_monomial(s)
    konst = poly.coeff_monomial(1)
    if not (coef.is_Integer and konst.is_Integer):
        return None
    if s.name not in sym_index:
        raise KeyError(f"unregistered symbol {s.name}")
    return (int(coef), sym_index[s.name], int(konst))


# ---------------------------------------------------------------- codegen
_PREAMBLE = r"""
// === DynaGraph planner -- auto-generated, do not edit by hand ===
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ int64_t floor_div(int64_t a, int64_t b) {
  int64_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) --q; return q;
}
__device__ __forceinline__ int64_t ceil_div(int64_t a, int64_t b) {
  return floor_div(a + b - 1, b);
}
__device__ __forceinline__ int64_t max_i64(int64_t a, int64_t b){ return a > b ? a : b; }
__device__ __forceinline__ int64_t min_i64(int64_t a, int64_t b){ return a < b ? a : b; }

// Table lookup for affine nodes. Non-affine ones go through the generated switch (see eval_special).
struct NodeDesc {
  int64_t coef;      // numel = coef * ctx[sym] + konst   (sym < 0 means constant)
  int64_t konst;
  int32_t sym;
  int32_t block;     // grid = ceil(numel / block)
  int32_t param_off; // byte offset of the numel argument in the kernel parameter buffer
  int32_t special;   // >=0 means go through eval_special(special, ...)
};
"""

_KERNEL = r"""
// One thread per node. The planner's own cost is proportional to the node count and independent of expression complexity.
extern "C" __global__ void dynagraph_planner(
    const cudaGraphDeviceNode_t* __restrict__ handles,
    const NodeDesc* __restrict__ descs,
    const int64_t* __restrict__ ctx,
    int32_t n_nodes)
{
#define S(i) (ctx[(i)])
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_nodes) return;
  const NodeDesc d = descs[i];

  int64_t numel = (d.special >= 0)
      ? eval_special(d.special, ctx)
      : (d.sym < 0 ? d.konst : d.coef * S(d.sym) + d.konst);

  int64_t g = ceil_div(numel, (int64_t)d.block);
  if (g <= 0) {
    // grid=0 is cudaErrorInvalidArgument (measured in microbench/grid_zero.cu),
    // so an empty tensor must be expressed by disabling the node, not by setting grid to 0.
    cudaGraphKernelNodeSetEnabled(handles[i], 0);
    return;
  }
  cudaGraphKernelNodeSetEnabled(handles[i], 1);
  cudaGraphKernelNodeSetGridDim(handles[i], dim3((unsigned)g, 1, 1));
  // Triton's numel argument is int32
  int32_t n32 = (int32_t)numel;
  cudaGraphKernelNodeSetParam(handles[i], d.param_off, &n32, sizeof(int32_t));
#undef S
}
"""


def generate(nodes: list[Node], symbols: list[str]) -> tuple[str, list[dict]]:
    """Return (CUDA source, NodeDesc table). The host fills the table into a device buffer."""
    sym_index = {s: i for i, s in enumerate(symbols)}
    descs, specials = [], []

    for n in nodes:
        aff = as_affine(n.numel_expr, sym_index)
        if aff is not None:
            coef, sym, konst = aff
            descs.append(dict(coef=coef, konst=konst, sym=sym,
                              block=n.block, param_off=n.param_offset, special=-1))
        else:
            specials.append((len(specials), sympy_to_c(n.numel_expr, sym_index), n.name))
            descs.append(dict(coef=0, konst=0, sym=-1,
                              block=n.block, param_off=n.param_offset,
                              special=len(specials) - 1))

    if specials:
        body = "\n".join(
            f"    case {i}: return {c};   // {name}" for i, c, name in specials)
        special_fn = (
            "#define S(i) (ctx[(i)])\n"
            "__device__ __forceinline__ int64_t eval_special(int32_t k, "
            "const int64_t* __restrict__ ctx) {\n"
            "  switch (k) {\n" + body + "\n"
            "    default: return 0;\n"
            "  }\n}\n#undef S\n")
    else:
        # no non-affine expressions at all; emit a stub so the kernel does not reference an undefined symbol
        special_fn = ("__device__ __forceinline__ int64_t eval_special("
                      "int32_t, const int64_t*) { return 0; }\n")

    header = (f"// symbol order: {', '.join(f'{s}=ctx[{i}]' for i, s in enumerate(symbols))}\n"
              f"// node count: {len(nodes)}, of which {len(specials)} non-affine\n")
    return header + _PREAMBLE + special_fn + _KERNEL, descs


if __name__ == "__main__":
    import sympy
    s77 = sympy.Symbol("s77", positive=True, integer=True)
    demo = [
        Node("triton_poi_fused_addmm_relu_0", 512 * s77, 1024, 16),
        Node("triton_poi_fused_add_addmm_relu_1", 256 * s77, 1024, 16),
        Node("const_node", sympy.Integer(4096), 256, 16),
        Node("non_affine_example", 512 * s77 * s77 + 3, 512, 24),
    ]
    src, descs = generate(demo, ["s77"])
    print(src)
    print("// NodeDesc table:")
    for n, d in zip(demo, descs):
        print(f"//   {n.name:<34} {d}")
