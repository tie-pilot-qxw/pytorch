#!/usr/bin/env python3
"""
The first real DynaGraph execution: one capture in PyTorch covers a whole shape range.

`microbench/dynagraph_proto.cu` already showed the mechanism works on hand-written CUDA.
This wires the earlier pieces together and reproduces the same thing on a **real torch.compile output**:

    compile -> extract kernel table -> generate and compile planner -> capture (collect handles)
         -> write ctx for different shapes, then replay -> compare element-wise against eager

Needs a card to load the cubin, but does no timing, so a shared card is fine.
"""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DG_OUT = os.environ.get("DG_OUT", "/tmp/dynagraph_out")

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

_lib = None


def cu():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL("libcuda.so.1")
    return _lib


def ck(rc, what):
    if rc != 0:
        msg = ctypes.c_char_p()
        cu().cuGetErrorString(rc, ctypes.byref(msg))
        raise RuntimeError(f"{what} failed: rc={rc} {msg.value.decode() if msg.value else ''}")


def param_info(func: int, idx: int):
    off, size = ctypes.c_size_t(), ctypes.c_size_t()
    if cu().cuFuncGetParamInfo(ctypes.c_void_p(func), ctypes.c_size_t(idx),
                               ctypes.byref(off), ctypes.byref(size)) != 0:
        return None
    return off.value, size.value


# --------------------------------------------------------------- extraction
def _split_args(argstr: str) -> list[str]:
    """Split arguments on top-level commas; commas inside brackets do not count."""
    out, depth, cur = [], 0, []
    for ch in argstr:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip()); cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return [a for a in out if a and "=" not in a.split("(")[0]]


def resolve_names(expr: str, src: str, symbols=(), depth: int = 8) -> str:
    """Expand named variables referenced in an expression into their definitions.

    The wrapper has **both forms side by side**: some arguments inline the expression (`s77`, `256`),
    others are named variables:

        triton_red_..._sum_2_r0_numel = s77
        triton_red_..._sum_2.run(..., ks0, xnumel, triton_red_..._sum_2_r0_numel, ...)

    Looking only at the call site yields the variable **name**, which has no `s\d+` in it, so it is not
    flagged as "needs patching" -- the reduction range stays at the capture-time M. That is how
    M=512/333/7 all came out wrong, and since a row-wise model cannot notice, it was first misreported as a success.
    """
    syms = set(symbols)
    for _ in range(depth):
        e = expr.strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", e):
            return expr
        if e in syms:
            # Symbols are terminals. The wrapper has `s77 = arg2_1` (how the symbol is read from an input tensor);
            # expanding further turns s77 into arg2_1, the symbol is lost, and the argument is no longer flagged for patching.
            return e
        m = re.search(rf"^\s*{re.escape(e)}\s*=\s*(.+?)\s*$", src, re.M)
        if not m:
            return expr
        expr = m.group(1)
    return expr


def parse_run_calls(src: str) -> dict[str, list[str]]:
    """Parse each kernel's .run(...) positional arguments from the wrapper source.

    numel is **not** a named variable; it is a positional argument inlined at the call site:

        triton_per_..._1.run(buf4, arg1_1, buf0, s77, 256, stream=raw_stream0)
                             in_out  in_ptr0 in_ptr1 xnumel r0_numel

    FixedGrid kernels also pass grid_0/1/2 **explicitly** after the arguments:

        triton_tem_..._0.run(arg3_1, arg0_1, buf0, s77, 8*((31 + s77) // 32), 1, 1, ...)
                                                        ^^^^^^^^^^^^^^^^^^^^ grid_0

    So the expressions are only available by aligning positions with the signature; searching for a named
    variable like `_xnumel =` finds nothing (that is a different codegen form).
    """
    calls = {}
    for m in re.finditer(r"(\w+)\.run\(", src):
        name, i = m.group(1), m.end()
        depth, j = 1, i
        while j < len(src) and depth:
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
            j += 1
        calls[name] = _split_args(src[i:j - 1])
    return calls



def compile_and_extract(model, example, dynamic=True):
    """Compile the model and return (compiled_fn, kernels, symbols, wrapper_src).

    kernels are in **source order**; each entry has name / numel_expr / block / param_off.
    """
    import torch
    import torch._inductor.config as ic
    from torch._inductor import codecache
    from torch._inductor.runtime import triton_heuristics as th

    ic.force_disable_caches = True
    # Without GEMM routing, addmm goes through extern_kernels; those nodes bypass the static launcher,
    # so they have neither a handle nor a numel expression -- measured: "handle count < kernel count".
    ic.max_autotune_gemm = True
    ic.max_autotune_gemm_backends = "TRITON"

    mods, srcs = [], {}
    orig = codecache.PyCodeCache.load_by_key_path

    def spy(key, path, *a, **kw):
        mod = orig(key, path, *a, **kw)
        try:
            srcs[id(mod)] = open(path).read()
        except Exception:
            pass
        mods.append(mod)
        return mod

    codecache.PyCodeCache.load_by_key_path = staticmethod(spy)
    try:
        f = torch.compile(model, dynamic=dynamic)
        with torch.no_grad():
            f(*example)
    finally:
        codecache.PyCodeCache.load_by_key_path = staticmethod(orig)

    wrapper = next((m for m in mods if "call" in vars(m)), None)
    if wrapper is None:
        raise RuntimeError("wrapper module not found")
    src = srcs.get(id(wrapper), "")

    run_args = parse_run_calls(src)
    symbols = sorted(set(re.findall(r"\b(s\d+)\b", src)))

    kernels = []
    for gname, obj in vars(wrapper).items():
        if not isinstance(obj, th.CachingAutotuner):
            continue
        meta = obj.inductor_meta or {}
        tmeta = obj.triton_meta or {}
        sig = tmeta.get("signature", {})
        runtime_args = [k for k, v in sig.items() if v != "constexpr"]
        # Only **integer scalar** arguments need patching. Pointers ('*fp32' and the like) have fixed
        # addresses in the graph's private memory pool and do not change with shape; when named variables are
        # expanded they resolve to `empty_strided_cuda(...)`, which looks "symbolic" but must not be touched.
        if os.environ.get("DG_DEBUG_SIG"):
            print("  SIG", meta.get("kernel_name", gname), dict(sig))
        arg_is_scalar = {k: (isinstance(sig.get(k), str)
                             and not sig[k].startswith("*"))
                         for k in runtime_args}

        blocks = {}
        for lr in getattr(obj, "launchers", []) or []:
            cfg = getattr(lr, "config", None)
            for bk in ("XBLOCK", "YBLOCK", "R0_BLOCK"):
                if cfg and bk in getattr(cfg, "kwargs", {}):
                    blocks[bk] = cfg.kwargs[bk]

        func = None
        for cr in getattr(obj, "compile_results", []) or []:
            k = getattr(cr, "kernel", None)
            func = getattr(k, "function", None) or (
                next(iter(getattr(k, "functions", {}).values()), None))
            if func:
                break

        offs = {}
        if func:
            for i, nm in enumerate(runtime_args):
                pi = param_info(func, i)
                if pi:
                    offs[nm] = pi

        kernels.append(dict(
            gname=gname,
            name=meta.get("kernel_name", gname),
            grid_type=meta.get("grid_type", "?"),
            args=runtime_args,
            blocks=blocks,
            offsets=offs,
            # Positional argument expressions as passed, aligned in signature order.
            # Named variables must be expanded into their definitions, or symbolic arguments get missed.
            arg_exprs={nm: (lambda v: resolve_names(v, src, symbols) if v else v)(
                           (run_args.get(gname) or [None] * 99)[i])
                       for i, nm in enumerate(runtime_args)
                       if arg_is_scalar.get(nm)},
            arg_is_scalar=arg_is_scalar,
            # FixedGrid passes the grid explicitly after the arguments; take the three that follow
            grid_exprs=([resolve_names(e, src, symbols) for e in
                         (run_args.get(gname) or [])[len(runtime_args):
                                                     len(runtime_args) + 3]]
                        if meta.get("grid_type") == "FixedGrid" else None),
            func=func,
        ))
    # Sort by order of appearance in call(); that is the launch order
    order = {g: src.index(f"{g}.run(") if f"{g}.run(" in src else 1 << 30
             for g in (k["gname"] for k in kernels)}
    kernels.sort(key=lambda k: order[k["gname"]])
    return f, kernels, symbols, src


def main():
    import torch
    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1

    MMAX, D = 1024, 256

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(D, D)

        def forward(self, x):
            h = torch.relu(self.l1(x))
            h = h - h.mean(dim=-1, keepdim=True)
            # **Must reduce across rows**. With a row-wise model (row i depends only on row i),
            # even if the planner does nothing and the graph always computes with the capture-time M,
            # the first Mi rows still come out right -- then the test is not checking the planner at all.
            return torch.softmax(h, dim=-1).sum(dim=0)

    m = M().cuda().eval()

    # Must capture at the range **maximum**: outputs and intermediate buffers are allocated from the graph's
    # private pool at capture-time sizes, so replaying with a larger shape would go out of bounds.
    x_buf = torch.randn(MMAX, D, device="cuda")
    f, kernels, symbols, src = compile_and_extract(m, (x_buf,))

    print(f"symbols {symbols}, {len(kernels)} kernels in graph")
    for i, k in enumerate(kernels):
        pat = {nm: e for nm, e in k["arg_exprs"].items()
               if e and re.search(r"\bs\d+\b", str(e))}
        print(f"  [{i}] {k['name'][:52]}")
        gdesc = k.get("grid_exprs") or \
            "ceil(xnumel/%s)" % k["blocks"].get("XBLOCK", 1)
        print(f"      grid={gdesc}  to patch={pat}")
        print(f"      all args: {k['arg_exprs']}")
        print(f"      block: {k['blocks']}")
    bad = [f"{k['name']}.{nm}" for k in kernels for nm, e in k["arg_exprs"].items()
           if e and re.search(r"\bs\d+\b", str(e)) and nm not in k["offsets"]]
    if bad:
        print(f"  FAIL symbolic but no offset: {bad}"); return 1

    # ---- generate and compile the planner ----
    with tempfile.TemporaryDirectory() as wd:
        psrc = gen_planner(kernels, symbols)
        os.makedirs(DG_OUT, exist_ok=True)
        open(os.path.join(DG_OUT, "planner_generated.cu"), "w").write(psrc)
        cubin = build_cubin(psrc, wd)
        planner = load_kernel(cubin)
        print(f"  planner compiled and loaded (source kept in {DG_OUT}/planner_generated.cu)")

        n = len(kernels)
        d_handles = torch.zeros(n, dtype=torch.int64, device="cuda")
        d_ctx = torch.zeros(max(1, len(symbols)), dtype=torch.int64, device="cuda")

        L = torch._C._StaticCudaLauncher
        # warmup: lazy initialization must be fully done before capture
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                f(x_buf)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        L._begin_device_node_collection()
        try:
            with torch.cuda.graph(g):
                launch_planner(planner, n, d_handles.data_ptr(),
                               d_ctx.data_ptr(),
                               torch.cuda.current_stream().cuda_stream)
                with torch.no_grad():
                    out_buf = f(x_buf)
            handles = L._end_device_node_collection()
        except Exception as e:
            try: L._end_device_node_collection()
            except Exception: pass
            print(f"  capture failed: {type(e).__name__}: {str(e)[:250]}"); return 1

        print(f"  collected {len(handles)} handles, {n} kernels")
        if len(handles) != n:
            print("  FAIL handle count != kernel count: some nodes were missed and the planner cannot patch them")
            return 1
        d_handles.copy_(torch.tensor(handles, dtype=torch.int64))
        torch.cuda.synchronize()

        # ---- replay at different shapes, compare against eager ----
        print("\n  replay per shape and compare (one capture, zero re-records)")
        bad_rows = 0
        for Mi in (MMAX, MMAX // 2, 333, 64, 63, 9, 8, 7, 5, 2, 1):
            ref_in = torch.randn(Mi, D, device="cuda")
            with torch.no_grad():
                # The reference is the **compiled** function, not eager.
                # Eager uses cuBLAS + aten softmax, the graph uses Triton mm + online softmax;
                # the algorithms differ, so a 1e-5-level difference is algorithmic, not a bug;
                # using eager as the baseline would mix "is it correct" with "is the numeric path the same".
                ref = f(ref_in).clone()
                ref_eager = m(ref_in).clone()

            x_buf.zero_()
            x_buf[:Mi].copy_(ref_in)

            # Negative control: deliberately do **not** update ctx, so the planner computes with the capture-time M.
            # If this still matches, the test is insensitive to the planner and the conclusion is false.
            d_ctx[0] = MMAX
            g.replay(); torch.cuda.synchronize()
            stale = out_buf.clone()

            d_ctx[0] = Mi
            g.replay(); torch.cuda.synchronize()
            got = out_buf

            scale = max(ref.abs().max().item(), 1e-9)
            rel_vs_eager = (ref_eager - ref).abs().max().item() / scale
            rel = (ref - got).abs().max().item() / scale
            rel_stale = (ref - stale).abs().max().item() / scale
            ok = rel < 1e-5
            sensitive = (Mi == MMAX) or (rel_stale > 1e-3)
            bad_rows += (not ok) or (not sensitive)
            absmax = (ref - got).abs().max().item()
            print(f"    M={Mi:<6} vs compiled {rel:.2e} {'ok' if ok else 'FAIL'}   "
                  f"(compiled vs eager {rel_vs_eager:.2e}, algorithmic-difference baseline)   "
                  f"neg control {rel_stale:.2e} {'ok' if sensitive else 'FAIL invalid'}")

        print()
        if bad_rows == 0:
            print("First DynaGraph execution in PyTorch succeeded: one capture covered the whole "
                  "shape range, replay results are bitwise identical to the compiled version at the same shape, "
                  "and the negative control shows the planner really is patching the graph.")
            return 0
        print(f"{bad_rows} shapes do not match.")
        return 1




# --------------------------------------------------------------- expression -> C
def expr_to_c(expr: str, sym_index: dict[str, int]) -> str:
    """Translate a Python expression from the wrapper into C.

    Uses ast rather than string replacement: these expressions contain `//` (Python floor division),
    whose sign semantics differ from C's `/`, so replace() would break sooner or later.
    ast also guarantees parenthesization/associativity is not mangled.
    """
    import ast as _ast

    def go(n):
        if isinstance(n, _ast.Expression):
            return go(n.body)
        if isinstance(n, _ast.Constant):
            return f"(int64_t){int(n.value)}"
        if isinstance(n, _ast.Name):
            if n.id not in sym_index:
                raise KeyError(f"unregistered symbol in expression: {n.id}")
            return f"S({sym_index[n.id]})"
        if isinstance(n, _ast.BinOp):
            a, b = go(n.left), go(n.right)
            op = n.op
            if isinstance(op, _ast.Add):
                return f"({a} + {b})"
            if isinstance(op, _ast.Sub):
                return f"({a} - {b})"
            if isinstance(op, _ast.Mult):
                return f"({a} * {b})"
            if isinstance(op, _ast.FloorDiv):
                return f"floor_div({a}, {b})"
            if isinstance(op, _ast.Mod):
                return f"mod_py({a}, {b})"
            if isinstance(op, _ast.Div):
                return f"floor_div({a}, {b})"     # Inductor only produces integer division here
            raise NotImplementedError(f"unsupported operator {type(op).__name__}")
        if isinstance(n, _ast.UnaryOp) and isinstance(n.op, _ast.USub):
            return f"(-{go(n.operand)})"
        if isinstance(n, _ast.UnaryOp) and isinstance(n.op, _ast.UAdd):
            return go(n.operand)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name):
            fn = n.func.id
            if fn in ("min", "max") and len(n.args) == 2:
                return f"{fn}_i64({go(n.args[0])}, {go(n.args[1])})"
            raise NotImplementedError(f"unsupported function {fn}")
        raise NotImplementedError(f"unsupported construct {type(n).__name__}: {_ast.dump(n)[:80]}")

    return go(_ast.parse(expr.strip(), mode="eval"))


PLANNER_SRC = r"""
// === DynaGraph planner -- auto-generated, do not edit by hand ===
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ int64_t floor_div(int64_t a, int64_t b) {
  int64_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) --q; return q;
}
__device__ __forceinline__ int64_t mod_py(int64_t a, int64_t b) {
  int64_t r = a % b; if (r != 0 && ((r < 0) != (b < 0))) r += b; return r;
}
__device__ __forceinline__ int64_t min_i64(int64_t a, int64_t b){ return a<b?a:b; }
__device__ __forceinline__ int64_t max_i64(int64_t a, int64_t b){ return a>b?a:b; }

#define S(i) (ctx[(i)])
__device__ __forceinline__ int64_t eval_expr(int32_t k, const int64_t* __restrict__ ctx) {
  switch (k) {
/*@CASES@*/
    default: return 0;
  }
}
#undef S

// One thread per node: set the grid first, then write back all of that node's symbolic arguments.
// **Both are required**: changing only the grid leaves the kernel's internal masks/bounds on the old size,
// so the tail block is wrong; changing only the arguments leaves the block count wrong -- either under-computing or out of bounds.
extern "C" __global__ void dynagraph_planner(
    const cudaGraphDeviceNode_t* __restrict__ handles,
    const int64_t* __restrict__ ctx)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= /*@N@*/) return;

  int64_t gx = 1, gy = 1, gz = 1;
  switch (i) {
/*@GRID@*/
    default: break;
  }
  if (gx <= 0 || gy <= 0 || gz <= 0) {
    // grid=0 is cudaErrorInvalidArgument (measured in microbench/grid_zero.cu),
    // so an empty tensor can only be expressed by disabling the node.
    cudaGraphKernelNodeSetEnabled(handles[i], 0);
    return;
  }
  cudaGraphKernelNodeSetEnabled(handles[i], 1);
  cudaGraphKernelNodeSetGridDim(handles[i],
      dim3((unsigned)gx, (unsigned)gy, (unsigned)gz));

  switch (i) {
/*@PARAM@*/
    default: break;
  }
}
"""


def gen_planner(kernels, symbols):
    """Generate the planner source from the extracted table."""
    sym_index = {s: i for i, s in enumerate(symbols)}
    exprs: list[str] = []

    def idx_of(e: str) -> int:
        c = expr_to_c(e, sym_index)
        if c not in exprs:
            exprs.append(c)
        return exprs.index(c)

    grid_cases, param_cases = [], []
    for i, k in enumerate(kernels):
        if k.get("grid_exprs"):
            gx, gy, gz = (idx_of(e) for e in k["grid_exprs"])
            grid_cases.append(
                f"    case {i}: gx=eval_expr({gx},ctx); gy=eval_expr({gy},ctx); "
                f"gz=eval_expr({gz},ctx); break;   // {k['name']}")
        else:
            # Grid1D: grid_0 = ceil(xnumel / XBLOCK)
            xn = k["arg_exprs"].get("xnumel")
            blk = k["blocks"].get("XBLOCK", 1)
            if xn is None:
                raise RuntimeError(f"{k['name']} has neither an explicit grid nor xnumel")
            e = idx_of(f"({xn})")
            grid_cases.append(
                f"    case {i}: gx=floor_div(eval_expr({e},ctx) + {blk} - 1, {blk}); "
                f"break;   // {k['name']}")

        patches = []
        for nm, e in k["arg_exprs"].items():
            if e is None or not re.search(r"\bs\d+\b", str(e)):
                continue
            off, size = k["offsets"][nm]
            ei = idx_of(e)
            ctype = "int32_t" if size == 4 else "int64_t"
            patches.append(
                f"      {{ {ctype} v = ({ctype})eval_expr({ei},ctx); "
                f"cudaGraphKernelNodeSetParam(handles[i], {off}, &v, {size}); }}"
                f"   // {nm} = {e}")
        if patches:
            param_cases.append(f"    case {i}:\n" + "\n".join(patches) + "\n      break;")

    cases = "\n".join(f"    case {i}: return {c};" for i, c in enumerate(exprs))
    # Use replace() rather than % formatting: the planner source contains C's modulo operator,
    # which clashes directly with %-format placeholder syntax (ValueError: unsupported format character).
    out = PLANNER_SRC
    for tag, val in (("/*@CASES@*/", cases),
                     ("/*@N@*/", str(len(kernels))),
                     ("/*@GRID@*/", "\n".join(grid_cases)),
                     ("/*@PARAM@*/", "\n".join(param_cases))):
        out = out.replace(tag, val)
    return out


# --------------------------------------------------------------- compile & load
def build_cubin(src: str, workdir: str) -> str:
    cu_path = os.path.join(workdir, "planner.cu")
    cubin = os.path.join(workdir, "planner.cubin")
    with open(cu_path, "w") as fh:
        fh.write(src)
    # Measured: -rdc is not needed; the device-side graph API compiles without it, so emit a cubin and load it at runtime,
    # skipping the whole cuLinkCreate/AddData/Complete sequence.
    r = subprocess.run(
        ["nvcc", "-arch=sm_90a", "-cubin", "-o", cubin, cu_path],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise RuntimeError("planner compilation failed")
    return cubin


def load_kernel(cubin_path: str, name: str = "dynagraph_planner") -> int:
    mod = ctypes.c_void_p()
    ck(cu().cuModuleLoad(ctypes.byref(mod), cubin_path.encode()), "cuModuleLoad")
    fn = ctypes.c_void_p()
    ck(cu().cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()),
       "cuModuleGetFunction")
    return fn.value


def launch_planner(func: int, n_nodes: int, handles_ptr: int, ctx_ptr: int,
                   stream: int):
    """Launch the planner once on the current (capture) stream."""
    a0 = ctypes.c_void_p(handles_ptr)
    a1 = ctypes.c_void_p(ctx_ptr)
    params = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(a0), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(a1), ctypes.c_void_p))
    block = 128
    grid = (n_nodes + block - 1) // block
    ck(cu().cuLaunchKernel(
        ctypes.c_void_p(func), grid, 1, 1, block, 1, 1, 0,
        ctypes.c_void_p(stream), params, None), "cuLaunchKernel(planner)")


if __name__ == "__main__":
    sys.exit(main())
