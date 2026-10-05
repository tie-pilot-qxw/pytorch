"""Structural probe: what lands in extern_kernels.* vs torch.ops.* fallbacks."""
import os, re, sys, json
import torch
import torch._inductor.config as icfg

print("=== torch ===", torch.__version__, torch.__file__)
print("max_autotune          =", icfg.max_autotune)
print("max_autotune_gemm     =", icfg.max_autotune_gemm)
print("max_autotune_conv     =", getattr(icfg, "max_autotune_conv", "N/A"))
print("max_autotune_gemm_backends =", icfg.max_autotune_gemm_backends)
print("decompose_sort_ops    =", icfg.triton.decompose_sort_ops)
print("fallback_random       =", icfg.fallback_random)

# force lowering import so extern_kernels is populated
import torch._inductor.lowering  # noqa
from torch._inductor.select_algorithm import extern_kernels, ExternKernelChoice

print("\n=== extern_kernels namespace members (name -> underlying callable) ===")
members = sorted(k for k in vars(extern_kernels) if not k.startswith("_"))
for name in members:
    fn = getattr(extern_kernels, name)
    mod = getattr(fn, "__module__", type(fn).__module__)
    print(f"  extern_kernels.{name:28s} -> {mod}.{getattr(fn,'__name__',repr(fn))}")
print("count =", len(members))

print("\n=== ExternKernelChoice registry (name, cpp_kernel, op_overload) ===")
for name, ch in sorted(ExternKernelChoice._registry.items()):
    print(f"  {name:28s} cpp={ch.cpp_kernel_name!r:34s} op={ch.op_overload} has_out={ch.has_out_variant}")

print("\n=== size of inductor fallback set (ops that go to dispatcher) ===")
from torch._inductor.lowering import fallbacks, lowerings
print("len(fallbacks) =", len(fallbacks))
interesting = ["sort", "topk", "cumsum", "scatter", "index_put", "nonzero",
               "convolution", "scaled_dot_product", "flash_attention",
               "efficient_attention", "mm", "bmm", "unique", "randperm",
               "repeat_interleave", "embedding_bag", "multinomial"]
names = sorted(str(o) for o in fallbacks)
for pat in interesting:
    hits = [n for n in names if pat in n]
    print(f"  {pat:24s}: {hits}")
