#!/usr/bin/env python3
"""
Under dynamic shapes, can mm/addmm be moved from cuBLAS to a Triton template?

Why ask: cuBLAS's m/n/k and algorithm choice are fixed on the host at capture time,
so changing the grid on the device does nothing for it -- this is the one part of DynaGraph with no known path.
But `extern_kernels` is only the **default choice**, not the only one:

    utils.py:3412 use_aten_gemm_kernels() -> not (max_autotune or max_autotune_gemm)
                                             or _use_autotune_backend("ATEN")
    utils.py:2305 use_triton_template()   -> (max_autotune or max_autotune_gemm) and ...

So restricting the backend to TRITON and turning on max_autotune_gemm should, in theory, knock ATEN out.
This script checks whether that really holds **under dynamic shapes**.

It only checks whether extern_kernels still shows up in the generated code, no performance measurement -- runs fine on a shared card.
"""
from __future__ import annotations

import os
import re
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")


def probe(label, setup, dynamic=True, dtype=None):
    import torch
    import torch._inductor.config as ic
    from torch._inductor import codecache

    torch._dynamo.reset()
    ic.force_disable_caches = True
    setup(ic)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(512, 512, bias=True)

        def forward(self, x):
            return torch.relu(self.l(x))

    dt = dtype or torch.float32
    m = M().cuda().to(dt).eval()
    x = torch.randn(64, 512, device="cuda", dtype=dt)

    srcs = []
    orig = codecache.PyCodeCache.load_by_key_path

    def spy(key, path, *a, **kw):
        try:
            s = open(path).read()
            if "def call(" in s:
                srcs.append(s)
        except Exception:
            pass
        return orig(key, path, *a, **kw)

    codecache.PyCodeCache.load_by_key_path = staticmethod(spy)
    try:
        with torch.no_grad():
            torch.compile(m, dynamic=dynamic)(x)
        err = None
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        codecache.PyCodeCache.load_by_key_path = staticmethod(orig)

    if err:
        print(f"  {label:<38} compile failed  {err}")
        return
    blob = "\n".join(srcs)
    extern = re.findall(r"extern_kernels\.(\w+)", blob)
    tmpl = re.findall(r"(triton_tem_fused\w*|cutlass_\w+|cuda_fused\w*|nvgemm\w*)", blob)
    syms = sorted(set(re.findall(r"\b(s\d+)\b", blob)))
    print(f"  {label:<38} extern_kernels={sorted(set(extern)) or 'none'}  "
          f"triton_templates={sorted(set(tmpl)) or 'none'}  symint={syms}")


def main():
    import torch
    if not torch.cuda.is_available():
        print("no CUDA device available"); return 1
    print(f"torch {torch.__version__}\n")
    print("Dynamic shapes (dynamic=True):")
    probe("default", lambda ic: None)
    probe("max_autotune_gemm=True",
          lambda ic: setattr(ic, "max_autotune_gemm", True))
    probe("+ backends restricted to TRITON", lambda ic: (
        setattr(ic, "max_autotune_gemm", True),
        setattr(ic, "max_autotune_gemm_backends", "TRITON")))
    print("\nbf16 (CUTLASS/NVGEMM do not support fp32, only fp16/bf16/int32):")
    for be in ("TRITON", "CUTLASS", "NVGEMM", "TRITON,CUTLASS,NVGEMM"):
        probe(f"backends={be}", lambda ic, be=be: (
            setattr(ic, "max_autotune_gemm", True),
            setattr(ic, "max_autotune_gemm_backends", be)),
            dtype=torch.bfloat16)
    print("\nStatic shapes as a control (dynamic=False):")
    probe("+ backends restricted to TRITON", lambda ic: (
        setattr(ic, "max_autotune_gemm", True),
        setattr(ic, "max_autotune_gemm_backends", "TRITON")), dynamic=False)
    print("""
How to read it
--------------
As soon as the extern_kernels column reads "none" and the triton_templates column has entries,
GEMM can bypass cuBLAS completely and land on a Triton kernel that Inductor generates itself --
then it is like any other Triton kernel: its grid is a closed-form function of the symints, which the planner can change.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
