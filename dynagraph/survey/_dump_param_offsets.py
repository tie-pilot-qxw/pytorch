#!/usr/bin/env python3
"""Ask the CUDA driver for the REAL byte offsets of each kernel param.

cuFuncGetParamInfo(CUfunction, paramIndex, &offset, &size) -- CUDA >= 12.4.
This is the ground truth that cudaGraphKernelNodeSetParam(handle, offset, ...) needs.
"""
from __future__ import annotations
import ctypes, os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch._inductor.config as ic
ic.force_disable_caches = True
from torch._inductor.runtime import triton_heuristics as th

libcuda = ctypes.CDLL("libcuda.so.1")

def param_info(func_handle, n_max=32):
    out = []
    off = ctypes.c_size_t(); sz = ctypes.c_size_t()
    for i in range(n_max):
        rc = libcuda.cuFuncGetParamInfo(ctypes.c_void_p(func_handle),
                                        ctypes.c_size_t(i),
                                        ctypes.byref(off), ctypes.byref(sz))
        if rc != 0:
            break
        out.append((i, off.value, sz.value))
    return out

RECORDS = []
orig = th.StaticTritonCompileResult.make_launcher
def patched(self):
    L = orig(self)
    k = self.kernel
    RECORDS.append((k.name, k.arg_tys, k.shared, k.num_warps,
                    {str(a): str(b) for a, b in
                     (self.compile_meta.get("signature") or {}).items()},
                    k.function, k.global_scratch_size, k.profile_scratch_size))
    return L
th.StaticTritonCompileResult.make_launcher = patched

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(256, 512)
        self.l2 = torch.nn.Linear(512, 256)
    def forward(self, x):
        h = torch.relu(self.l1(x))
        return torch.relu(self.l2(h)) + x, h.sum(-1), torch.softmax(h, -1)

m = M().cuda()
f = torch.compile(m, dynamic=True)
f(torch.randn(37, 256, device="cuda"))
torch.cuda.synchronize()

seen = set()
for name, tys, shared, nw, sig, fnhandle, gss, pss in RECORDS:
    if (name, tys) in seen:
        continue
    seen.add((name, tys))
    print("=" * 76)
    print(f"{name}")
    print(f"  arg_tys={tys!r}  shared={shared}  num_warps={nw}")
    print(f"  global_scratch_size={gss}  profile_scratch_size={pss}")
    print(f"  signature={sig}")
    pi = param_info(fnhandle)
    names = [k for k in sig if sig[k] != "constexpr"]
    print(f"  driver cuFuncGetParamInfo -> {len(pi)} params:")
    for (i, o, s) in pi:
        nm = names[i] if i < len(names) else f"<extra#{i-len(names)}>"
        ty = sig.get(nm, "?")
        print(f"    idx={i:2d}  offset={o:4d}  size={s}   {nm} : {ty}")
