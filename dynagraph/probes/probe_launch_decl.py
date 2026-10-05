#!/usr/bin/env python3
"""Declared vs actual: for reshape_and_cache_flash over a series of num_tokens,
is the declared launch byte-for-byte identical to the recorded one? Then deliberately get one argument wrong and see whether it is refused."""
import os
import sys

import torch
from torch.utils import _capture_launch as cl
import vllm._custom_ops  # noqa: F401  (loads _C_cache_ops)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "serving"))  # vllm_launches.py lives in ../serving
import vllm_launches  # noqa: F401

op = torch.ops._C_cache_ops.reshape_and_cache_flash
H, D, B, NB = 8, 128, 16, 64
kv = torch.zeros(NB, 2, B, H, D, device="cuda", dtype=torch.bfloat16)
kc, vc = kv[:, 0], kv[:, 1]
ks = torch.ones(1, device="cuda"); vs = torch.ones(1, device="cuda")
ok = 0
for n in (1, 3, 16, 37, 200):
    k = torch.randn(n, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(n, H, D, device="cuda", dtype=torch.bfloat16)
    sm = torch.randperm(NB * B, device="cuda")[:n].long()
    args = (k, v, kc, vc, sm, "auto", ks, vs)
    dec = cl.launches_of("_C_cache_ops::reshape_and_cache_flash", *args)
    rec = cl.record(op, *args)
    funcs = cl.check(dec, rec)
    ok += 1
    print(f"n={n}: {len(rec)} launch, grid {rec[0].grid}, {len(rec[0].params)} params, match; func {funcs[0]:#x}")
bad = cl.Launch(*dec[0][:4], args=dec[0].args[:10] + (dec[0].args[10] + 1,) + dec[0].args[11:])
try:
    cl.check([bad], rec)
    print("FAIL: wrong declaration accepted")
except cl.Mismatch as e:
    print("wrong declaration refused:", e)
    ok += 1
print("all passed" if ok == 6 else f"{6 - ok} failed")
