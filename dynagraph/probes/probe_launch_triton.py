#!/usr/bin/env python3
"""Launch described by triton_launch vs the one recorded from the real launch: byte-for-byte identical?
Covers: plain kernel, grid lambda, constexpr, int==1 specialized away, heuristics."""
import torch, triton, triton.language as tl
from torch.utils import _capture_launch as cl


@triton.jit
def axpy(x, y, out, n, alpha, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, alpha * tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)


@triton.heuristics({"EVEN": lambda a: a["n"] % a["BLOCK"] == 0})
@triton.jit
def scale_rows(x, out, n, stride, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    v = tl.load(x + r * stride + i, mask=i < n)
    tl.store(out + r * stride + i, v * 2, mask=i < n)


ok = bad = 0
for n in (1, 7, 16, 1000, 4096):
    x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda"); o = torch.empty_like(x)
    grid = lambda m: (triton.cdiv(n, m["BLOCK"]),)
    axpy[grid](x, y, o, n, 0.5, BLOCK=128)  # compile outside capture (prepare)
    d = [cl.triton_launch(axpy, grid, x, y, o, n, 0.5, BLOCK=128)]
    r = cl.record(lambda: axpy[grid](x, y, o, n, 0.5, BLOCK=128))
    try:
        cl.check(d, r); ok += 1; print(f"axpy n={n}: match, grid {r[0].grid}, {len(r[0].params)} params")
    except cl.Mismatch as e:
        bad += 1; print(f"axpy n={n}: MISMATCH {e}")
for rows, n in ((3, 64), (5, 100)):
    x = torch.randn(rows, 128, device="cuda"); o = torch.empty_like(x)
    scale_rows[(rows,)](x, o, n, x.stride(0), BLOCK=128)
    d = [cl.triton_launch(scale_rows, (rows,), x, o, n, x.stride(0), BLOCK=128)]
    r = cl.record(lambda: scale_rows[(rows,)](x, o, n, x.stride(0), BLOCK=128))
    try:
        cl.check(d, r); ok += 1; print(f"scale_rows rows={rows} n={n}: match")
    except cl.Mismatch as e:
        bad += 1; print(f"scale_rows: MISMATCH {e}")
print("all passed" if bad == 0 else f"{bad} items failed")
