"""Does DeepGEMM's C describe (third_party_patches/patch_dgd_c.py) match the Python describe byte for byte? What does one call cost?"""
import ctypes as ct
import time

import deep_gemm
import torch


class Op(ct.Structure):
    _fields_ = [("ptr", ct.c_uint64), ("dtype", ct.c_int32), ("ndim", ct.c_int32),
                ("sizes", ct.c_int64 * 8), ("strides", ct.c_int64 * 8)]


class Launch(ct.Structure):
    _fields_ = [("func", ct.c_uint64), ("grid", ct.c_uint32 * 3), ("block", ct.c_uint32 * 3),
                ("smem", ct.c_uint32), ("cluster", ct.c_uint32), ("pdl", ct.c_uint32), ("nargs", ct.c_uint32)]


lib = ct.CDLL(deep_gemm._C.__file__)
fn = lib.dg_describe_bf16_gemm_nt
fn.restype = ct.c_int
BF16 = 15  # c10::ScalarType::BFloat16


def op(t):
    o = Op(t.data_ptr(), BF16, t.dim())
    for i in range(t.dim()):
        o.sizes[i], o.strides[i] = t.size(i), t.stride(i)
    return o


ops = (Op * 3)()
out = (Launch * 16)()
args = ct.create_string_buffer(16384)
sizes = (ct.c_uint32 * 256)()
err = ct.create_string_buffer(512)


def c_describe(a, b, d, dims):
    ops[0], ops[1], ops[2] = op(a), op(b), op(d)
    n = fn(ops, 3, dims.encode(), out, 16, args, 16384, sizes, 256, err, 512)
    assert n >= 0, (n, err.value)
    res, at, ai = [], 0, 0
    for i in range(n):
        L = out[i]
        a_ = []
        for _ in range(L.nargs):
            a_.append(args.raw[at: at + sizes[ai]])
            at += sizes[ai]
            ai += 1
        res.append((L.func, tuple(L.grid), tuple(L.block), L.smem, L.cluster, bool(L.pdl), a_))
    return res


def py_describe(a, b, d, dims):
    deep_gemm._C.describe_begin()
    deep_gemm.bf16_gemm_nt(a, b, d, compiled_dims=dims)
    got = deep_gemm._C.describe_end()
    return [(f, tuple(g), tuple(bl), s, c, p, list(ar)) for f, g, bl, s, c, p, ar in got]


e = lambda *sh: torch.empty(*sh, device="cuda", dtype=torch.bfloat16)
K = N = 480
W = e(N, K)
bad = 0
cases = []
for M in (1, 7, 64, 200, 541, 1623, 3000, 4096, 7999, 8192):
    for n, k in ((480, 480), (1920, 480), (480, 1920)):
        a, b, d = e(M, k), e(n, k), e(M, n)
        deep_gemm.bf16_gemm_nt(a, b, d, compiled_dims="nk")  # JIT outside the timing
        cases.append((a, b, d))
        if c_describe(a, b, d, "nk") != py_describe(a, b, d, "nk"):
            bad += 1
            print("DIFF at", M, n, k)
torch.cuda.synchronize()
print(f"{len(cases)} cases, {bad} differ")
for name, f in (("C", c_describe), ("Python", py_describe)):
    ts = []
    for _ in range(20):
        for a, b, d in cases:
            t = time.perf_counter()
            f(a, b, d, "nk")
            ts.append(time.perf_counter() - t)
    ts.sort()
    print(f"{name} describe (incl. ctypes marshalling here): median {ts[len(ts)//2]*1e6:.1f} us")
