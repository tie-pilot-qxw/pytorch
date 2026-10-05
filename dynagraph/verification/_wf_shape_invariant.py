#!/usr/bin/env python3
"""Across shapes, where exactly does an extern site's capture differ?

If the param bytes captured at two shapes differ only in pointers, then this harvest was never needed --
changing a few pointers is enough, and the site can be "captured once, reused for every shape". That is exactly the
cheap form of tier 3 for FA3: no param builder needed, only a declaration that "my capture is valid across shapes".

Conversely, if non-pointer bytes change too (cuBLAS M/N/K, tile counts), there is no way around it.

After each harvest, read out and keep the full param block of every node of every site, then at the end compare
the differences between shapes byte by byte per site, and mark which offsets are pointers.
"""
import collections, ctypes, os, sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
_STUB = os.environ.get("DG_DEPS", "/workspace/_deps") + "/fa4site"
if os.path.isdir(_STUB) and _STUB not in sys.path:
    sys.path.insert(0, _STUB)

import torch
from cuda.bindings import driver as cu, runtime as cr
from torch.cuda._utils import _check_cuda_bindings as ck
import torch._inductor.config as ic

ic.triton.cudagraphs = True
ic.triton.dynagraph = True
ic.triton.dynagraph_extern_child = True

from torch._inductor import dynagraph as _dgm


def blobs_of(raw):
    """Full param bytes of every kernel node + the (offset, size) of every param."""
    n = int(ck(cr.cudaGraphGetNodes(raw))[1])
    nodes = ck(cr.cudaGraphGetNodes(raw, n))[0]
    kernel = cr.cudaGraphNodeType.cudaGraphNodeTypeKernel
    out = []
    for nd in nodes:
        if ck(cr.cudaGraphNodeGetType(nd)) != kernel:
            continue
        p = ck(cu.cuGraphKernelNodeGetParams(cu.CUgraphNode(int(nd))))
        info = []
        for j in range(256):
            try:
                off, size = ck(cu.cuFuncGetParamInfo(p.func, j))
            except Exception:
                break
            info.append((int(off), int(size)))
        if not info:
            continue
        total = max(o + s for o, s in info)
        buf = bytearray(total)
        kp, ex = int(p.kernelParams or 0), int(p.extra or 0)
        if kp:
            for j, (off, size) in enumerate(info):
                at = ctypes.c_void_p.from_address(kp + 8 * j).value
                if at:
                    buf[off:off + size] = bytes(
                        (ctypes.c_ubyte * size).from_address(at)
                    )
        elif ex:
            ents = (ctypes.c_void_p * 8).from_address(ex)
            vals = [ents[k] or 0 for k in range(8)]
            ptr = size = None
            k = 0
            while k < 8 and vals[k]:
                if vals[k] == 1:
                    ptr = vals[k + 1]; k += 2
                elif vals[k] == 2:
                    size = ctypes.c_size_t.from_address(vals[k + 1]).value; k += 2
                else:
                    k += 1
            if ptr and size:
                got = bytes((ctypes.c_ubyte * size).from_address(ptr))
                buf = bytearray(got[:total].ljust(total, b"\0"))
        out.append((ck(cu.cuFuncGetName(p.func)).decode()[:44], bytes(buf),
                    (p.gridDimX, p.gridDimY, p.gridDimZ), int(p.sharedMemBytes)))
    return out


seen = collections.defaultdict(dict)   # site -> shape -> [(name, blob, grid, smem)]
sigs = {}                               # (site, shape) -> signature
_cur = {"shape": None}
_oh = _dgm.DynaGraphRunner._harvest


def _align(p):
    a = 1
    while a < 256 and p % (a * 2) == 0:
        a *= 2
    return a


def signature(name, a, kw):
    """Op + geometry/dtype/alignment of every argument + literal values of non-tensor arguments."""
    def one(v):
        if isinstance(v, torch.Tensor):
            return ("T", tuple(v.shape), tuple(v.stride()), str(v.dtype), _align(v.data_ptr()))
        if isinstance(v, (list, tuple)):
            return ("S", tuple(one(x) for x in v))
        return ("L", repr(v))
    return (name, tuple(one(x) for x in a), tuple(sorted((k, one(v)) for k, v in kw.items())))


_ori = _dgm.DynaGraphRunner._run_intercepted


def ri(self, args, views, on_extern):
    def wrapped(i, fn, a, kw):
        if _cur["shape"] is not None:
            sigs[(i, _cur["shape"])] = signature(self.extern_sites[i], a, kw)
        return on_extern(i, fn, a, kw)
    return _ori(self, args, views, wrapped)


_dgm.DynaGraphRunner._run_intercepted = ri


def h(self, env, key, hkey, inputs, *a, **kw):
    _cur["shape"] = tuple(sorted(env.items()))
    try:
        r = _oh(self, env, key, hkey, inputs, *a, **kw)
    finally:
        _cur["shape"] = None
    if r and hkey in self.child_graphs:
        shape = tuple(sorted(env.items()))
        for i, raw in enumerate(self.child_graphs[hkey]):
            if raw is None:
                continue
            name = self.extern_sites[i]
            if shape not in seen[(name, i)]:
                try:
                    seen[(name, i)][shape] = blobs_of(raw)
                except Exception as e:
                    seen[(name, i)][shape] = f"err {e}"
    return r


_dgm.DynaGraphRunner._harvest = h

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationMode

llm = LLM(
    model=os.environ.get("MODEL", "Qwen/Qwen3-0.6B"),
    max_model_len=1024,
    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.3")),
    enforce_eager=False, max_num_seqs=8, disable_log_stats=True,
    compilation_config={
        "mode": int(CompilationMode.STOCK_TORCH_COMPILE),
        "cudagraph_mode": "NONE",
        "inductor_compile_config": {"triton.cudagraphs": True},
    },
)
sp = SamplingParams(temperature=0.0, max_tokens=8)
for b in [int(v) for v in os.environ.get("BS", "1,2,3,4,5").split(",")]:
    llm.generate([f"Count from {i} to ten:" for i in range(b)], sp, use_tqdm=False)

import bisect as _bisect

_segs = sorted(
    (sg["address"], sg["address"] + sg["total_size"])
    for sg in torch.cuda.memory_snapshot()
)
_starts = [a for a, _ in _segs]
print(f"\n  {len(_segs)} live allocation segments, address range "
      f"{hex(_segs[0][0]) if _segs else '-'} .. {hex(_segs[-1][1]) if _segs else '-'}")


def looks_ptr(v):
    # A value counts as a pointer only if it falls inside a live allocation segment (same test as the binding table).
    # A threshold test treats low-address device pointers as scalars, so each layer's own pointers look like "params that differ between layers".
    j = _bisect.bisect_right(_starts, v) - 1
    return j >= 0 and _segs[j][0] <= v < _segs[j][1]


print("\n  per-site differences across shapes (only sites captured at two or more shapes)")
agg = collections.Counter()
detail = collections.defaultdict(collections.Counter)
for (name, i), by_shape in seen.items():
    shapes = [s for s, v in by_shape.items() if isinstance(v, list)]
    if len(shapes) < 2:
        continue
    a0 = by_shape[shapes[0]]
    for s in shapes[1:]:
        b0 = by_shape[s]
        if len(a0) != len(b0) or any(x[0] != y[0] for x, y in zip(a0, b0)):
            agg[(name, "kernel or node count changed")] += 1
            continue
        gridchg = any(x[2] != y[2] for x, y in zip(a0, b0))
        ptr = nonptr = 0
        for x, y in zip(a0, b0):
            for off in range(0, min(len(x[1]), len(y[1])) - 7, 8):
                u = int.from_bytes(x[1][off:off + 8], "little")
                v = int.from_bytes(y[1][off:off + 8], "little")
                if u == v:
                    continue
                if looks_ptr(u) and looks_ptr(v):
                    ptr += 1
                    detail[name]["pointer"] += 1
                else:
                    nonptr += 1
                    detail[name][f"non-pointer@{off}"] += 1
        if nonptr == 0 and not gridchg:
            agg[(name, "only pointers change" if ptr else "identical")] += 1
        elif nonptr == 0:
            agg[(name, "pointers + grid change")] += 1
        else:
            agg[(name, f"non-pointer bytes change")] += 1
for (name, what), c in agg.most_common():
    short = name.replace("ops:vllm.", "").replace(".default", "")
    print(f"    {short:<34} {what:<24} x{c}")
print("\n  what changes (per site type, the 8 most common)")
for name, cnt in detail.items():
    short = name.replace("ops:vllm.", "").replace(".default", "")
    top = ", ".join(f"{k} x{v}" for k, v in cnt.most_common(8))
    print(f"    {short:<34} {top[:120]}")


# ---- same shape, compared across layers: how many distinct "pointers removed" contents does each op really have ----
# If an op has only a few across its 28 layers, a new shape needs only that many captures; the other sites copy the template and swap pointers.
def fingerprint(nodes):
    out = []
    for kname, blob, grid, smem in nodes:
        b = bytearray(blob)
        for off in range(0, len(b) - 7, 8):
            if looks_ptr(int.from_bytes(b[off:off + 8], "little")):
                b[off:off + 8] = b"\0" * 8
        out.append((kname, grid, smem, bytes(b)))
    return tuple(out)


print("\n  across layers at the same shape: site count vs distinct contents after removing pointers")
per = collections.defaultdict(lambda: [0, set()])
shapes_all = set()
for (name, i), by_shape in seen.items():
    for shape, nodes in by_shape.items():
        if not isinstance(nodes, list):
            continue
        shapes_all.add(shape)
        per[(name, shape)][0] += 1
        per[(name, shape)][1].add(fingerprint(nodes))
agg2 = collections.defaultdict(list)
for (name, shape), (n, fps) in per.items():
    agg2[name].append((n, len(fps)))
for name, rows in agg2.items():
    short = name.replace("ops:vllm.", "").replace(".default", "")
    ns = [r[0] for r in rows]
    ks = [r[1] for r in rows]
    print(f"    {short:<34} {len(rows)} shapes; sites per shape {min(ns)}-{max(ns)}, "
          f"distinct contents {min(ks)}-{max(ks)}")
tot_sites = sum(n for n, _ in per.values())
tot_fp = sum(len(f) for _, f in per.values())
print(f"    total: {tot_sites} site captures -> only {tot_fp} needed ({tot_sites / max(tot_fp, 1):.1f}x)")


# ---- look directly: at the same shape, which words differ between two layers of kv_cache_update ----
print("\n  kv_cache_update: layer 0 vs layer 1 at the same shape, differing words")
for shape in sorted(shapes_all)[:1]:
    kv = [(i, by[shape]) for (name, i), by in seen.items()
          if "kv_cache_update" in name and isinstance(by.get(shape), list)]
    kv.sort()
    if len(kv) >= 2:
        (i0, n0), (i1, n1) = kv[0], kv[1]
        for (ka, ba, ga, sa), (kb, bb, gb, sb) in zip(n0, n1):
            print(f"    kernel {ka}  grid {ga} vs {gb}")
            for off in range(0, min(len(ba), len(bb)) - 7, 8):
                u = int.from_bytes(ba[off:off + 8], "little")
                v = int.from_bytes(bb[off:off + 8], "little")
                if u != v:
                    tag = "pointer" if (looks_ptr(u) and looks_ptr(v)) else "non-pointer"
                    print(f"      @{off:<4} {hex(u):>20} vs {hex(v):>20}  {tag}")


print("\n  grouping by signature vs grouping by capture fingerprint (within one shape)")
res = collections.defaultdict(lambda: [0, 0, 0, 0])   # op -> [sites, signature groups, fingerprint kinds, unreliable groups]
for shape in shapes_all:
    groups = collections.defaultdict(set)   # (op, sig) -> {fingerprint}
    fps = collections.defaultdict(set)      # op -> {fingerprint}
    nsite = collections.Counter()
    for (name, i), by_shape in seen.items():
        nodes = by_shape.get(shape)
        if not isinstance(nodes, list) or (i, shape) not in sigs:
            continue
        fp = fingerprint(nodes)
        groups[(name, sigs[(i, shape)])].add(fp)
        fps[name].add(fp)
        nsite[name] += 1
    for name in nsite:
        gs = [g for (n, _), g in groups.items() if n == name]
        r = res[name]
        r[0] += nsite[name]; r[1] += len(gs); r[2] += len(fps[name])
        r[3] += sum(1 for g in gs if len(g) > 1)
for name, (n, ng, nf, bad) in res.items():
    short = name.replace("ops:vllm.", "").replace(".default", "")
    print(f"    {short:<32} sites {n:>5}  sig groups {ng:>4}  fingerprints {nf:>4}  "
          f"sig groups with >1 fingerprint {bad}")
