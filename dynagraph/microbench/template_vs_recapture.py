"""'Why not just replay the graph you just captured' -- because the next step's shape has changed again.
A: capture a new graph for every new shape (= what cudagraph trees does)
B: capture a template only once; afterwards ctx supplies shape/pointers each step, never re-record
Same kernels, same input sequence; compare total time, number of captures, and memory."""
import time, torch, triton, triton.language as tl
DEV="cuda"; BLOCK=1024; DEPTH=20                     # 20 chained kernels, simulating a small model
NSM = torch.cuda.get_device_properties(0).multi_processor_count
PGRID = NSM * 4

@triton.jit                                           # static version: n and pointers are both launch args
def k_static(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK); m = off < n
    tl.store(o_ptr+off, tl.load(x_ptr+off, mask=m)*1.0001 + 0.001, mask=m)

@triton.jit                                           # template version: n and pointers read from ctx, persistent grid
def k_ctx(ctx, slot, BLOCK: tl.constexpr):
    xp = tl.load(ctx+slot*2+0).to(tl.pointer_type(tl.float32))
    op = tl.load(ctx+slot*2+1).to(tl.pointer_type(tl.float32))
    n  = tl.load(ctx+64).to(tl.int32)
    for b in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        off = b*BLOCK + tl.arange(0, BLOCK); m = off < n
        tl.store(op+off, tl.load(xp+off, mask=m)*1.0001 + 0.001, mask=m)

NMAX = 1 << 20
bufs = [torch.zeros(NMAX, device=DEV) for _ in range(DEPTH+1)]
ctx  = torch.zeros(128, dtype=torch.int64, device=DEV)
pin  = torch.zeros(128, dtype=torch.int64).pin_memory()
for i in range(DEPTH):
    pin[i*2+0] = bufs[i].data_ptr(); pin[i*2+1] = bufs[i+1].data_ptr()
ctx.copy_(pin)
s = torch.cuda.Stream()

def set_n(n):
    pin[64] = n; ctx[64:65].copy_(pin[64:65], non_blocking=True)

# ---------- B: template captured once ----------
with torch.cuda.stream(s):
    for i in range(DEPTH): k_ctx[(PGRID,)](ctx, i, BLOCK=BLOCK)
torch.cuda.synchronize()
g_tmpl = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_tmpl, stream=s):
    for i in range(DEPTH): k_ctx[(PGRID,)](ctx, i, BLOCK=BLOCK)
torch.cuda.synchronize()

# ---------- A: one graph per shape ----------
cache = {}
def graph_for(n):
    if n in cache: return cache[n]
    with torch.cuda.stream(s):
        for i in range(DEPTH): k_static[(triton.cdiv(n,BLOCK),)](bufs[i], bufs[i+1], n, BLOCK=BLOCK)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(DEPTH): k_static[(triton.cdiv(n,BLOCK),)](bufs[i], bufs[i+1], n, BLOCK=BLOCK)
    cache[n] = g
    return g

import random
random.seed(0)
for dist_name, seq in [("a different shape every step (variable-length workload)", [random.randint(1000, NMAX) for _ in range(200)]),
                       ("only 8 shapes (bucketable)", [random.choice([2**k for k in range(13,21)]) for _ in range(200)])]:
    print(f"\n### {dist_name}, 200 steps")
    # A
    cache.clear(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for n in seq:
        graph_for(n).replay()
    torch.cuda.synchronize(); tA = time.perf_counter()-t0
    nA, memA = len(cache), torch.cuda.memory_reserved()
    # B
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for n in seq:
        set_n(n); g_tmpl.replay()
    torch.cuda.synchronize(); tB = time.perf_counter()-t0
    print(f"  A one graph per shape  : {tA:7.3f} s   captured {nA:3d} graphs")
    print(f"  B one template + ctx   : {tB:7.3f} s   captured   1 graph    ({tA/tB:.1f}x)")

# correctness
print("\n### Correctness cross-check")
ok = True
for n in [1, 777, 65537, NMAX]:
    bufs[0][:n] = 1.0
    set_n(n); g_tmpl.replay(); torch.cuda.synchronize()
    got = bufs[DEPTH][:n].clone()
    bufs[0][:n] = 1.0
    graph_for(n).replay(); torch.cuda.synchronize()
    ok &= torch.allclose(got, bufs[DEPTH][:n], atol=1e-6)
print(f"  template vs per-shape re-record, results match on 4 sizes: {ok}")
