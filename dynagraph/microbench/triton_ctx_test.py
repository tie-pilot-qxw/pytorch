import os, glob, torch, triton, triton.language as tl
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_ctx_cache")
dev = "cuda"

@triton.jit
def k_direct(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m); y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x * y + 1.0, mask=m)

@triton.jit
def k_ctx(ctx, BLOCK: tl.constexpr, HINT: tl.constexpr):
    xi = tl.load(ctx + 0); yi = tl.load(ctx + 1); oi = tl.load(ctx + 2); n = tl.load(ctx + 3)
    if HINT:
        xi = tl.multiple_of(xi, 16); yi = tl.multiple_of(yi, 16); oi = tl.multiple_of(oi, 16)
    x_ptr = xi.to(tl.pointer_type(tl.float32)); y_ptr = yi.to(tl.pointer_type(tl.float32)); o_ptr = oi.to(tl.pointer_type(tl.float32))
    nblk = tl.cdiv(n, BLOCK)
    for b in range(tl.program_id(0), nblk, tl.num_programs(0)):   # persistent virtual grid
        offs = b * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m); y = tl.load(y_ptr + offs, mask=m)
        tl.store(o_ptr + offs, x * y + 1.0, mask=m)

BLOCK = 1024
NSM = torch.cuda.get_device_properties(0).multi_processor_count
PGRID = NSM * 8
ctx = torch.zeros(4, dtype=torch.int64, device=dev)          # the one stable address the graph sees

def set_ctx(x, y, o):
    ctx.copy_(torch.tensor([x.data_ptr(), y.data_ptr(), o.data_ptr(), x.numel()], dtype=torch.int64), non_blocking=True)

# --- correctness + graph replay across different sizes & addresses
g = torch.cuda.CUDAGraph()
x0 = torch.randn(1 << 20, device=dev); y0 = torch.randn_like(x0); o0 = torch.empty_like(x0)
set_ctx(x0, y0, o0); torch.cuda.synchronize()
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    k_ctx[(PGRID,)](ctx, BLOCK=BLOCK, HINT=True)      # warm/compile
torch.cuda.synchronize()
with torch.cuda.graph(g, stream=s):
    k_ctx[(PGRID,)](ctx, BLOCK=BLOCK, HINT=True)
ok = True
for n in [1, 777, 1 << 12, 3_000_001, 1 << 24]:
    x = torch.randn(n, device=dev); y = torch.randn_like(x); o = torch.empty_like(x)
    set_ctx(x, y, o); g.replay(); torch.cuda.synchronize()
    ok &= torch.allclose(o, x * y + 1.0)
print("graph replay across 5 sizes/addresses correct:", ok)

# --- vectorization check: does the loaded-pointer path still get ld.global.v4 ?
def ptx_has_v4(tag):
    hits = []
    for f in glob.glob(os.environ["TRITON_CACHE_DIR"] + "/**/*.ptx", recursive=True):
        src = open(f).read()
        if tag in src: hits.append(("ld.global.v4" in src) or ("ld.global.L1::evict_last.v4" in src) or ("ld.global.v2" in src))
    return hits
k_ctx[(PGRID,)](ctx, BLOCK=BLOCK, HINT=False); torch.cuda.synchronize()
x = torch.randn(1 << 20, device=dev); y = torch.randn_like(x); o = torch.empty_like(x)
k_direct[(triton.cdiv(x.numel(), BLOCK),)](x, y, o, x.numel(), BLOCK=BLOCK); torch.cuda.synchronize()
print("v4 vector loads  direct:", ptx_has_v4("k_direct"), " ctx:", ptx_has_v4("k_ctx"))
for f in glob.glob(os.environ["TRITON_CACHE_DIR"] + "/**/*.ptx", recursive=True):
    src = open(f).read()
    if "k_ctx" in src:
        print("  k_ctx ptx", os.path.basename(f), "v4:", "ld.global.v4" in src, "v2:", "ld.global.v2" in src, "scalar f32 ld:", "ld.global.f32" in src or "ld.global.b32" in src)

# --- overhead: GPU time per kernel, exact-grid direct (graphed) vs persistent ctx (graphed)
def time_graph(fn, iters=200):
    gg = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(gg, stream=s):
        for _ in range(20): fn()
    torch.cuda.synchronize()
    for _ in range(3): gg.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): gg.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (iters * 20) * 1000  # us per kernel
for n in [1 << 12, 1 << 16, 1 << 20, 1 << 24]:
    x = torch.randn(n, device=dev); y = torch.randn_like(x); o = torch.empty_like(x)
    set_ctx(x, y, o); torch.cuda.synchronize()
    td = time_graph(lambda: k_direct[(triton.cdiv(n, BLOCK),)](x, y, o, n, BLOCK=BLOCK))
    tc = time_graph(lambda: k_ctx[(PGRID,)](ctx, BLOCK=BLOCK, HINT=True))
    tn = time_graph(lambda: k_ctx[(PGRID,)](ctx, BLOCK=BLOCK, HINT=False))
    print(f"n={n:>9}  direct exact-grid {td:7.2f} us | ctx persistent(hint) {tc:7.2f} us | ctx persistent(nohint) {tn:7.2f} us")
