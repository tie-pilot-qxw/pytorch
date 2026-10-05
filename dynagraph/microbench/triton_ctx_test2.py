import os, glob, torch, triton, triton.language as tl
dev = "cuda"
@triton.jit
def k_direct(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid * BLOCK + tl.arange(0, BLOCK); m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) * tl.load(y_ptr + offs, mask=m) + 1.0, mask=m)

# variant A: int64 from ctx -> cast to pointer, hint AFTER cast
@triton.jit
def k_ctxA(ctx, BLOCK: tl.constexpr):
    x_ptr = tl.load(ctx + 0).to(tl.pointer_type(tl.float32)); y_ptr = tl.load(ctx + 1).to(tl.pointer_type(tl.float32)); o_ptr = tl.load(ctx + 2).to(tl.pointer_type(tl.float32))
    x_ptr = tl.multiple_of(x_ptr, 16); y_ptr = tl.multiple_of(y_ptr, 16); o_ptr = tl.multiple_of(o_ptr, 16)
    n = tl.load(ctx + 3).to(tl.int32)
    nblk = tl.cdiv(n, BLOCK)
    for b in range(tl.program_id(0), nblk, tl.num_programs(0)):
        offs = b * BLOCK + tl.arange(0, BLOCK); m = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) * tl.load(y_ptr + offs, mask=m) + 1.0, mask=m)

# variant B: real pointer arg 'base' (fixed at capture) + int64 element offsets from ctx (base+off keeps divisibility)
@triton.jit
def k_ctxB(base, ctx, BLOCK: tl.constexpr):
    xo = tl.multiple_of(tl.load(ctx + 0), 16); yo = tl.multiple_of(tl.load(ctx + 1), 16); oo = tl.multiple_of(tl.load(ctx + 2), 16)
    x_ptr = base + xo; y_ptr = base + yo; o_ptr = base + oo
    n = tl.load(ctx + 3).to(tl.int32)
    nblk = tl.cdiv(n, BLOCK)
    for b in range(tl.program_id(0), nblk, tl.num_programs(0)):
        offs = b * BLOCK + tl.arange(0, BLOCK); m = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) * tl.load(y_ptr + offs, mask=m) + 1.0, mask=m)

BLOCK = 1024; NSM = torch.cuda.get_device_properties(0).multi_processor_count; PGRID = NSM * 8
ctx = torch.zeros(4, dtype=torch.int64, device=dev)
base = torch.zeros(16, device=dev)            # any 16B-aligned float32 tensor; offsets are relative to it
pin = torch.zeros(4, dtype=torch.int64).pin_memory()
def set_ctx(x, y, o, rel=False):
    b = base.data_ptr() if rel else 0
    pin.copy_(torch.tensor([(x.data_ptr()-b)//(4 if rel else 1), (y.data_ptr()-b)//(4 if rel else 1), (o.data_ptr()-b)//(4 if rel else 1), x.numel()]))
    ctx.copy_(pin, non_blocking=True)

s = torch.cuda.Stream()
def build(fn):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s): fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=s): fn()
    return g
x0 = torch.randn(1 << 20, device=dev); y0 = torch.randn_like(x0); o0 = torch.empty_like(x0)
set_ctx(x0, y0, o0); torch.cuda.synchronize()
gA = build(lambda: k_ctxA[(PGRID,)](ctx, BLOCK=BLOCK))
set_ctx(x0, y0, o0, rel=True); torch.cuda.synchronize()
gB = build(lambda: k_ctxB[(PGRID,)](base, ctx, BLOCK=BLOCK))
for n in [1, 777, 1 << 12, 3_000_001, 1 << 24]:
    x = torch.randn(n, device=dev); y = torch.randn_like(x); o = torch.empty_like(x); ref = x * y + 1.0
    set_ctx(x, y, o); gA.replay(); torch.cuda.synchronize(); okA = torch.equal(o, ref)
    o.zero_(); set_ctx(x, y, o, rel=True); gB.replay(); torch.cuda.synchronize(); okB = torch.equal(o, ref)
    o.zero_(); set_ctx(x, y, o); k_ctxA[(PGRID,)](ctx, BLOCK=BLOCK); torch.cuda.synchronize(); okE = torch.equal(o, ref)
    print(f"n={n:>9}: graphA {okA}  graphB {okB}  eagerA {okE}")

for f in glob.glob(os.environ["TRITON_CACHE_DIR"] + "/**/*.ptx", recursive=True):
    src = open(f).read()
    for tag in ["k_direct", "k_ctxA", "k_ctxB"]:
        if f".entry {tag}" in src or f"// .globl\t{tag}" in src:
            print(f"{tag:9s} PTX: v4 loads={src.count('ld.global.v4')} v4 stores={src.count('st.global.v4')} scalar f32 loads={src.count('ld.global.f32')+src.count('ld.global.b32')}")

def time_graph(fn, iters=200, reps=20):
    gg = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s): fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(gg, stream=s):
        for _ in range(reps): fn()
    for _ in range(3): gg.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(iters): gg.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / (iters * reps) * 1000
for n in [1 << 12, 1 << 16, 1 << 20, 1 << 24]:
    x = torch.randn(n, device=dev); y = torch.randn_like(x); o = torch.empty_like(x)
    set_ctx(x, y, o); torch.cuda.synchronize()
    td = time_graph(lambda: k_direct[(triton.cdiv(n, BLOCK),)](x, y, o, n, BLOCK=BLOCK))
    ta = time_graph(lambda: k_ctxA[(PGRID,)](ctx, BLOCK=BLOCK))
    set_ctx(x, y, o, rel=True); torch.cuda.synchronize()
    tb = time_graph(lambda: k_ctxB[(PGRID,)](base, ctx, BLOCK=BLOCK))
    print(f"n={n:>9}  direct {td:6.2f} us | ctxA(int->ptr, hint) {ta:6.2f} us | ctxB(base+off) {tb:6.2f} us")
