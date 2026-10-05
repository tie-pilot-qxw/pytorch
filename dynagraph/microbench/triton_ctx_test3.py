import os, glob, torch, triton, triton.language as tl
dev = "cuda"; BLOCK = 1024
@triton.jit
def body(x_ptr, y_ptr, o_ptr, b, n, BLOCK: tl.constexpr):
    offs = b * BLOCK + tl.arange(0, BLOCK); m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) * tl.load(y_ptr + offs, mask=m) + 1.0, mask=m)
@triton.jit
def k_direct_exact(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    body(x_ptr, y_ptr, o_ptr, tl.program_id(0), n, BLOCK)
@triton.jit
def k_direct_loop(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    for b in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        body(x_ptr, y_ptr, o_ptr, b, n, BLOCK)
@triton.jit
def k_ctx_exact(ctx, BLOCK: tl.constexpr):
    x_ptr = tl.multiple_of(tl.load(ctx + 0).to(tl.pointer_type(tl.float32)), 16); y_ptr = tl.multiple_of(tl.load(ctx + 1).to(tl.pointer_type(tl.float32)), 16); o_ptr = tl.multiple_of(tl.load(ctx + 2).to(tl.pointer_type(tl.float32)), 16)
    n = tl.load(ctx + 3).to(tl.int32)
    body(x_ptr, y_ptr, o_ptr, tl.program_id(0), n, BLOCK)
@triton.jit
def k_ctx_loop(ctx, BLOCK: tl.constexpr):
    x_ptr = tl.multiple_of(tl.load(ctx + 0).to(tl.pointer_type(tl.float32)), 16); y_ptr = tl.multiple_of(tl.load(ctx + 1).to(tl.pointer_type(tl.float32)), 16); o_ptr = tl.multiple_of(tl.load(ctx + 2).to(tl.pointer_type(tl.float32)), 16)
    n = tl.load(ctx + 3).to(tl.int32)
    for b in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        body(x_ptr, y_ptr, o_ptr, b, n, BLOCK)
@triton.jit
def k_ctx_loop_nohint(ctx, BLOCK: tl.constexpr):
    x_ptr = tl.load(ctx + 0).to(tl.pointer_type(tl.float32)); y_ptr = tl.load(ctx + 1).to(tl.pointer_type(tl.float32)); o_ptr = tl.load(ctx + 2).to(tl.pointer_type(tl.float32))
    n = tl.load(ctx + 3).to(tl.int32)
    for b in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        body(x_ptr, y_ptr, o_ptr, b, n, BLOCK)

NSM = torch.cuda.get_device_properties(0).multi_processor_count; PGRID = NSM * 8
ctx = torch.zeros(4, dtype=torch.int64, device=dev); pin = torch.zeros(4, dtype=torch.int64).pin_memory()
def set_ctx(x, y, o):
    pin.copy_(torch.tensor([x.data_ptr(), y.data_ptr(), o.data_ptr(), x.numel()])); ctx.copy_(pin, non_blocking=True)
n = 777
x = torch.randn(n, device=dev); y = torch.randn_like(x); o = torch.empty_like(x); ref = torch.addcmul(torch.ones_like(x), x, y)  # fma-like
set_ctx(x, y, o); k_ctx_loop[(PGRID,)](ctx, BLOCK=BLOCK); torch.cuda.synchronize()
print(f"n=777 ctx_loop: max|err| vs (x*y+1) = {(o - (x*y+1.0)).abs().max().item():.3e}, allclose={torch.allclose(o, x*y+1.0)}, equal-to-fma-ref={torch.equal(o, ref)}")
k_direct_exact[(triton.cdiv(n, BLOCK),)](x, y, o, n, BLOCK=BLOCK); k_direct_loop[(PGRID,)](x, y, o, n, BLOCK=BLOCK); k_ctx_exact[(triton.cdiv(n, BLOCK),)](ctx, BLOCK=BLOCK); k_ctx_loop_nohint[(PGRID,)](ctx, BLOCK=BLOCK); torch.cuda.synchronize()
for f in sorted(glob.glob(os.environ["TRITON_CACHE_DIR"] + "/**/*.ptx", recursive=True)):
    src = open(f).read(); name = os.path.basename(f)[:-4]
    print(f"{name:20s} PTX: v4 loads={src.count('ld.global.v4'):2d} v4 stores={src.count('st.global.v4'):2d} scalar32 loads={src.count('ld.global.f32')+src.count('ld.global.b32'):2d}")
s = torch.cuda.Stream()
def time_graph(fn, iters=300, reps=20):
    gg = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s): fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(gg, stream=s):
        for _ in range(reps): fn()
    for _ in range(5): gg.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(3):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
        for _ in range(iters): gg.replay()
        e1.record(); torch.cuda.synchronize(); best = min(best, e0.elapsed_time(e1) / (iters * reps) * 1000)
    return best
print("GPU us per kernel (best of 3), graphed:")
for n in [1 << 12, 1 << 16, 1 << 20, 1 << 24]:
    x = torch.randn(n, device=dev); y = torch.randn_like(x); o = torch.empty_like(x); set_ctx(x, y, o); torch.cuda.synchronize()
    r = {}
    r["direct_exact"] = time_graph(lambda: k_direct_exact[(triton.cdiv(n, BLOCK),)](x, y, o, n, BLOCK=BLOCK))
    r["direct_loop"] = time_graph(lambda: k_direct_loop[(PGRID,)](x, y, o, n, BLOCK=BLOCK))
    r["ctx_exact"] = time_graph(lambda: k_ctx_exact[(triton.cdiv(n, BLOCK),)](ctx, BLOCK=BLOCK))
    r["ctx_loop"] = time_graph(lambda: k_ctx_loop[(PGRID,)](ctx, BLOCK=BLOCK))
    r["ctx_loop_nohint"] = time_graph(lambda: k_ctx_loop_nohint[(PGRID,)](ctx, BLOCK=BLOCK))
    print(f"n={n:>9} " + " | ".join(f"{k} {v:6.2f}" for k, v in r.items()))
