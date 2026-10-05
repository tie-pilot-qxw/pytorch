import torch, triton, triton.language as tl
dev="cuda"; BLOCK=1024
@triton.jit
def body(x_ptr, y_ptr, o_ptr, b, n, BLOCK: tl.constexpr):
    offs = b * BLOCK + tl.arange(0, BLOCK); m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) * tl.load(y_ptr + offs, mask=m) + 1.0, mask=m)
@triton.jit
def k_direct_exact(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    body(x_ptr, y_ptr, o_ptr, tl.program_id(0), n, BLOCK)
@triton.jit
def k_direct_loop(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    for b in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)): body(x_ptr, y_ptr, o_ptr, b, n, BLOCK)
@triton.jit
def k_ctx_loop(ctx, BLOCK: tl.constexpr):
    x_ptr = tl.load(ctx + 0).to(tl.pointer_type(tl.float32)); y_ptr = tl.load(ctx + 1).to(tl.pointer_type(tl.float32)); o_ptr = tl.load(ctx + 2).to(tl.pointer_type(tl.float32)); n = tl.load(ctx + 3).to(tl.int32)
    for b in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)): body(x_ptr, y_ptr, o_ptr, b, n, BLOCK)
NSM = torch.cuda.get_device_properties(0).multi_processor_count; PGRID = NSM * 8
ctx = torch.zeros(4, dtype=torch.int64, device=dev); pin = torch.zeros(4, dtype=torch.int64).pin_memory()
def set_ctx(x, y, o): pin.copy_(torch.tensor([x.data_ptr(), y.data_ptr(), o.data_ptr(), x.numel()])); ctx.copy_(pin, non_blocking=True)
s = torch.cuda.Stream()
def time_graph(fn, iters=300, reps=20):
    gg = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s): fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(gg, stream=s):
        for _ in range(reps): fn()
    for _ in range(5): gg.replay()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(5):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
        for _ in range(iters): gg.replay()
        e1.record(); torch.cuda.synchronize(); best = min(best, e0.elapsed_time(e1) / (iters * reps) * 1000)
    return best
print("n NOT divisible by 16 (apples-to-apples with dynamic shapes): GPU us/kernel, best of 5")
for n in [4096+7, 65536+7, (1<<20)+7, (1<<24)+7]:
    x = torch.randn(n, device=dev); y = torch.randn_like(x); o = torch.empty_like(x); set_ctx(x, y, o); torch.cuda.synchronize()
    td = time_graph(lambda: k_direct_exact[(triton.cdiv(n, BLOCK),)](x, y, o, n, BLOCK=BLOCK))
    tl_ = time_graph(lambda: k_direct_loop[(PGRID,)](x, y, o, n, BLOCK=BLOCK))
    tc = time_graph(lambda: k_ctx_loop[(PGRID,)](ctx, BLOCK=BLOCK))
    print(f"n={n:>9}  direct_exact {td:6.2f} | direct_persistent {tl_:6.2f} | ctx_persistent {tc:6.2f}   (ctx overhead {tc-td:+.2f} us, {100*(tc/td-1):+.0f}%)")
