"""How expensive is the v0 scheme "grid fixed at max, mask makes the surplus blocks idle"?
Compare: exact grid (needs one graph per shape) vs fixed max grid + mask (one template is enough)"""
import torch, triton, triton.language as tl
DEV="cuda"; BLOCK=1024
@triton.jit
def k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr+off, tl.load(x_ptr+off, mask=m)*2.0+1.0, mask=m)

NMAX = 1<<22
x = torch.randn(NMAX, device=DEV); o = torch.empty_like(x)
MAXGRID = triton.cdiv(NMAX, BLOCK)
s = torch.cuda.Stream()
def timeg(grid, n, iters=300, reps=20):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s): k[(grid,)](x, o, n, BLOCK=BLOCK)
    torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps): k[(grid,)](x, o, n, BLOCK=BLOCK)
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); best=1e9
    for _ in range(5):
        e0,e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
        for _ in range(iters): g.replay()
        e1.record(); torch.cuda.synchronize(); best=min(best, e0.elapsed_time(e1)/(iters*reps)*1000)
    return best
# correctness
n=777; o.zero_(); k[(MAXGRID,)](x,o,n,BLOCK=BLOCK); torch.cuda.synchronize()
ref = x[:n]*2.0+1.0
print(f"max grid + mask correctness (n=777): first n correct = {torch.allclose(o[:n],ref)}, "
      f"nothing written past n = {bool((o[n:]==0).all())}")
print(f"\nmax grid = {MAXGRID} blocks (computed for n={NMAX})")
print(f"{'actual n':>10} {'exact grid':>10} {'exact time':>10} {'max grid time':>14} {'extra blocks':>12} {'overhead':>10}")
for n in [1024, 8192, 65536, 262144, 1<<20, 1<<22]:
    ex = triton.cdiv(n, BLOCK)
    t_ex = timeg(ex, n); t_mx = timeg(MAXGRID, n)
    print(f"{n:>10} {ex:>10} {t_ex:>9.2f}us {t_mx:>13.2f}us {MAXGRID-ex:>12} {t_mx-t_ex:>+9.2f}us")
