import torch, triton, triton.language as tl
DEV="cuda"; DT=torch.bfloat16; K=N=4096
NSM = torch.cuda.get_device_properties(0).multi_processor_count
a_full = torch.randn(8192, K, device=DEV, dtype=DT)
b = torch.randn(K, N, device=DEV, dtype=DT).contiguous()      # proper [K,N] contiguous
c_buf = torch.empty(8192, N, device=DEV, dtype=DT)
M_dev = torch.zeros(1, dtype=torch.int32, device=DEV)
triton.set_allocator(lambda size, align, stream: torch.empty(size, device=DEV, dtype=torch.int8))

@triton.jit
def tma_pers(a_ptr, b_ptr, c_ptr, M_ptr, N, K,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NUM_SMS: tl.constexpr):
    M = tl.load(M_ptr)                                        # problem size read from device memory
    ad = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])   # TMA descriptor built on device
    bd = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BK, BN])
    cd = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BM, BN])
    nm = tl.cdiv(M, BM); nn = tl.cdiv(N, BN); nt = nm * nn
    for tid in range(tl.program_id(0), nt, NUM_SMS):
        pm = tid // nn; pn = tid % nn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(tl.cdiv(K, BK)):
            acc = tl.dot(ad.load([pm*BM, k*BK]), bd.load([k*BK, pn*BN]), acc)
        cd.store([pm*BM, pn*BN], acc.to(tl.bfloat16))

def bench(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(5):
        e0,e1=torch.cuda.Event(True),torch.cuda.Event(True); e0.record()
        for _ in range(iters): fn()
        e1.record(); torch.cuda.synchronize(); best=min(best,e0.elapsed_time(e1)/iters)
    return best

CFGS = [(128,256,64,3,8),(128,128,64,4,8),(128,64,64,4,4),(128,32,64,4,4),(64,64,64,4,4)]
def run(cfg):
    BM,BN,BK,ST,NW = cfg
    tma_pers[(NSM,)](a_full, b, c_buf, M_dev, N, K, BM=BM, BN=BN, BK=BK, NUM_SMS=NSM, num_stages=ST, num_warps=NW)

M_dev.fill_(1024); run(CFGS[0]); torch.cuda.synchronize()
ref = (a_full[:1024] @ b).float()
print(f"# TMA persistent (M from device) correctness: rel err {(c_buf[:1024].float()-ref).abs().max().item()/ref.abs().max().item():.1e}  SMs={NSM}")
print(f"\n{'M':>6} {'cuBLAS':>8} " + " ".join(f"{str(c[0])+'x'+str(c[1]):>9}" for c in CFGS) + f" {'best/cuBLAS':>12} {'bestBN':>7}")
agg_cb = agg_best = agg_fixed = 0.0
for M in [128, 256, 512, 777, 1024, 2048, 4096, 8192]:
    a = a_full[:M]; flops = 2*M*N*K
    t_cb = bench(lambda: a @ b)
    M_dev.fill_(M); torch.cuda.synchronize()
    ts = []
    for cfg in CFGS:
        try: ts.append(bench(lambda: run(cfg)))
        except Exception: ts.append(1e9)
    tf = lambda t: flops/(t*1e-3)/1e12 if t < 1e8 else 0.0
    bi = min(range(len(ts)), key=lambda i: ts[i])
    agg_cb += t_cb; agg_best += ts[bi]; agg_fixed += ts[0]
    print(f"{M:>6} {tf(t_cb):>8.0f} " + " ".join(f"{tf(t):>9.0f}" for t in ts) + f" {t_cb/ts[bi]:>11.0%} {CFGS[bi][1]:>7}")
print(f"\n  aggregate: single fixed 128x256 = {agg_cb/agg_fixed:.0%} of cuBLAS;  best-of-{len(CFGS)} variants = {agg_cb/agg_best:.0%} of cuBLAS")
print("  (shared GPU @550W cap: ratios are the signal)")
