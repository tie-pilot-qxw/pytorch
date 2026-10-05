"""Why does cuBLAS pick 61 kernels, and what does ONE device-parameterized kernel cost?

Part A: map M -> cuBLAS kernel, decode tile from the nvjet name, test the wave-quantization hypothesis.
Part B: benchmark torch.mm (cuBLAS heuristic, host-side dispatch) vs a single persistent Triton
        kernel whose grid is FIXED and whose M is READ FROM DEVICE MEMORY (i.e. CUDA-graph-safe).
"""
import math, re, torch, triton, triton.language as tl
from collections import Counter, OrderedDict
from torch.profiler import profile, ProfilerActivity

torch.manual_seed(0)
DEV = "cuda"
DT = torch.bfloat16
K = N = 4096
NSM = torch.cuda.get_device_properties(0).multi_processor_count
print(f"# device={torch.cuda.get_device_name(0)}  SMs={NSM}  K={K} N={N} dtype=bf16")

Ms = sorted(set([1, 2, 4, 8, 16, 32, 64] + [128 * i for i in range(1, 65)] +
                [i for i in range(100, 8192, 337)] + [513, 1025, 1731, 2049, 3000, 5000, 7777, 8192]))
xmax = torch.randn(8192, K, device=DEV, dtype=DT)
w = torch.randn(N, K, device=DEV, dtype=DT)
wt = w.t().contiguous()

# ---------------- Part A: which kernel per M, and why -----------------
for M in [128, 1024, 4096]:
    _ = xmax[:M] @ wt
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for M in Ms:
        _ = xmax[:M] @ wt
    torch.cuda.synchronize()
ev = [e.name for e in prof.events() if e.device_type.name == "CUDA" and ("nvjet" in e.name or "gemm" in e.name.lower() or "sm90" in e.name)]
print(f"\n## Part A: {len(Ms)} distinct M -> {len(set(ev))} distinct cuBLAS kernels ({len(ev)} launches recorded)")

pat = re.compile(r"nvjet_sm\d+_\w+?_(\d+)x(\d+)_(\d+)x(\d+)_")
rows, waves_ok = [], []
if len(ev) == len(Ms):
    for M, name in zip(Ms, ev):
        m = pat.search(name)
        if not m:
            continue
        BM, BN, BK, ST = map(int, m.groups())
        ctas = math.ceil(M / BM) * math.ceil(N / BN)
        waves = ctas / NSM
        rows.append((M, BM, BN, ctas, waves, name))
        # "good" wave quantization = last wave is nearly full (or only one partial wave total)
        waves_ok.append(waves - math.floor(waves) if waves > 1 else waves)
    print(f"{'M':>6} {'tile BMxBN':>12} {'CTAs':>6} {'waves=CTAs/SM':>14}   kernel")
    for M, BM, BN, ctas, wv, name in rows[::max(1, len(rows)//18)]:
        print(f"{M:>6} {BM:>5}x{BN:<6} {ctas:>6} {wv:>14.2f}   {name[:62]}")
    frac = [w - math.floor(w) for _, _, _, _, w, _ in rows]
    good = sum(1 for f in frac if f == 0 or f > 0.75)
    print(f"\n  tail-wave fill: {good}/{len(frac)} shapes land on a full or >75%-full last wave "
          f"(random tiling would give ~25%)")
    bn_vals = sorted(set(r[2] for r in rows))
    print(f"  distinct BN values cuBLAS used: {bn_vals}")
    print(f"  distinct BM values cuBLAS used: {sorted(set(r[1] for r in rows))}")
else:
    print(f"  (kernel/M zip mismatch: {len(ev)} kernels vs {len(Ms)} M; skipping tile analysis)")

# ---------------- Part B: one device-parameterized persistent kernel -----------------
@triton.jit
def persistent_mm(a_ptr, b_ptr, c_ptr, M_ptr, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)
    M = tl.load(M_ptr)                       # <-- problem size lives in device memory
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_K)
    for tile_id in range(start_pid, num_tiles, NUM_SMS):     # <-- fixed grid, dynamic tile count
        gsz = GROUP_M * num_pid_n
        group_id = tile_id // gsz
        first_m = group_id * GROUP_M
        gm = min(num_pid_m - first_m, GROUP_M)
        pid_m = first_m + ((tile_id % gsz) % gm)
        pid_n = (tile_id % gsz) // gm
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_am = tl.where(offs_m < M, offs_m, 0)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(k_tiles):
            offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + offs_am[:, None] * sam + offs_k[None, :] * sak)
            b = tl.load(b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn)
            acc = tl.dot(a, b, acc)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc.to(tl.bfloat16), mask=mask)

CFGS = [(128, 256, 64, 8, 3, 8), (128, 128, 64, 8, 4, 8), (64, 256, 64, 8, 4, 4),
        (128, 256, 64, 8, 4, 8), (256, 128, 64, 8, 3, 8)]
M_dev = torch.zeros(1, dtype=torch.int32, device=DEV)
c_buf = torch.empty(8192, N, device=DEV, dtype=DT)

def run_triton(M, cfg, grid_mult=1):
    BM, BN, BK, GM, ST, NW = cfg
    persistent_mm[(NSM * grid_mult,)](xmax, w.t(), c_buf, M_dev, N, K,
                                      xmax.stride(0), xmax.stride(1), w.t().stride(0), w.t().stride(1),
                                      c_buf.stride(0), c_buf.stride(1),
                                      BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
                                      NUM_SMS=NSM * grid_mult, num_stages=ST, num_warps=NW)

M_dev.fill_(1024)
run_triton(1024, CFGS[0]); torch.cuda.synchronize()
ref = (xmax[:1024] @ w.t()).float()
err = (c_buf[:1024].float() - ref).abs().max().item() / ref.abs().max().item()
print(f"\n## Part B: persistent Triton correctness (M from device, M=1024): rel max err {err:.2e}")

def bench(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters): fn()
        e1.record(); torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / iters)
    return best

BENCH_M = [128, 256, 512, 777, 1024, 1731, 2048, 3000, 4096, 6144, 8192]
print(f"\n{'M':>6} {'cuBLAS TF':>10} {'1-cfg pers TF':>14} {'best-cfg pers TF':>17} {'1cfg/cuBLAS':>12} {'best/cuBLAS':>12}")
tot = {"cublas": 0.0, "fixed": 0.0, "best": 0.0}
FIXED = CFGS[0]
for M in BENCH_M:
    a = xmax[:M]
    flops = 2 * M * N * K
    t_cb = bench(lambda: a @ wt)
    M_dev.fill_(M); torch.cuda.synchronize()
    t_fx = bench(lambda: run_triton(M, FIXED))
    t_bs = t_fx
    for cfg in CFGS[1:]:
        try: t_bs = min(t_bs, bench(lambda: run_triton(M, cfg)))
        except Exception: pass
    tf = lambda t: flops / (t * 1e-3) / 1e12
    tot["cublas"] += t_cb; tot["fixed"] += t_fx; tot["best"] += t_bs
    print(f"{M:>6} {tf(t_cb):>10.1f} {tf(t_fx):>14.1f} {tf(t_bs):>17.1f} "
          f"{t_cb/t_fx:>11.0%} {t_cb/t_bs:>11.0%}")
print(f"\n  aggregate over these M: 1-config persistent = {tot['cublas']/tot['fixed']:.0%} of cuBLAS, "
      f"best-of-{len(CFGS)} persistent = {tot['cublas']/tot['best']:.0%} of cuBLAS")
print("  (shared GPU: absolute TFLOPS are contended, ratios are the signal)")
