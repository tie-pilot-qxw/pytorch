"""cuBLAS: time of mm at exact M vs M padded to a bucket (pow2 / multiple of 128), BERT-base GEMM shapes, bf16.
Each timing replays a CUDA graph of 20 mms (no launch overhead), so this is GPU time only."""
import torch
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
dev = "cuda"
shapes = {"qkv/o 768x768": (768, 768), "ffn1 768x3072": (768, 3072), "ffn2 3072x768": (3072, 768), "lmhead 768x30522": (768, 30522)}
def t(M, K, N):
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16); b = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    for _ in range(3): a @ b
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): a @ b
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    best = 1e9
    for _ in range(5):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e) / 20)
    return best * 1e3
p2 = lambda m: 1 << (m - 1).bit_length()
r128 = lambda m: (m + 127) // 128 * 128
Ms = [1, 7, 33, 100, 200, 300, 490, 700, 1000, 1500, 2100, 3000, 4000, 6000, 9000, 12000, 16000]
for name, (K, N) in shapes.items():
    print(f"== {name}  (us: exact / pad128 / pow2)")
    tot = [0, 0, 0]
    for M in Ms:
        x = [t(M, K, N), t(r128(M), K, N), t(p2(M), K, N)]
        tot = [u + v for u, v in zip(tot, x)]
        print(f"  M={M:6d}  {x[0]:8.1f} {x[1]:8.1f} ({x[1]/x[0]:.2f}x) {x[2]:8.1f} ({x[2]/x[0]:.2f}x)")
    print(f"  sum over Ms: pad128 {tot[1]/tot[0]:.3f}x  pow2 {tot[2]/tot[0]:.3f}x")
