"""DeepGEMM with dims that are not multiples of 8 but strides padded to 8: correct?"""
import torch
import vllm.third_party.deep_gemm as dg
def padded(r, c, p):
    return torch.randn(r, p, device="cuda", dtype=torch.bfloat16)[:, :c]
for M, N, K in [(1000, 128, 50), (1000, 50, 128), (128, 50, 1000), (50, 128, 1000), (1000, 3, 50)]:
    pk, pn = -(-K // 8) * 8, -(-N // 8) * 8
    a = padded(M, K, pk)                   # (M,K) stride (pk,1)
    b = padded(N, K, pk).t()               # (K,N) as N x K rows padded
    d = torch.empty(M, pn, device="cuda", dtype=torch.bfloat16)[:, :N]
    try:
        dg.bf16_gemm_nt(a, b.t(), d); torch.cuda.synchronize()
        r = a.float() @ b.float()
        print((M, N, K), "err", ((d.float() - r).abs().max() / r.abs().max()).item())
    except Exception as e:
        print((M, N, K), "ERR", str(e)[-80:])
    # dW-like: a col-major (M,K) with M stride padded
    a2 = padded(K, M, -(-M // 8) * 8).t()  # (M,K) strides (1, pM)
    try:
        dg.bf16_gemm_nt(a2, b.t(), d); torch.cuda.synchronize()
        r = a2.float() @ b.float()
        print((M, N, K), "colA err", ((d.float() - r).abs().max() / r.abs().max()).item())
    except Exception as e:
        print((M, N, K), "colA ERR", str(e)[-80:])
