"""DeepGEMM bf16_gemm_nt(a (M,K), b (N,K), d): the major of each side is inferred from its stride. Measure all four major combinations x various misalignments."""
import torch
import vllm.third_party.deep_gemm as dg
torch.manual_seed(0)
def mk(r, c, major):
    return torch.randn(r, c, device="cuda", dtype=torch.bfloat16) if major == "row" else torch.randn(c, r, device="cuda", dtype=torch.bfloat16).t()
for am in ("row", "col"):
    for bm in ("row", "col"):
        for M, N, K in [(13001, 256, 128), (256, 128, 13001), (256, 128, 13000), (40, 40, 128), (13001, 40, 128), (256, 47, 128), (100, 128, 100), (13000, 128, 256), (8, 8, 8)]:
            a = mk(M, K, am); b = mk(K, N, bm)
            d = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            try:
                dg.bf16_gemm_nt(a, b.t(), d); torch.cuda.synchronize()
                r = a.float() @ b.float()
                res = f"err {((d.float() - r).abs().max() / r.abs().max()).item():.1e}"
            except Exception as e:
                res = "ERR " + str(e).split("\n")[0][-60:]
            print(f"a {am} b {bm} MNK {(M, N, K)} a.stride {a.stride()} bT.stride {b.t().stride()}: {res}")
