import inspect, torch
import deep_gemm
import spconv.pytorch as spconv
from spconv.pytorch import ops
print("align", deep_gemm.get_mk_alignment_for_contiguous_layout())
print(deep_gemm.m_grouped_bf16_gemm_nt_contiguous.__doc__[:300])
print("get_indice_pairs", inspect.signature(ops.get_indice_pairs))
# grouped gemm with -1 padding rows
G, N, K = 27, 32, 64
A = deep_gemm.get_mk_alignment_for_contiguous_layout()
counts = [5, 0, 130, 1] + [7] * 23
rows = sum(-(-c // A) * A for c in counts)
lay = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
a = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16)
at = 0
for g, c in enumerate(counts):
    lay[at:at + c] = g
    at += -(-c // A) * A
d = torch.full((rows, N), 7.0, device="cuda", dtype=torch.bfloat16)
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, w, d, lay)
torch.cuda.synchronize()
ok = lay >= 0
ref = torch.einsum("rk,rnk->rn", a[ok].float(), w[lay[ok].long()].float())
print("rows", rows, "err", ((d[ok].float() - ref).abs().max() / ref.abs().max()).item(),
      "padding rows untouched:", bool((d[~ok] == 7).all()), "or zero:", bool((d[~ok] == 0).all()))
deep_gemm._C.describe_begin(); deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, w, d, lay); print("describe", [(x[1], x[4], len(x[6])) for x in deep_gemm._C.describe_end()])
