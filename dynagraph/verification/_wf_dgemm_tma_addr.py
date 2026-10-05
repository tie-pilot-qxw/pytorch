"""Change only the addresses of a/b/d (same shapes): see which bytes of the DeepGEMM launch params change and how addresses are encoded."""
import torch
import vllm.third_party.deep_gemm as deep_gemm
from torch.utils import _capture_launch as cl
M, N, K = 15000, 256, 256
ts = [(torch.randn(M, K, device="cuda", dtype=torch.bfloat16), torch.randn(N, K, device="cuda", dtype=torch.bfloat16),
       torch.empty(M, N, device="cuda", dtype=torch.bfloat16)) for _ in range(2)]
f = lambda a, b, d: deep_gemm.bf16_gemm_nt(a, b, d)
for t in ts: f(*t)
r = [cl._record_raw(f, t, {})[0] for t in ts]
for j, (x, y) in enumerate(zip(r[0].params, r[1].params)):
    if x != y:
        d = [i for i in range(len(x)) if x[i] != y[i]]
        print(f"param {j} ({len(x)}B) differs at bytes {d}")
for k, t in enumerate(ts[0]):
    p = t.data_ptr()
    for j, x in enumerate(r[0].params):
        for sh in (0, 4):
            v = (p >> sh).to_bytes(8, "little")[:6 if sh else 8]
            at = x.find(v)
            if at >= 0:
                print(f"tensor {'abd'[k]} ptr {p:#x} >>{sh} found in param {j} at byte {at}")
