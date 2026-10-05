#!/usr/bin/env python3
"""Host-overhead breakdown of one FA3 call: real fwd (issues the launch, no sync) vs fwd_describe (only
writes the sink), on typical vLLM decode parameters (paged KV, varlen, caller-supplied scheduler_metadata,
fixed num_splits)."""
import glob, os, time, torch
import vllm.vllm_flash_attn  # noqa: F401  loads _vllm_fa3_C
torch.ops.load_library(glob.glob(os.environ.get("DG_DEPS", "/workspace/_deps") + "/fa3d-src/build/_fa3d_C*.so")[0])
from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata

B, H, HK, D, PAGE, NB = 8, 16, 8, 128, 16, 512
q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
kv = torch.randn(NB, 2, PAGE, HK, D, device="cuda", dtype=torch.bfloat16)
k, v = kv[:, 0], kv[:, 1]
out = torch.empty_like(q)
cu_q = torch.arange(B + 1, device="cuda", dtype=torch.int32)
seqused = torch.full((B,), 300, device="cuda", dtype=torch.int32)
bt = torch.arange(B * 32, device="cuda", dtype=torch.int32).view(B, 32)
splits = 32
sm = get_scheduler_metadata(B, 1, 300, H, HK, D, seqused, cu_seqlens_q=cu_q, page_size=PAGE,
                            causal=True, num_splits=splits)
args = [q, k, v, None, None, None, out, cu_q, None, None, None, seqused, 1, 300, bt, None, None,
        None, None, None, None, None, None, D ** -0.5, True, -1, -1, 0.0, True, sm, splits, None, 0,
        None, 1, 0, None]
sink = torch.zeros(1 << 16, dtype=torch.int64)
for name, fn in (("fwd (launch)", lambda: torch.ops._vllm_fa3_C.fwd(*args)),
                 ("fwd_describe", lambda: torch.ops._fa3d_C.fwd_describe(sink, *args)),
                 ("fwd_describe + tolist", lambda: (torch.ops._fa3d_C.fwd_describe(sink, *args), sink.tolist()))):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(300):
        fn()
    dt = (time.perf_counter() - t0) / 300
    torch.cuda.synchronize()
    print(f"{name:24s} {dt * 1e6:7.1f} us/call")
print("records:", int(sink[0]), "first kind/func:", sink[1:3].tolist())
