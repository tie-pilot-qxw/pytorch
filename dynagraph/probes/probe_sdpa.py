#!/usr/bin/env python3
"""SDPA region: Inductor lowers F.scaled_dot_product_attention to a fallback call of aten._scaled_dot_product_flash_attention
(or efficient) -- multiple outputs, self-allocated outputs, optional dropout. Take the child route and see whether it is served.

    UPDATE=host|device python probe_sdpa.py
"""
from __future__ import annotations
import logging, os, sys
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo
import torch.nn.functional as F
import torch._inductor.config as ic
import torch._inductor.cudagraph_trees as ct

H, DH = 8, 64
class Attn(torch.nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.qkv = torch.nn.Linear(H * DH, 3 * H * DH)
        self.o = torch.nn.Linear(H * DH, H * DH)
        self.causal = causal
    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, H, DH).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.o(a.transpose(1, 2).reshape(B, L, H * DH)).sum(-1)

class Grab(logging.Handler):
    def __init__(self): super().__init__(); self.msgs = []
    def emit(self, rec): self.msgs.append(rec.getMessage())

def run(name, m, dtype, shapes):
    torch._dynamo.reset()
    ic.triton.dynagraph = True
    ic.triton.dynagraph_extern_child = True
    m = m.cuda().to(dtype).eval()
    grab = Grab(); lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.INFO); lg.addHandler(grab)
    n_rec = {"v": 0}; orig = ct.CUDAGraphTreeManager.record_function
    def spy(self, *a, **kw): n_rec["v"] += 1; return orig(self, *a, **kw)
    ct.CUDAGraphTreeManager.record_function = spy
    try:
        f = torch.compile(m, dynamic=True, mode="reduce-overhead")
        worst = 0.0
        with torch.no_grad():
            for L in shapes:
                g = torch.Generator(device="cuda"); g.manual_seed(L)
                x = torch.randn(2, L, H * DH, device="cuda", generator=g).to(dtype)
                for _ in range(2): got = f(x)
                ref = m(x)
                worst = max(worst, ((got.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-6)).item())
    except Exception as exc:
        print(f"    {name:<22} ERR {type(exc).__name__}: {str(exc)[:120]}"); return 1
    finally:
        ct.CUDAGraphTreeManager.record_function = orig; lg.removeHandler(grab)
    tags = sorted({t.split("[", 1)[1].split("]", 1)[0] for t in grab.msgs if t.startswith("DynaGraph fallback [")})
    detail = [t[:110] for t in grab.msgs if t.startswith("DynaGraph fallback [")][:1]
    ok = not tags and n_rec["v"] == 0 and worst <= 2e-2
    print(f"    {name:<22} recordings {n_rec['v']}  rel diff {worst:.1e}  {','.join(tags) or '-'} {'OK' if ok else 'FAIL'} {detail[0] if detail and not ok else ''}")
    return 0 if ok else 1

def main() -> int:
    shapes = [512, 200, 333, 64]
    bad = 0
    bad += run("flash bf16 causal", Attn(True), torch.bfloat16, shapes)
    bad += run("flash bf16", Attn(False), torch.bfloat16, shapes)
    bad += run("sdpa fp32", Attn(False), torch.float32, shapes)
    print("\n  " + ("all passed" if not bad else f"{bad} failed"))
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
