"""Compile small models with dynamic=True and count extern_kernels.* / torch.ops.* in wrapper."""
import os, re, sys, collections
import torch, torch.nn as nn, torch.nn.functional as F
import torch._dynamo
import torch._inductor.config as icfg
from torch._inductor.utils import run_and_get_code

torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_scalar_outputs = True
OUT = os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "_wf_externsurv_wrappers")
os.makedirs(OUT, exist_ok=True)
dev = "cuda"

CALL_RE = re.compile(r"(extern_kernels\.[A-Za-z_][\w.]*|torch\.ops\.[\w.]+|aten\.[\w.]+)\s*\(")

def report(tag, codes):
    print(f"\n########## {tag} ##########")
    for i, c in enumerate(codes):
        p = f"{OUT}/{tag}_{i}.py"
        open(p, "w").write(c)
        cnt = collections.Counter(CALL_RE.findall(c))
        ntriton = len(re.findall(r"^\s*(triton_\w+)\.run\(|\.run\(", c, re.M))
        tri_defs = sorted(set(re.findall(r"(triton_[a-z_]+_\d+)", c)))
        print(f"-- {p}  (triton kernel objs: {len(tri_defs)})")
        if not cnt:
            print("   (no extern/aten calls)")
        for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
            print(f"   {v:3d} x {k}")

# ---------------- 1. transformer block ----------------
class Attn(nn.Module):
    def __init__(self, d=256, h=4):
        super().__init__()
        self.h, self.d = h, d
        self.qkv = nn.Linear(d, 3*d)
        self.o = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4*d); self.fc2 = nn.Linear(4*d, d)
    def forward(self, x):
        B, S, D = x.shape
        y = self.ln1(x)
        qkv = self.qkv(y).view(B, S, 3, self.h, D//self.h).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1,2).reshape(B, S, D)
        x = x + self.o(a)
        y = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(y)))

m = Attn().to(dev).to(torch.bfloat16).eval()
cm = torch.compile(m, dynamic=True)
x = torch.randn(2, 128, 256, device=dev, dtype=torch.bfloat16)
with torch.no_grad():
    _, codes = run_and_get_code(cm, x)
report("transformer_fwd_bf16", codes)

# manual bmm/baddbmm attention (no SDPA) -> bmm path
class ManualAttn(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.wq = nn.Linear(d, d, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, q, k, v, bias):
        s = torch.bmm(q, k.transpose(1,2))
        s = torch.baddbmm(bias, q, k.transpose(1,2))
        p = s.softmax(-1)
        return torch.bmm(p, v)
mm2 = ManualAttn().to(dev).to(torch.bfloat16)
cm2 = torch.compile(mm2, dynamic=True)
q = torch.randn(4, 64, 32, device=dev, dtype=torch.bfloat16)
k = torch.randn(4, 64, 32, device=dev, dtype=torch.bfloat16)
v = torch.randn(4, 64, 32, device=dev, dtype=torch.bfloat16)
bias = torch.randn(4, 64, 64, device=dev, dtype=torch.bfloat16)
with torch.no_grad():
    _, codes = run_and_get_code(cm2, q, k, v, bias)
report("bmm_baddbmm", codes)

# ---------------- 2. convnet ----------------
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 64, 1)          # 1x1
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.b2(self.c2(x)))
        x = self.c3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
cn = ConvNet().to(dev).eval()
ccn = torch.compile(cn, dynamic=True)
xi = torch.randn(4, 3, 64, 64, device=dev)
with torch.no_grad():
    _, codes = run_and_get_code(ccn, xi)
report("convnet_fwd_fp32", codes)

# convnet training (fwd+bwd)
cn_t = ConvNet().to(dev).train()
ccn_t = torch.compile(cn_t, dynamic=True)
def fwdbwd(inp):
    out = ccn_t(inp).sum()
    out.backward()
    return out
_, codes = run_and_get_code(fwdbwd, torch.randn(4, 3, 64, 64, device=dev, requires_grad=True))
report("convnet_fwdbwd_fp32", codes)

# ---------------- 3. misc data-dependent ops ----------------
class Misc(nn.Module):
    def forward(self, x, idx, src, idx2):
        s, si = torch.sort(x, dim=-1)
        t, ti = torch.topk(x, 4, dim=-1)
        c = torch.cumsum(x, dim=-1)
        z = x.clone()
        z.scatter_add_(1, idx, src)
        w = x.clone()
        w[idx2] = 1.0
        return s.sum(), t.sum(), c.sum(), z.sum(), w.sum(), si.sum(), ti.sum()
ms = Misc().to(dev)
cms = torch.compile(ms, dynamic=True)
xm = torch.randn(8, 64, device=dev)
idx = torch.randint(0, 64, (8, 16), device=dev)
src = torch.randn(8, 16, device=dev)
idx2 = torch.randint(0, 8, (4,), device=dev)
with torch.no_grad():
    _, codes = run_and_get_code(cms, xm, idx, src, idx2)
report("misc_sort_topk_cumsum_scatter", codes)

# nonzero separately (data-dependent output shape)
def nz(x):
    n = torch.nonzero(x > 0)
    return n.sum()
cnz = torch.compile(nz, dynamic=True)
with torch.no_grad():
    _, codes = run_and_get_code(cnz, torch.randn(256, device=dev))
report("nonzero", codes)
print("\nDONE")
