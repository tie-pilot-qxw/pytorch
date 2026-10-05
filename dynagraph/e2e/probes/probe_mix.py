"""DeepGEMM inline site + cuBLAS child site in one DG region, dynamic M. Reproduce illegal instruction."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # e2e/, for harness
os.environ["GEMM"] = "deepgemm"; os.environ["AMP"] = "1"
import harness, torch
MODE = os.environ.get("PMODE", "mix")
class M(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.a = torch.nn.Linear(128, 128)   # aligned: DeepGEMM
        s.b = torch.nn.Linear(50, 128)    # K=50: stays cuBLAS
    def forward(s, x, e):
        if MODE == "dg_only":
            return s.a(x).relu()
        if MODE == "cublas_only":
            return s.b(e).relu()
        return s.a(x).relu() + s.b(e)
torch.manual_seed(0)
m = M().cuda()
f = harness.compiled(m, "dg")
for n in [1000, 1200, 1100, 1300, 900, 1250, 1400]:
    x = torch.randn(n, 128, device="cuda"); e = torch.randn(n, 50, device="cuda")
    with harness.amp(), torch.no_grad():
        y = f(x, e); r = m(x, e)
    torch.cuda.synchronize()
    print(n, (y.float() - r.float()).abs().max().item(), flush=True)
print("served", harness.C.served, "other", harness.C.other, harness.C.tags)
