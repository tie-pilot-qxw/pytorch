import os, torch
M = int(os.environ.get("WF_M", "947"))
DT = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("WF_DT", "fp32")]
op = os.environ.get("WF_OP", "addmm")
x = torch.randn(M, 512, device="cuda", dtype=DT)
w = torch.randn(512, 512, device="cuda", dtype=DT)
b = torch.randn(512, device="cuda", dtype=DT)
torch.cuda.synchronize()
print("### MARK ###", flush=True)
y = torch.addmm(b, x, w) if op == "addmm" else torch.mm(x, w)
torch.cuda.synchronize()
print("### END ###", flush=True)
