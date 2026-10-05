import torch, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _wf_aten_common import probe, capture, kernels, short, pshow

torch.cuda.init()
dev = torch.device("cuda")
x = torch.randn(1024, 512, device=dev)
y = torch.randn(1024, 512, device=dev)
out = torch.empty_like(x)

g, nodes = capture(lambda: torch.add(x, y, out=out))
print("total nodes:", len(nodes), "types:", [n["type"] for n in nodes])
for n in kernels(nodes):
    print("  func=%#x mod=%#x grid=%s block=%s smem=%d kp=%s extra=%s"
          % (n["func"], n["module"], n["grid"], n["block"], n["smem"],
             n["has_kernelParams"], n["has_extra"]))
    print("  name:", short(n["name"]))
    for p in n["params"]:
        print("     p%-2d off=%-4d sz=%-3d %s" % (p["index"], p["offset"], p["size"], pshow(p)))
print("free/total GB:", [v/2**30 for v in torch.cuda.mem_get_info()])
