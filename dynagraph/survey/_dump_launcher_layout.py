#!/usr/bin/env python3
"""Dump: (a) exact generated launcher body, (b) arg_tys, (c) PTX .param layout."""
from __future__ import annotations
import os, re, glob, json
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch._inductor.config as ic
ic.force_disable_caches = True

from torch._inductor.runtime import triton_heuristics as th

CAPTURED = []

orig = th.CompileResult._gen_launcher_code
def patched(self, scope, def_args, runner_args, pre_runner_lines=None):
    grid = th.GridExpr.from_meta(self.inductor_meta, self.config)
    lines = [
        f"def launcher({', '.join(def_args)}, stream):",
        *[f"    {line}" for line in grid.prefix],
        f"    grid_0 = {grid.x_grid}",
        f"    grid_1 = {grid.y_grid}",
        f"    grid_2 = {grid.z_grid}",
        *(f"    {l}" for l in (pre_runner_lines or [])),
        f"    runner({', '.join(runner_args)})",
    ]
    rec = {
        "cls": type(self).__name__,
        "grid_type": self.inductor_meta.get("grid_type"),
        "kernel_name": getattr(getattr(self, "kernel", None), "name", None)
                        or getattr(getattr(self, "kernel", None), "src", None).fn.__name__
                        if getattr(self, "kernel", None) is not None else None,
        "config": str(self.config),
        "code": "\n".join(lines),
        "arg_tys": getattr(getattr(self, "kernel", None), "arg_tys", None),
        "shared": getattr(getattr(self, "kernel", None), "shared", None),
        "signature": None,
    }
    k = getattr(self, "kernel", None)
    try:
        rec["signature"] = {str(a): str(b) for a, b in k.src.signature.items()}
    except Exception:
        try:
            rec["signature"] = {str(a): str(b) for a, b in self.compile_meta["signature"].items()}
        except Exception:
            pass
    try:
        rec["constants"] = {str(a): str(b) for a, b in self.compile_meta.get("constants", {}).items()}
    except Exception:
        pass
    CAPTURED.append(rec)
    return orig(self, scope, def_args, runner_args, pre_runner_lines=pre_runner_lines)
th.CompileResult._gen_launcher_code = patched

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(256, 512)
        self.l2 = torch.nn.Linear(512, 256)
    def forward(self, x):
        h = torch.relu(self.l1(x))
        y = torch.relu(self.l2(h)) + x
        return y, h.sum(dim=-1), torch.softmax(h, dim=-1)

m = M().cuda()
f = torch.compile(m, dynamic=True)
for n in (37, 41):
    f(torch.randn(n, 256, device="cuda"))
torch.cuda.synchronize()

print("=" * 78)
print(f"CAPTURED {len(CAPTURED)} launchers")
for r in CAPTURED:
    print("=" * 78)
    print("compile_result_class:", r["cls"], " grid_type:", r["grid_type"])
    print("kernel:", r["kernel_name"], " config:", r["config"])
    print("arg_tys:", repr(r["arg_tys"]), " shared:", r["shared"])
    print("signature:", json.dumps(r.get("signature")))
    print("constants:", json.dumps(r.get("constants")))
    print("--- launcher body ---")
    print(r["code"])

# Now dump PTX param layouts from the triton cache
from torch._inductor.runtime.runtime_utils import triton_cache_dir
cd = triton_cache_dir(torch.cuda.current_device())
print("=" * 78)
print("TRITON CACHE:", cd)
for ptx in sorted(glob.glob(os.path.join(cd, "*", "*.ptx"))):
    txt = open(ptx).read()
    m2 = re.search(r"\.visible \.entry ([^\(]+)\(([^\)]*)\)", txt, re.S)
    if not m2:
        continue
    print("-" * 78)
    print(os.path.basename(ptx))
    print(".visible .entry", m2.group(1).strip())
    for ln in m2.group(2).strip().splitlines():
        print("   ", ln.strip())
    # also grep ld.param offsets
    offs = re.findall(r"ld\.param\.(\w+)\s+%\w+, \[([\w\$]+)\];", txt)
    print("    ld.param uses:", offs[:20])
