"""VERIFY claims 7/8/11/12: does cuBLAS change kernel with M? is it deterministic
(by NAME, not just by pointer)? is node count really constant for the GEMM family?"""
import sys, os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _wf_vfy_util import capture, short

dev = "cuda"; bf = torch.bfloat16
Ms = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192]

def mm_case(M, dt):
    a = torch.randn(M, 256, device=dev, dtype=dt)
    b = torch.randn(256, 256, device=dev, dtype=dt)
    o = torch.empty(M, 256, device=dev, dtype=dt)
    return lambda: torch.mm(a, b, out=o), (a, b, o)

print("=== EXP1  mm bf16  K=N=256, M sweep, pass0 ascending / pass1 descending (same process)")
recs = {}
for p, order in enumerate([Ms, list(reversed(Ms))]):
    for M in order:
        fn, keep = mm_case(M, bf)
        nodes, g = capture(fn)
        kn = [n for n in nodes if n["type"] == "KERNEL"]
        rec = dict(ntot=len(nodes), types=[n["type"] for n in nodes],
                   names=[n["name"] for n in kn], grids=[n["grid"] for n in kn],
                   smem=[n["smem"] for n in kn], funcs=[n["func"] for n in kn],
                   mods=[n["module"] for n in kn], pinfo=[n["paraminfo"] for n in kn])
        recs.setdefault(M, []).append(rec)
        del g, keep, nodes
    torch.cuda.empty_cache()

for M in Ms:
    a, b = recs[M]
    same_name = a["names"] == b["names"]
    same_func = a["funcs"] == b["funcs"]
    print(f"M={M:5d} nodes={a['ntot']} {a['types']} grid={a['grids']} smem={a['smem']} "
          f"nameStable={same_name} funcStable={same_func}")
    print(f"        name={a['names'][0] if a['names'] else None}")
    if not same_name:
        print(f"        PASS1 name={b['names']}")

print("\n--- distinct kernel names seen over the M sweep (bf16 mm):")
seen = {}
for M in Ms:
    seen.setdefault(recs[M]["names"][0] if False else recs[M][0]["names"][0], []).append(M)
for k, v in seen.items():
    print(f"   {v}  ->  {k}")

print("\n--- func handle -> name map (are two handles ever the same name?)")
h2n = {}
for M in Ms:
    for p in (0, 1):
        for f, n in zip(recs[M][p]["funcs"], recs[M][p]["names"]):
            h2n.setdefault(f, set()).add((n, M))
for f, s in sorted(h2n.items()):
    names = {x[0] for x in s}
    print(f"   0x{f:X}  Ms={sorted({x[1] for x in s})}  distinct_names={len(names)}  {list(names)[0][:70]}")

print("\n=== EXP1b  node COUNT for the whole GEMM family across shapes")
cases = []
for M in (8, 128, 1024, 8192):
    a = torch.randn(M,256,device=dev,dtype=bf); b=torch.randn(256,256,device=dev,dtype=bf)
    bias = torch.randn(256,device=dev,dtype=bf); o=torch.empty(M,256,device=dev,dtype=bf)
    cases.append((f"addmm_bf16_M{M}", (lambda a=a,b=b,bias=bias,o=o: torch.addmm(bias,a,b,out=o))))
for B in (1, 4, 32):
    for M in (8, 128, 1024):
        a=torch.randn(B,M,256,device=dev,dtype=bf); b=torch.randn(B,256,256,device=dev,dtype=bf)
        o=torch.empty(B,M,256,device=dev,dtype=bf)
        cases.append((f"bmm_bf16_B{B}_M{M}", (lambda a=a,b=b,o=o: torch.bmm(a,b,out=o))))
        bi=torch.randn(B,M,256,device=dev,dtype=bf)
        cases.append((f"baddbmm_bf16_B{B}_M{M}", (lambda bi=bi,a=a,b=b,o=o: torch.baddbmm(bi,a,b,out=o))))
for tag, fn in cases:
    nodes, g = capture(fn)
    print(f"{tag:28s} nodes={len(nodes)} {[n['type'] for n in nodes]}")
    for n in nodes:
        if n["type"] == "KERNEL":
            print(f"      {n['name'][:80]}  grid={n['grid']} smem={n['smem']} nparams={len(n['paraminfo'])}")
    del g, nodes
torch.cuda.empty_cache()
print("DONE")
