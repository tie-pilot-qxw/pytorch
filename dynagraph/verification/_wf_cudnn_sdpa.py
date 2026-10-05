"""(A)(B)(C) for SDPA: FLASH vs EFFICIENT vs CUDNN backends, two seq lengths.
Also: which backend's kernels live in PyTorch's own .so (source available) vs a closed lib."""
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from _wf_cudnn_lib import describe, graph_nodes, demangle

torch.manual_seed(0)
DEV="cuda"; DT=torch.bfloat16
B, H, D = 2, 8, 64
LMAX = 2048
QB = torch.randn(B,H,LMAX,D,device=DEV,dtype=DT)
KB = torch.randn(B,H,LMAX,D,device=DEV,dtype=DT)
VB = torch.randn(B,H,LMAX,D,device=DEV,dtype=DT)

def capture(backend, L, causal=True):
    q,k,v = QB[:,:,:L], KB[:,:,:L], VB[:,:,:L]
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), sdpa_kernel(backend):
        for _ in range(4):
            o = F.scaled_dot_product_attention(q,k,v,is_causal=causal)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g), sdpa_kernel(backend):
        o = F.scaled_dot_product_attention(q,k,v,is_causal=causal)
    torch.cuda.synchronize()
    return g, describe(graph_nodes(g.raw_cuda_graph()))

BACKENDS = [("FLASH", SDPBackend.FLASH_ATTENTION),
            ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
            ("CUDNN", SDPBackend.CUDNN_ATTENTION),
            ("MATH", SDPBackend.MATH)]
KEEP = {}
for nm, bk in BACKENDS:
    print(f"\n{'='*92}\n=== SDPA backend {nm} ===")
    for L in (512, 1024, 1536):
        try:
            g, info = capture(bk, L)
            KEEP[(nm,L)] = info
            print(f"\n  L={L}: {len(info)} nodes")
            for d in info:
                if d["type"]!="KERNEL":
                    print(f"    {d['type']}"); continue
                pi = d.get("param_info",[]); tot=(pi[-1][1]+pi[-1][2]) if pi else 0
                print(f"    grid={str(d['grid']):16s} blk={str(d['block']):14s} smem={d['smem']:6d} "
                      f"np={len(pi)} pbytes={tot:5d} kp={d['has_kernelParams']}")
                print(f"      raw : {d['name'][:160]}")
                dm = demangle(d['name'])
                if dm != d['name']: print(f"      demg: {dm[:230]}")
        except Exception as e:
            print(f"  L={L}: FAILED {type(e).__name__}: {str(e)[:200]}")

print(f"\n\n{'#'*92}\n# (C) kernel stability across L, per backend")
for nm,_ in BACKENDS:
    sigs={}
    for L in (512,1024,1536):
        info = KEEP.get((nm,L))
        if not info: continue
        sig = tuple(d.get("name",d["type"]) for d in info)
        sigs.setdefault(sig,[]).append(L)
    print(f"\n  {nm}: {len(sigs)} distinct node-sequence(s) over L in {sorted(L for (n,L) in KEEP if n==nm)}")
    for s,Ls in sigs.items():
        print(f"     L={Ls}  nodes={len(s)}")
        for x in s: print(f"         {x[:110]}")

print(f"\n\n{'#'*92}\n# (B) byte diff L=512 vs L=1024 (same-kernel nodes only)")
for nm,_ in BACKENDS:
    a,b = KEEP.get((nm,512)), KEEP.get((nm,1024))
    if not a or not b: continue
    print(f"\n  --- {nm} ---")
    for na,nb in zip(a,b):
        if na["type"]!="KERNEL": continue
        if na.get("name")!=nb.get("name"):
            print(f"    node{na['i']}: KERNEL CHANGED"); continue
        print(f"    node{na['i']} grid {na['grid']}->{nb['grid']} {na['name'][:70]}")
        pa=na.get("param_bytes") or []; pb=nb.get("param_bytes") or []
        if not pa: print("      <no kernelParams>"); continue
        for (i,off,sz,ba),(_,_,_,bbb) in zip(pa,pb):
            if ba is None or bbb is None or ba==bbb: continue
            runs=[]; j=0
            while j<sz:
                if ba[j]!=bbb[j]:
                    kk=j
                    while kk<sz and ba[kk]!=bbb[kk]: kk+=1
                    lo=(j//4)*4; hi=((kk+3)//4)*4; runs.append((lo,hi)); j=kk
                else: j+=1
            merged=[]
            for lo,hi in runs:
                if merged and lo<=merged[-1][1]: merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
                else: merged.append((lo,hi))
            desc=[]
            for lo,hi in merged:
                va=int.from_bytes(ba[lo:hi],"little"); vb=int.from_bytes(bbb[lo:hi],"little")
                desc.append(f"[{lo}:{hi}] {'PTR' if va>0x1000000000 else 'int'} {va}->{vb}")
            print(f"      param{i}(off={off},sz={sz}): {len(merged)} run(s): " + "; ".join(desc[:14]))

# ---------- provenance: which .so contains the kernel name ----------
print(f"\n\n{'#'*92}\n# provenance: which shared object contains each kernel symbol")
import glob
libs = sorted(set(glob.glob("/workspace/pytorch-main/build/lib/libtorch_cuda.so")
              + glob.glob("/usr/lib/x86_64-linux-gnu/libcudnn*.so.9")
              + glob.glob("/usr/lib/x86_64-linux-gnu/libcudnn_engines*.so.9")
              + glob.glob("/usr/local/cuda/lib64/libcudnn*.so.9")
              + glob.glob("/usr/local/cuda/lib64/libcublas*.so.1[0-9]")
              + glob.glob("/usr/lib/x86_64-linux-gnu/libcublas*.so.1[0-9]")
              + glob.glob("/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/*.so.9")
              + glob.glob("/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/*.so.1[0-9]")))
libs = [l for l in libs if os.path.getsize(l) > 100000]
print("  scanning:", [os.path.basename(l) for l in libs])
names = sorted({d.get("name") for info in KEEP.values() for d in info
                if d["type"]=="KERNEL" and d.get("name")})
for n in names:
    hits=[]
    for l in libs:
        try:
            r = subprocess.run(["grep","-c","-a","-F",n,l],capture_output=True,text=True,timeout=120)
            if r.stdout.strip() not in ("","0"): hits.append(os.path.basename(l))
        except Exception as e: pass
    print(f"\n  {n[:120]}\n      -> {hits if hits else 'NOT FOUND in scanned libs'}")
