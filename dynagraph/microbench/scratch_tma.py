import torch, torch._inductor.config as c, glob, os, re
c.max_autotune=True; c.max_autotune_gemm_backends="TRITON"
try: c.triton.enable_persistent_tma_matmul=True
except Exception as e: print("cfg:",e)
a=torch.randn(4096,4096,device="cuda",dtype=torch.bfloat16); b=a.clone()
torch.compile(lambda x,y:x@y, dynamic=True)(a,b); torch.cuda.synchronize()
cd=torch._inductor.codecache.cache_dir()
tot=nz=mx=0
for f in glob.glob(os.path.join(cd,"**","*.json"),recursive=True):
    try: s=open(f,errors="ignore").read()
    except Exception: continue
    for m in re.finditer(r'"global_scratch_size"\s*:\s*(\d+)', s):
        tot+=1; v=int(m.group(1))
        if v>0: nz+=1; mx=max(mx,v)
d=[f for f in glob.glob(os.path.join(cd,"**","*.py"),recursive=True)
   if "make_tensor_descriptor" in open(f,errors="ignore").read()]
print(f"\nTMA persistent matmul -> kernels containing make_tensor_descriptor: {len(d)}")
print(f"metadata entries: {tot}, with global_scratch_size>0: {nz}, max: {mx} bytes/CTA")
