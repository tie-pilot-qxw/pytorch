# Q3: which Inductor kernels actually request global scratch (sized by launch grid)?
import torch, torch._inductor.config as icfg, glob, os, re
torch._dynamo.reset()

def probe(tag, fn, *a):
    torch._dynamo.reset()
    c = torch.compile(fn, dynamic=True)
    c(*a); torch.cuda.synchronize()
    hits = []
    for d in glob.glob(os.path.join(torch._inductor.codecache.cache_dir(), "**", "*.py"), recursive=True):
        try: src = open(d).read()
        except Exception: continue
        if "triton_" not in src: continue
        if "make_tensor_descriptor" in src or "_experimental_make_tensor_descriptor" in src:
            hits.append(os.path.basename(d))
    print(f"{tag:<28} kernels using make_tensor_descriptor: {len(hits)}")

dev='cuda'
probe("pointwise",      lambda x: (x*2+1).relu(),              torch.randn(4096,4096,device=dev))
probe("reduction",      lambda x: x.sum(-1),                    torch.randn(4096,4096,device=dev))
probe("layernorm",      lambda x: torch.nn.functional.layer_norm(x,(4096,)), torch.randn(4096,4096,device=dev))
probe("matmul(bf16)",   lambda a,b: a@b, torch.randn(4096,4096,device=dev,dtype=torch.bfloat16),
                                          torch.randn(4096,4096,device=dev,dtype=torch.bfloat16))

# direct: read global_scratch_size off the compiled binaries in the cache
n_tot=n_nz=0
for f in glob.glob(os.path.join(torch._inductor.codecache.cache_dir(),"**","*.json"),recursive=True):
    try: s=open(f).read()
    except Exception: continue
    for m in re.finditer(r'"global_scratch_size"\s*:\s*(\d+)', s):
        n_tot+=1
        if int(m.group(1))>0: n_nz+=1
print(f"\ncompiled-kernel metadata in cache: {n_tot} entries, {n_nz} with global_scratch_size > 0")
