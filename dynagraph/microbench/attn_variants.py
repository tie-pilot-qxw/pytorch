"""How many distinct kernels does attention dispatch across shapes, vs cuBLAS's 36-61?"""
import torch, random
from collections import Counter, defaultdict
from torch.profiler import profile, ProfilerActivity
torch.manual_seed(0); random.seed(0)
DEV="cuda"; DT=torch.bfloat16
H, D = 32, 128            # Llama-7B-ish: 32 heads, headdim 128

def kernels_of(fn, tag):
    fn()  # warm
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type.name=="CUDA"
            and not e.name.startswith(("Memset","Memcpy","void at::native","at::native"))]

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
    import flash_attn; print(f"# flash_attn {flash_attn.__version__}, heads={H} headdim={D}, bf16")
except ImportError:
    print("no flash_attn"); raise SystemExit

def make_varlen(seqlens):
    tot = sum(seqlens)
    q = torch.randn(tot, H, D, device=DEV, dtype=DT, requires_grad=True)
    k = torch.randn(tot, H, D, device=DEV, dtype=DT, requires_grad=True)
    v = torch.randn(tot, H, D, device=DEV, dtype=DT, requires_grad=True)
    cu = torch.tensor([0]+list(torch.tensor(seqlens).cumsum(0)), device=DEV, dtype=torch.int32)
    return q,k,v,cu,max(seqlens)

# ---- 1. varlen fwd+bwd across many length distributions (the training case)
configs = []
for b, base in [(1,512),(2,1024),(4,777),(8,333),(4,2048),(1,8192),(16,128),(3,5000),(2,4095),(8,1731)]:
    configs.append([max(1,base + random.randint(-base//3, base//3)) for _ in range(b)])
fwd_k, bwd_k, grids = Counter(), Counter(), defaultdict(set)
for sl in configs:
    q,k,v,cu,mx = make_varlen(sl)
    def f():
        o = flash_attn_varlen_func(q,k,v,cu,cu,mx,mx,causal=True)
        o.sum().backward()
    names = kernels_of(f, "varlen")
    for n in names:
        (fwd_k if "bwd" not in n.lower() and "backward" not in n.lower() else bwd_k)[n]+=1
print(f"\n## FA2 varlen fwd+bwd, {len(configs)} different (batch, seqlen-distribution) configs")
print(f"   total tokens ranged {min(sum(c) for c in configs)} .. {max(sum(c) for c in configs)}, max_seqlen 85..10905")
print(f"   distinct FORWARD kernels : {len(fwd_k)}")
for n,c in fwd_k.most_common(): print(f"      {c:2d}x  {n[:100]}")
print(f"   distinct BACKWARD kernels: {len(bwd_k)}")
for n,c in bwd_k.most_common(): print(f"      {c:2d}x  {n[:100]}")

# ---- 2. does headdim / causal change the kernel? (these are STATIC in a training run)
print(f"\n## sensitivity: what actually changes the attention kernel")
for label, (hd, causal) in [("headdim=64,causal",(64,True)),("headdim=128,causal",(128,True)),
                             ("headdim=128,non-causal",(128,False)),("headdim=256,causal",(256,True))]:
    try:
        q = torch.randn(2,2048,H,hd,device=DEV,dtype=DT,requires_grad=True)
        k = torch.randn_like(q,requires_grad=True); v = torch.randn_like(q,requires_grad=True)
        def f(): flash_attn_func(q,k,v,causal=causal).sum().backward()
        ns = set(kernels_of(f,label))
        print(f"   {label:24s} -> {len(ns)} kernels")
    except Exception as e: print(f"   {label:24s} -> {type(e).__name__}")

# ---- 3. baseline for comparison: torch SDPA backends
print(f"\n## torch SDPA across seqlens (for comparison)")
from torch.nn.attention import sdpa_kernel, SDPBackend
for bk, nm in [(SDPBackend.FLASH_ATTENTION,"flash"),(SDPBackend.CUDNN_ATTENTION,"cudnn"),(SDPBackend.EFFICIENT_ATTENTION,"mem_eff")]:
    allk=set()
    try:
        with sdpa_kernel(bk):
            for L in [128,333,512,777,1024,2048,3000,4096,8192]:
                q=torch.randn(2,H,L,D,device=DEV,dtype=DT,requires_grad=True)
                k=torch.randn_like(q,requires_grad=True); v=torch.randn_like(q,requires_grad=True)
                def f(): torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).sum().backward()
                allk |= set(kernels_of(f,nm))
        print(f"   {nm:8s} over 9 seqlens -> {len(allk)} distinct kernels")
    except Exception as e: print(f"   {nm:8s} -> {type(e).__name__}: {str(e)[:60]}")
