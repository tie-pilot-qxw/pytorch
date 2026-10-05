import os,sys
W=os.environ.get('DG_DATA','/workspace/_deps/data'); sys.path.insert(0,W+'/pylibs')
import numpy as np
rng=np.random.default_rng(0)
print("MoE expert-GEMM shape space (ANALYTIC simulation, uniform router; real routers are more skewed)")
for (T,E,k,name) in [(4096,8,2,"Mixtral-8x7B style"),(8192,64,6,"Qwen-MoE style"),(8192,256,8,"DeepSeek-V3 style")]:
    steps=200
    cnts=[]
    for s in range(steps):
        # top-k routing of T tokens to E experts, uniform -> multinomial with n=T*k
        c=rng.multinomial(T*k,[1/E]*E)
        cnts.append(tuple(c.tolist()))
    a=np.array(cnts)
    print(f"  {name}: T={T} tokens, E={E} experts, top-{k}")
    print(f"    per-expert m: mean {a.mean():.0f}, min {a.min()}, max {a.max()}, std {a.std():.1f}")
    print(f"    distinct m for expert 0 over {steps} steps: {len(set(a[:,0].tolist()))}")
    print(f"    distinct FULL (m_1..m_E) tuples over {steps} steps: {len(set(cnts))}")
    cap=1.25
    capv=int(cap*T*k/E)
    waste=capv*E/(T*k)
    drop=np.maximum(a-capv,0).sum()/ (T*k*steps)
    print(f"    capacity-factor {cap} padding: pad/actual = {waste:.2f}x, tokens dropped = {100*drop:.2f}%")
