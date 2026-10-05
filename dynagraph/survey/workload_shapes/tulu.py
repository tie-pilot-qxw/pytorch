import os,sys
W=os.environ.get('DG_DATA','/workspace/_deps/data'); sys.path.insert(0,W+'/pylibs')
import numpy as np, pyarrow.parquet as pq
from tokenizers import Tokenizer
tok=Tokenizer.from_file(W+'/qwen_tok.json')
t=pq.read_table(W+'/tulu0.parquet')
print("columns:",t.column_names,"rows:",t.num_rows)
msgs=t['messages'].to_pylist()
rng=np.random.default_rng(0)
idx=rng.choice(len(msgs),size=min(40000,len(msgs)),replace=False)
texts=[]
for i in idx:
    m=msgs[i]
    texts.append("\n".join((x.get('content') or '') for x in m))
enc=tok.encode_batch(texts)
L=np.array([len(e.ids) for e in enc])
L.sort(); n=len(L)
print("tulu-3-sft shard, %d sampled conversations (Qwen2.5 tokenizer)"%n)
print("tokens: min %d p50 %d p90 %d p99 %d max %d mean %.0f"%(L[0],L[n//2],L[int(n*.9)],L[int(n*.99)],L[-1],L.mean()))
print("distinct token lengths: %d  (range width %d)"%(len(set(L.tolist())),L[-1]-L[0]))
rng2=np.random.default_rng(1)
for B in (8,16,32):
    mx=[];tot=[]
    for s in range(500):
        sel=rng2.choice(n,size=B,replace=False)
        mx.append(int(L[sel].max())); tot.append(int(L[sel].sum()))
    print("  pad-to-longest B=%d, 500 steps: distinct padded seqlen=%d (range %d..%d), padded/actual tokens=%.2fx"%(
        B,len(set(mx)),min(mx),max(mx),np.mean([m*B/t for m,t in zip(mx,tot)])))
    print("     varlen/jagged (no pad) B=%d: distinct total_tokens=%d (range %d..%d)"%(B,len(set(tot)),min(tot),max(tot)))
# packing to fixed budget: shapes fixed, but nseq per pack varies
for budget in (4096,8192,32768):
    order=rng2.permutation(n); packs=[]; cur=0; cnt=0
    for i in order:
        l=min(int(L[i]),budget)
        if cur+l>budget:
            packs.append((cur,cnt)); cur=0; cnt=0
        cur+=l; cnt+=1
    seqs=[c for _,c in packs]
    print("  greedy packing to %d tokens: %d packs, seqs/pack min %d max %d distinct=%d  (total-token shape is FIXED at %d; only cu_seqlens length varies)"%(
        budget,len(packs),min(seqs),max(seqs),len(set(seqs)),budget))
