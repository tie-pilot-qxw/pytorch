import os,sys,csv
W=os.environ.get('DG_DATA','/workspace/_deps/data'); sys.path.insert(0,W+'/pylibs')
import numpy as np
u=[]
with open(W+'/ml/ml-25m/ratings.csv') as f:
    f.readline()
    for line in f:
        u.append(int(line.split(',',1)[0]))
u=np.array(u,dtype=np.int64)
cnt=np.bincount(u)
cnt=cnt[cnt>0]
cnt.sort()
n=len(cnt)
print("MovieLens-25M: %d users, %d interactions"%(n,cnt.sum()))
print("history length per user: min %d p50 %d p90 %d p99 %d p99.9 %d max %d mean %.1f"%(
 cnt[0],cnt[n//2],cnt[int(n*.9)],cnt[int(n*.99)],cnt[int(n*.999)],cnt[-1],cnt.mean()))
print("distinct history lengths: %d"%len(set(cnt.tolist())))
rng=np.random.default_rng(0)
for B in (256,1024,4096):
    tot=[];mx=[]
    for s in range(500):
        sel=rng.choice(n,size=B,replace=False)
        tot.append(int(cnt[sel].sum())); mx.append(int(cnt[sel].max()))
    print("  jagged batch B=%d, 500 steps: total_len distinct=%d range %d..%d ; if padded to batch-max: distinct max-len=%d range %d..%d, padded/actual FLOP ratio mean %.1fx"%(
        B,len(set(tot)),min(tot),max(tot),len(set(mx)),min(mx),max(mx), np.mean([m*B/t for m,t in zip(mx,tot)])))
    # truncated-history (SASRec style cap 200)
    c2=np.minimum(cnt,200)
    tot2=[];
    for s in range(500):
        sel=rng.choice(n,size=B,replace=False); tot2.append(int(c2[sel].sum()))
    print("     with history truncated to 200 (SASRec-style): total_len distinct=%d range %d..%d, pad-to-200 waste %.2fx"%(
        len(set(tot2)),min(tot2),max(tot2), 200*B/np.mean(tot2)))
