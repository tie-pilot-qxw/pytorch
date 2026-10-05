import os,sys
W=os.environ.get('DG_DATA','/workspace/_deps/data'); sys.path.insert(0,W+'/pylibs')
import numpy as np, pyarrow.parquet as pq
t=pq.read_table(W+'/qm9.parquet',columns=['num_atoms','pos'])
na=np.array(t['num_atoms'])
print("QM9 molecules:",len(na))
print("atoms/molecule: min %d max %d mean %.2f distinct=%d"%(na.min(),na.max(),na.mean(),len(set(na.tolist()))))
vals,cnt=np.unique(na,return_counts=True)
print("  histogram:", dict(zip(vals.tolist(),cnt.tolist())))
pos=t['pos'].to_pylist()
print("computing neighbor pairs...")
rng=np.random.default_rng(0)
M=20000
idx=rng.choice(len(na),size=M,replace=False)
P=[np.array(pos[i],dtype=np.float64).reshape(-1,3) for i in idx]
for cutoff in (4.0,5.0):
    E=np.empty(M,dtype=np.int64)
    for k,p in enumerate(P):
        d=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)
        E[k]=int(((d<cutoff)&(d>0)).sum())
    A=na[idx]
    print("\ncutoff %.1f A: pairs/molecule min %d max %d mean %.1f distinct=%d"%(cutoff,E.min(),E.max(),E.mean(),len(set(E.tolist()))))
    for B in (32,64,128):
        steps=2000
        perm=rng.permutation(M)
        tuples=[]
        for s in range(0,min(steps*B,M-B),B):
            sel=perm[s:s+B]
            tuples.append((int(A[sel].sum()),int(E[sel].sum())))
        nb=len(tuples)
        print("  batch B=%d over %d batches: total_atoms distinct=%d (range %d..%d), total_edges distinct=%d (range %d..%d), (atoms,edges) tuples distinct=%d"%(
            B,nb,len(set(a for a,_ in tuples)),min(a for a,_ in tuples),max(a for a,_ in tuples),
            len(set(e for _,e in tuples)),min(e for _,e in tuples),max(e for _,e in tuples),
            len(set(tuples))))
