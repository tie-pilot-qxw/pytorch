import os,sys,glob
W=os.environ.get('DG_DATA','/workspace/_deps/data'); sys.path.insert(0,W+'/pylibs')
import numpy as np
files=sorted(glob.glob(W+'/md22_*.npz')+glob.glob(W+'/MD22/md22_*.npz'))+[f for f in (W+'/md17_aspirin.npz',W+'/MD17/aspirin/raw/md17_aspirin.npz') if os.path.exists(f)][:1]
rows=[]
for fp in files:
    d=np.load(fp,allow_pickle=True)
    R=d['R']; z=d['z']; N=len(z)
    nf=min(3000,R.shape[0])
    sub=R[:nf]
    for cutoff in (5.0,):
        E=np.empty(nf,dtype=np.int64)
        for k,p in enumerate(sub):
            dm=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)
            E[k]=int(((dm<cutoff)&(dm>0)).sum())
        ch=(np.diff(E)!=0).mean()
        rows.append((os.path.basename(fp),N,nf,int(E.min()),int(E.max()),float(E.mean()),float(E.std()),len(set(E.tolist())),ch))
        print("%-28s N=%4d frames=%4d pairs(5A): min=%6d max=%6d mean=%8.1f std=%6.2f distinct=%4d consec-change=%.0f%%"%(
            os.path.basename(fp),N,nf,E.min(),E.max(),E.mean(),E.std(),len(set(E.tolist())),100*ch))
print()
print("scaling of pair-count std with N (CLT expects std ~ sqrt(N)):")
for r in sorted(rows,key=lambda x:x[1]):
    print("  N=%4d  mean_pairs=%9.1f  pairs/atom=%5.2f  std=%7.2f  std/sqrt(N)=%6.3f  distinct_in_%d=%d"%(r[1],r[5],r[5]/r[1],r[6],r[6]/np.sqrt(r[1]),r[2],r[7]))
