import os,sys
W=os.environ.get('DG_DATA','/workspace/_deps/data'); sys.path.insert(0,W+'/pylibs')
import numpy as np
d=np.load(W+'/md17_aspirin.npz')
print("keys:",list(d.keys()))
R=d['R']; z=d['z']
print("frames",R.shape,"atoms",len(z),"species",np.unique(z))
n=R.shape[0]
sub=R[:20000]   # consecutive MD frames
for cutoff in (4.0,5.0,6.0):
    E=np.empty(len(sub),dtype=np.int64)
    for k,p in enumerate(sub):
        dm=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)
        E[k]=int(((dm<cutoff)&(dm>0)).sum())
    print("cutoff %.1f A, FIXED 21 atoms, %d consecutive MD frames: pairs min %d max %d mean %.1f distinct=%d"%(
        cutoff,len(sub),E.min(),E.max(),E.mean(),len(set(E.tolist()))))
    # how often does the pair count change between consecutive frames?
    ch=(np.diff(E)!=0).mean()
    print("   consecutive-frame change rate: %.1f%%   (a new shape this often)"%(100*ch))
    # first-occurrence curve: how many distinct values seen after k frames
    seen=set(); curve=[]
    for i,v in enumerate(E.tolist()):
        seen.add(v)
        if i+1 in (10,100,1000,5000,20000): curve.append((i+1,len(seen)))
    print("   distinct values after k frames:",curve)
