import sys, os, gzip, time
W=os.environ.get('DG_DATA','/workspace/_deps/data'); sys.path.insert(0,W+'/pylibs')
import numpy as np
t=time.time()
E=np.loadtxt(W+'/arxiv_d/arxiv/raw/edge.csv.gz',delimiter=',',dtype=np.int64)
N=169343
print("edges",E.shape,"load %.1fs"%(time.time()-t))
src=np.concatenate([E[:,0],E[:,1]]); dst=np.concatenate([E[:,1],E[:,0]])
order=np.argsort(src,kind='stable'); src=src[order]; dst=dst[order]
deg=np.bincount(src,minlength=N); ptr=np.concatenate([[0],np.cumsum(deg)])
print("N=%d undirected-degree: mean %.1f max %d"%(N,deg.mean(),deg.max()))
rng=np.random.default_rng(0)
train=np.loadtxt(W+'/arxiv_d/arxiv/split/time/train.csv.gz',dtype=np.int64)
print("train nodes",len(train))
def sample(seeds,fanout):
    frontier=seeds; allnodes=set(seeds.tolist()); stats=[]
    for f in fanout:
        d=deg[frontier]
        take=np.minimum(d,f)
        tot=int(take.sum())
        out=np.empty(tot,dtype=np.int64); pos=0
        # sample with replacement (DGL default for fanout<deg is w/o repl; approximate)
        for i,(v,k) in enumerate(zip(frontier,take)):
            if k==0: continue
            s=ptr[v]; e=ptr[v+1]
            if k==e-s: out[pos:pos+k]=dst[s:e]
            else: out[pos:pos+k]=dst[rng.integers(s,e,size=k)]
            pos+=k
        new=np.unique(out)
        stats.append((tot,len(new)))
        allnodes.update(new.tolist())
        frontier=new
    return stats,len(allnodes)
for B,fanout in [(1024,[15,10]),(1024,[15,10,5]),(4096,[15,10])]:
    res=[]
    steps=200
    for it in range(steps):
        seeds=rng.choice(train,size=B,replace=False)
        st,tot=sample(seeds,fanout)
        key=tuple([x for pair in st for x in pair])+(tot,)
        res.append(key)
    arr=np.array(res)
    print(f"\n--- B={B} fanout={fanout}, {steps} steps ---")
    names=[]
    for i,f in enumerate(fanout):
        names += [f"L{i+1}_edges",f"L{i+1}_newnodes"]
    names.append("total_nodes")
    for j,nm in enumerate(names):
        c=arr[:,j]
        print(f"  {nm:14s} min={c.min():8d} max={c.max():8d} mean={c.mean():10.1f} distinct_in_{steps}_steps={len(set(c.tolist())):4d}")
    print("  distinct FULL shape tuples in %d steps: %d"%(steps,len(set(map(tuple,res)))))
