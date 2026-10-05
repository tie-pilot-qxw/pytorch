import os,sys,json
W=os.environ.get('DG_DATA','/workspace/_deps/data'); sys.path.insert(0,W+'/pylibs')
import numpy as np
d=json.load(open(W+'/coco_ann/annotations/instances_train2017.json'))
imgs=d['images']; anns=d['annotations']
print("COCO train2017: %d images, %d instance annotations"%(len(imgs),len(anns)))
wh=[(im['width'],im['height']) for im in imgs]
print("raw (W,H) distinct pairs: %d"%len(set(wh)))
from collections import Counter
c=Counter(wh); print("  top-5 raw sizes:",c.most_common(5))
def resize_shortest(w,h,short=800,maxs=1333):
    s=min(short/min(w,h), maxs/max(w,h))
    return int(round(h*s)),int(round(w*s))
r=[resize_shortest(w,h) for w,h in wh]
print("after ResizeShortestEdge(800,1333): distinct (H,W) = %d"%len(set(r)))
def pad(x,m): return ((x+m-1)//m)*m
for m in (32,64):
    p=[(pad(a,m),pad(b,m)) for a,b in r]
    print("  padded to multiple of %d: distinct = %d"%(m,len(set(p))))
# batched: detectron2 pads batch to max H,W in batch
rng=np.random.default_rng(0)
R=np.array(r)
for B in (2,8,16):
    shapes=[]
    for s in range(500):
        sel=rng.choice(len(R),size=B,replace=False)
        H=pad(int(R[sel,0].max()),32); Wd=pad(int(R[sel,1].max()),32)
        shapes.append((H,Wd))
    print("  batch B=%d (pad-to-batch-max, /32), 500 steps: distinct=%d"%(B,len(set(shapes))))
# instances per image
ipi=Counter()
for a in anns: ipi[a['image_id']]+=1
v=np.array([ipi.get(im['id'],0) for im in imgs])
v.sort(); n=len(v)
print("instances/image: min %d p50 %d p90 %d p99 %d max %d mean %.2f distinct=%d"%(v[0],v[n//2],v[int(n*.9)],v[int(n*.99)],v[-1],v.mean(),len(set(v.tolist()))))
# total instances per batch (jagged targets)
for B in (8,16):
    tot=[]
    for s in range(500):
        sel=rng.choice(n,size=B,replace=False); tot.append(int(v[sel].sum()))
    print("  total instances in batch B=%d: distinct=%d range %d..%d"%(B,len(set(tot)),min(tot),max(tot)))
