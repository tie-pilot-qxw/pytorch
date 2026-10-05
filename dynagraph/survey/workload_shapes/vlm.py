import os,sys,json,math
W=os.environ.get('DG_DATA','/workspace/_deps/data'); sys.path.insert(0,W+'/pylibs')
import numpy as np
d=json.load(open(W+'/coco_ann/annotations/instances_train2017.json'))
wh=[(im['width'],im['height']) for im in d['images']]
# Qwen2-VL smart_resize: round each side to multiple of 28 (patch14 x merge2), clamp total tokens
def smart_resize(w,h,factor=28,minpix=4*28*28,maxpix=16384*28*28):
    hb=max(factor,round(h/factor)*factor); wb=max(factor,round(w/factor)*factor)
    if hb*wb>maxpix:
        b=math.sqrt(h*w/maxpix); hb=math.floor(h/b/factor)*factor; wb=math.floor(w/b/factor)*factor
    elif hb*wb<minpix:
        b=math.sqrt(minpix/(h*w)); hb=math.ceil(h*b/factor)*factor; wb=math.ceil(w*b/factor)*factor
    return hb,wb
tok=[]; grids=[]
for w,h in wh:
    hb,wb=smart_resize(w,h)
    g=(hb//28,wb//28); grids.append(g); tok.append(g[0]*g[1])
tok=np.array(tok)
print("COCO train2017 through Qwen2-VL smart_resize (native dynamic resolution):")
print("  distinct (grid_h,grid_w) = %d ; distinct visual-token counts = %d"%(len(set(grids)),len(set(tok.tolist()))))
print("  tokens/image: min %d p50 %d p90 %d max %d"%(tok.min(),np.percentile(tok,50),np.percentile(tok,90),tok.max()))
rng=np.random.default_rng(0)
for B in (1,4,8):
    tot=[]
    for s in range(500):
        sel=rng.choice(len(tok),size=B,replace=False); tot.append(int(tok[sel].sum()))
    print("  batch of %d images, packed (no pad): distinct total visual tokens over 500 steps = %d (range %d..%d)"%(B,len(set(tot)),min(tot),max(tot)))
