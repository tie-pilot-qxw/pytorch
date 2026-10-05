import os, struct, sys, glob
from collections import Counter
W=os.environ.get('DG_DATA','/workspace/_deps/data')
def flac_info(p):
    with open(p,'rb') as f:
        hdr=f.read(4)
        if hdr!=b'fLaC': return None
        b=f.read(4)
        blk=f.read(34)
        # STREAMINFO: 16b minblk,16b maxblk,24b minframe,24b maxframe,20b rate,3b ch,5b bps,36b samples
        bits=int.from_bytes(blk[10:18],'big')
        rate=(bits>>44)&0xFFFFF
        total=bits & ((1<<36)-1)
        return rate,total
ds=[]
for p in glob.glob(W+'/libri/LibriSpeech/dev-clean/**/*.flac',recursive=True):
    r=flac_info(p)
    if r: ds.append(r[1]/r[0])
ds.sort()
import statistics
n=len(ds)
print("LibriSpeech dev-clean utterances:",n)
print("duration s: min %.2f p50 %.2f p90 %.2f p99 %.2f max %.2f mean %.2f"%(ds[0],ds[n//2],ds[int(n*.9)],ds[int(n*.99)],ds[-1],sum(ds)/n))
for hop_ms,name in [(10,'10ms hop (80x melframes)'),(20,'20ms hop')]:
    fr=[int(d*1000/hop_ms) for d in ds]
    print(f"  frames @{hop_ms}ms: distinct={len(set(fr))} range={min(fr)}..{max(fr)}")
# conv subsample by 4 (wav2vec/whisper-like encoder output len)
enc=[int(d*1000/10)//4 for d in ds]
print("  encoder frames (10ms hop /4): distinct=%d range=%d..%d"%(len(set(enc)),min(enc),max(enc)))
# batch of 16, padded to max in batch -> distinct padded lengths
import random
random.seed(0)
for B in (8,16,32):
    mx=[]
    idx=list(range(n))
    for trial in range(200):
        random.shuffle(idx)
        for i in range(0,n-B,B):
            mx.append(max(int(ds[j]*100) for j in idx[i:i+B]))
    print(f"  random batching B={B}: distinct padded frame-lengths over {len(mx)} batches = {len(set(mx))}")
    break
