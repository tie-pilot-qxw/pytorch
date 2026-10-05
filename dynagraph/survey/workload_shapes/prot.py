import gzip,os,sys
W=os.environ.get('DG_DATA','/workspace/_deps/data')
L=[];cur=0
with gzip.open(W+'/human_proteome.fasta.gz','rt') as f:
    for line in f:
        if line.startswith('>'):
            if cur: L.append(cur)
            cur=0
        else: cur+=len(line.strip())
if cur: L.append(cur)
L.sort(); n=len(L)
print("UniProt human reference proteome (UP000005640): %d sequences"%n)
print("length: min %d p50 %d p90 %d p99 %d max %d mean %.0f"%(L[0],L[n//2],L[int(n*.9)],L[int(n*.99)],L[-1],sum(L)/n))
print("distinct lengths: %d"%len(set(L)))
for cap in (1024,2048,4096):
    sub=[x for x in L if x<=cap]
    print("  <=%d: %d seqs (%.1f%%), distinct lengths %d"%(cap,len(sub),100*len(sub)/n,len(set(sub))))
# AF3-style bucket padding waste, compute ~ O(L^2) for pair repr / O(L^3)-ish for triangle ops
import bisect
def waste(buckets,p):
    tot=0; pad=0
    for x in L:
        b=buckets[bisect.bisect_left(buckets,x)] if x<=buckets[-1] else None
        if b is None: continue
        tot+=x**p; pad+=b**p
    return pad/tot
for name,bk in [("AF3 default [256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120]",[256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120]),
                ("coarse [512,1024,2048,4096]",[512,1024,2048,4096]),
                ("fine, step 64 up to 5120",list(range(64,5121,64)))]:
    print("  bucket set %s -> pad/actual work: L^2 %.2fx, L^3 %.2fx, nbuckets=%d"%(name,waste(bk,2),waste(bk,3),len(bk)))
