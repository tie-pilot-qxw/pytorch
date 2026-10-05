"""(B) param semantics: hold every tensor POINTER fixed, vary only M, diff the
cuBLAS kernel param blob.  Also show the full small-M table that got truncated."""
import sys, os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _wf_vfy_util import capture

dev="cuda"; bf=torch.bfloat16
K=N=256; MAXM=256
A=torch.randn(MAXM,K,device=dev,dtype=bf); B=torch.randn(K,N,device=dev,dtype=bf)
O=torch.empty(MAXM,N,device=dev,dtype=bf)
print("ptrs A=0x%X B=0x%X O=0x%X" % (A.data_ptr(), B.data_ptr(), O.data_ptr()))

blobs={}
for M in (32,48,64,96,128,192,256):
    a=A[:M]; o=O[:M]
    assert a.data_ptr()==A.data_ptr() and o.data_ptr()==O.data_ptr()
    nodes,g = capture(lambda a=a,o=o: torch.mm(a,B,out=o))
    kn=[n for n in nodes if n["type"]=="KERNEL"]
    n0=kn[0]
    print(f"M={M:4d} name={n0['name']} grid={n0['grid']} blk={n0['block']} smem={n0['smem']} "
          f"paraminfo={n0['paraminfo']} kernelParams={n0['kernelParams']} extra={n0['extra']} func=0x{n0['func']:X}")
    blobs[M]=(n0["name"], n0["func"], n0["grid"], n0["extrablob"])
    del g,nodes

print("\n--- pairwise diff of the single param struct, same func only")
import itertools
def hexd(b): return b.hex() if b else None
base=None
for M,(nm,f,grid,blob) in blobs.items():
    print(f"M={M:4d} func=0x{f:X} bloblen={len(blob) if blob else None}")
ref_M=64
refname,reffunc,_,refblob = blobs[ref_M]
for M,(nm,f,grid,blob) in blobs.items():
    if blob is None or refblob is None: continue
    if f!=reffunc:
        print(f"M={M}: different func, skip byte diff"); continue
    if len(blob)!=len(refblob):
        print(f"M={M}: different bloblen"); continue
    diffs=[i for i in range(len(blob)) if blob[i]!=refblob[i]]
    # group contiguous
    groups=[]
    for i in diffs:
        if groups and i==groups[-1][-1]+1: groups[-1].append(i)
        else: groups.append([i])
    print(f"\nM={M:4d} vs M={ref_M}: {len(diffs)} differing bytes in {len(groups)} runs")
    for gseq in groups:
        o0,o1=gseq[0],gseq[-1]+1
        va=int.from_bytes(refblob[o0:o1],'little'); vb=int.from_bytes(blob[o0:o1],'little')
        print(f"   off {o0:4d}..{o1:4d} ({o1-o0}B)  M{ref_M}={va} (0x{va:X})   M{M}={vb} (0x{vb:X})")

print("\n--- full blob hexdump for M=64 and M=128 (32B/line)")
for M in (64,128):
    b=blobs[M][3]
    if b is None: continue
    print(f"M={M} len={len(b)}")
    for i in range(0,len(b),32):
        print(f"  {i:4d}: {b[i:i+32].hex(' ')}")
print("DONE")
