"""Diagnose why synthesis worked bit-exactly for L=1792 but NaN'd for L=900/320:
is my closed-form model wrong at those L, or is the failure elsewhere?
Capture L=900/320 for GROUND TRUTH and diff against the synthesized image."""
import ctypes, math, os
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "_wf_vrfy_synth2.py")).read().split("FIT=[256,512,1024]")[0])

FIT=[256,512,1024]
C={L:cap(L) for L in FIT}
GT={L:cap(L) for L in [320,900,1792,1500]}
def magic(d):
    if d & (d-1) == 0: return 0x80000000
    L=math.ceil(math.log2(d)); return (1<<(31+L))//d + 1
def shiftv(d): return math.ceil(math.log2(d))
def model(L):
    t64=-(-L//64); t128=-(-L//128)
    return {(0,16):L,(0,20):L,(1,0):8*t64,(2,0):4*t64,(2,4):shiftv(4*t64),(2,8):magic(4*t64),
            (3,0):t128,(3,4):shiftv(t128),(3,8):magic(t128),
            **{(p,16):H*L for p in (5,6,9,10)},**{(p,20):D*L for p in (5,6,9,10)},
            **{(p,36):L-1 for p in (5,6,9,10)}}
def grid_of(L): return (min(132, B*H*(-(-L//128))),1,1)

def ptr_slots(v):
    known={"aux":v['memset'][0],"q":v['q'].data_ptr(),"out":v['out'].data_ptr()}
    s=[]
    for k,b in enumerate(v['img']):
        for o in range(0,len(b)-7,8):
            val=int.from_bytes(b[o:o+8],'little')
            for nm,base in known.items():
                if 0<=val-base<(1<<16): s.append((k,o,nm,val-base)); break
    return s
slots=ptr_slots(C[512]); pw={(k,o+d) for k,o,_,_ in slots for d in (0,4)}

for L in [320,900,1500,1792]:
    v=GT[L]; m=model(L)
    wrong=[]; unexp=[]
    for (k,o),want in m.items():
        got=int.from_bytes(v['img'][k][o:o+4],'little')
        if got!=want: wrong.append(((k,o),"actual",got,"model",want))
    for k,b in enumerate(v['img']):
        for o in range(0,len(b)-3,4):
            if (k,o) in m or (k,o) in pw: continue
            a=int.from_bytes(C[512]['img'][k][o:o+4],'little'); c2=int.from_bytes(b[o:o+4],'little')
            if a!=c2: unexp.append(((k,o),"L512",a,f"L{L}",c2))
    print(f"L={L:5d} grid actual={v['grid']} model={grid_of(L)} match={v['grid']==grid_of(L)} "
          f"block={v['block']} smem={v['smem']} | wrong={len(wrong)} unexplained={len(unexp)}")
    for x in wrong: print("     WRONG      ", x)
    for x in unexp: print("     UNEXPLAINED", x)
