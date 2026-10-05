"""FINAL test vs claim 6.  Two parts.
 A) formula validation against ground-truth captures over a wide L set
 B) the decisive one: capture ONLY L=512 in this process, then synthesize the whole
    param image + grid for L values that are NEVER captured, patch, replay, compare
    to eager.  If this passes, 'you can only get the bytes by really capturing' is false.
"""
import ctypes, math, os, sys
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "_wf_vrfy_synth2.py")).read().split("FIT=[256,512,1024]")[0])

def magic(d):
    if d & (d-1)==0: return 0x80000000
    return (1<<(31+math.ceil(math.log2(d))))//d + 1
def sh(d): return math.ceil(math.log2(d))
def model(L):
    t=-(-L//128)                      # number of 128-row query tiles
    ctas = B*H*t                      # unclamped total CTAs
    return {(0,16):L,(0,20):L,(1,0):ctas,(2,0):ctas//2,(2,4):sh(ctas//2),(2,8):magic(ctas//2),
            (3,0):t,(3,4):sh(t),(3,8):magic(t),
            **{(p,16):H*L for p in (5,6,9,10)},**{(p,20):D*L for p in (5,6,9,10)},
            **{(p,36):L-1 for p in (5,6,9,10)}}
def grid_of(L): return (min(132, B*H*(-(-L//128))),1,1)

MODE = sys.argv[1]
if MODE == "A":
    base = cap(512)
    known={"aux":base['memset'][0],"q":base['q'].data_ptr(),"out":base['out'].data_ptr()}
    slots=[]
    for k,b in enumerate(base['img']):
        for o in range(0,len(b)-7,8):
            v=int.from_bytes(b[o:o+8],'little')
            for nm,bs in known.items():
                if 0<=v-bs<(1<<16): slots.append((k,o,nm,v-bs)); break
    pw={(k,o+d) for k,o,_,_ in slots for d in (0,4)}
    Ls=[7,31,64,100,129,192,257,320,333,500,512,640,777,900,1024,1333,1500,1792,2048,3000,4096,5000]
    allok=True
    for L in Ls:
        v=cap(L); m=model(L); wrong=[]; unexp=[]
        for (k,o),want in m.items():
            got=int.from_bytes(v['img'][k][o:o+4],'little')
            if got!=want: wrong.append(((k,o),got,want))
        for k,b in enumerate(v['img']):
            for o in range(0,len(b)-3,4):
                if (k,o) in m or (k,o) in pw: continue
                if bytes(b[o:o+4])!=bytes(base['img'][k][o:o+4]):
                    unexp.append(((k,o),int.from_bytes(base['img'][k][o:o+4],'little'),
                                  int.from_bytes(b[o:o+4],'little')))
        ok = not wrong and not unexp and v['grid']==grid_of(L) and v['block']==base['block'] and v['smem']==base['smem']
        allok &= ok
        print(f"  L={L:5d} grid={v['grid']} pred={grid_of(L)} block={v['block']} smem={v['smem']} "
              f"wrong={len(wrong)} unexplained={len(unexp)} -> {'OK' if ok else 'MISMATCH'}")
        for x in wrong[:5]: print("       WRONG", x)
        for x in unexp[:5]: print("       UNEXPLAINED", x)
        del v
    print(f"\n  closed-form model reproduces EVERY byte of the param image over {len(Ls)} seqlens: {allok}")

else:
    print("### only L=512 is ever captured in this process")
    base = cap(512)
    known={"aux":base['memset'][0],"q":base['q'].data_ptr(),"out":base['out'].data_ptr()}
    slots=[]
    for k,b in enumerate(base['img']):
        for o in range(0,len(b)-7,8):
            v=int.from_bytes(b[o:o+8],'little')
            for nm,bs in known.items():
                if 0<=v-bs<(1<<16): slots.append((k,o,nm,v-bs)); break
    print("  pointer slots:", slots)
    KEEP=[]
    base['g'].instantiate(); ex=base['g'].raw_cuda_graph_exec()
    for Ltgt in [320,900,1333,1792,3000,4096,129]:
        qN=torch.randn(B,H,Ltgt,D,device=DEV,dtype=DT)
        outN=torch.empty(B,H,Ltgt,D,device=DEV,dtype=DT)
        aux=torch.zeros(1024*1024,device=DEV,dtype=torch.float32)
        roles={"q":qN.data_ptr(),"out":outN.data_ptr(),"aux":aux.data_ptr()}
        img=[bytearray(b) for b in base['img']]
        for (k,o),val in model(Ltgt).items(): img[k][o:o+4]=int(val).to_bytes(4,'little')
        for k,o,nm,d in slots: img[k][o:o+8]=(roles[nm]+d).to_bytes(8,'little')
        n=len(img); ptrs=(ctypes.c_void_p*n)(); bufs=[]
        for i,b in enumerate(img):
            cb=(ctypes.c_ubyte*len(b)).from_buffer_copy(bytes(b)); bufs.append(cb); ptrs[i]=ctypes.cast(cb,ctypes.c_void_p)
        KEEP.append((bufs,ptrs,qN,outN,aux))
        new=KNP(); new.func=base['p'].func
        new.gx,new.gy,new.gz=grid_of(Ltgt)
        new.bx,new.by,new.bz=base['block']; new.smem=base['smem']
        new.kernelParams=ctypes.cast(ptrs,ctypes.POINTER(ctypes.c_void_p)); new.extra=None; new.kern=None; new.ctx=None
        ms=MSP(); ms.dst=roles["aux"]; ms.pitch=0; ms.value=0
        ms.elementSize=base['memset'][3]; ms.width=base['memset'][4]; ms.height=base['memset'][5]
        rcm=libcuda.cuGraphExecMemsetNodeSetParams(ctypes.c_void_p(ex),ctypes.c_void_p(base['mn']),ctypes.byref(ms),ctypes.c_void_p(0))
        rck=libcuda.cuGraphExecKernelNodeSetParams_v2(ctypes.c_void_p(ex),ctypes.c_void_p(base['kn']),ctypes.byref(new))
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            ref=F.scaled_dot_product_attention(qN,qN,qN,is_causal=True)
        outN.fill_(float("nan")); torch.cuda.synchronize()
        base['g'].replay(); torch.cuda.synchronize()
        nan=bool(torch.isnan(outN).any())
        print(f"  L={Ltgt:5d} grid={grid_of(Ltgt)} rc=({en(rcm)},{en(rck)}) "
              f"any_nan={nan} bit_exact_vs_eager={bool(torch.equal(outN,ref))} "
              f"max_abs_diff={0.0 if nan else (outN.float()-ref.float()).abs().max().item()}")
