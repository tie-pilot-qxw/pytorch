"""THE decisive test the other agent never ran: can cuGraphExecKernelNodeSetParams
actually (a) patch a cuBLAS node's opaque param buffer to a different M, and
(b) swap the node's func to a different cuBLAS kernel variant?"""
import os, sys, ctypes, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from _wf_vfy_util import capture, graph_nodes, KNP, _cu

dev="cuda"; bf=torch.bfloat16
K=N=256; MAXM=256
torch.manual_seed(0)
A=torch.randn(MAXM,K,device=dev,dtype=bf); Bm=torch.randn(K,N,device=dev,dtype=bf)
O=torch.empty(MAXM,N,device=dev,dtype=bf)
SENT=torch.tensor(-999.0,dtype=bf)

def cap_mm(M):
    a=A[:M]; o=O[:M]
    nodes,g = capture(lambda a=a,o=o: torch.mm(a,Bm,out=o))
    g.instantiate()
    return nodes,g

def kparams(n):  return n["func"], n["grid"], n["block"], n["smem"], n["extrablob"]

def set_exec(g, node, func, grid, block, smem, blob, hold):
    buf=(ctypes.c_ubyte*len(blob)).from_buffer_copy(blob)
    sz=ctypes.c_size_t(len(blob))
    extra=(ctypes.c_void_p*5)()
    extra[0]=ctypes.c_void_p(1); extra[1]=ctypes.cast(buf,ctypes.c_void_p)
    extra[2]=ctypes.c_void_p(2); extra[3]=ctypes.cast(ctypes.pointer(sz),ctypes.c_void_p)
    extra[4]=ctypes.c_void_p(0)
    hold += [buf,sz,extra]
    p=KNP(); p.func=func
    p.gridDimX,p.gridDimY,p.gridDimZ=grid
    p.blockDimX,p.blockDimY,p.blockDimZ=block
    p.sharedMemBytes=smem
    p.kernelParams=None
    p.extra=ctypes.cast(extra,ctypes.POINTER(ctypes.c_void_p))
    p.kern=None; p.ctx=None
    r=_cu.cuGraphExecKernelNodeSetParams_v2(ctypes.c_void_p(g.raw_cuda_graph_exec()),
                                            ctypes.c_void_p(node), ctypes.byref(p))
    nm=ctypes.c_char_p(); _cu.cuGetErrorName(r,ctypes.byref(nm))
    return r, nm.value

hold=[]
OFFS=[276,420,596,660,980]

print("### TEST A: capture M=64, patch the 5 'M-1' bytes to 47, replay -> does it become M=48?")
n64,g64 = cap_mm(64)
k=[x for x in n64 if x["type"]=="KERNEL"][0]
f,grid,blk,sm,blob = kparams(k)
print(f"  captured: {k['name']} grid={grid} smem={sm} blob={len(blob)}B")
print(f"  bytes at {OFFS} = {[blob[o] for o in OFFS]}  (expect 63)")
nb=bytearray(blob)
for o in OFFS: nb[o]=47
r,nm = set_exec(g64, k["node"], f, grid, blk, sm, bytes(nb), hold)
print(f"  cuGraphExecKernelNodeSetParams_v2 -> {r} {nm}")
O.fill_(-999.0); torch.cuda.synchronize()
g64.replay(); torch.cuda.synchronize()
ref48 = torch.mm(A[:48], Bm)
ok48 = torch.equal(O[:48], ref48)
tail_untouched = bool((O[48:64] == SENT.to(dev)).all())
ref64 = torch.mm(A[:64], Bm)
rows48_64_correct = torch.equal(O[48:64], ref64[48:64])
print(f"  rows[0:48] == mm(A[:48],B) : {ok48}")
print(f"  rows[48:64] still sentinel : {tail_untouched}   (True => M really became 48)")
print(f"  rows[48:64] == mm result   : {rows48_64_correct}  (True => patch ignored, still M=64)")
del g64

print()
print("### TEST B: capture M=256 (variant _64x24_), swap func+grid+smem+blob to the")
print("###         M=128 variant (_64x8_2x4_) and replay -> does the node really change kernel?")
n128,g128 = cap_mm(128)
k128=[x for x in n128 if x["type"]=="KERNEL"][0]
f128,grid128,blk128,sm128,blob128 = kparams(k128)
print(f"  donor  M=128: {k128['name']} grid={grid128} smem={sm128}")
n256,g256 = cap_mm(256)
k256=[x for x in n256 if x["type"]=="KERNEL"][0]
f256,grid256,blk256,sm256,blob256 = kparams(k256)
print(f"  target M=256: {k256['name']} grid={grid256} smem={sm256}")
print(f"  same func? {f128==f256}   same smem? {sm128==sm256}")
r,nm = set_exec(g256, k256["node"], f128, grid128, blk128, sm128, blob128, hold)
print(f"  cuGraphExecKernelNodeSetParams_v2(func -> other variant) -> {r} {nm}")
if r==0:
    O.fill_(-999.0); torch.cuda.synchronize()
    g256.replay(); torch.cuda.synchronize()
    ref128 = torch.mm(A[:128], Bm)
    print(f"  rows[0:128] == mm(A[:128],B) : {torch.equal(O[:128], ref128)}")
    print(f"  max|err| rows[0:128]         : {(O[:128].float()-ref128.float()).abs().max().item()}")
    print(f"  rows[128:256] still sentinel : {bool((O[128:256]==SENT.to(dev)).all())}")
print("DONE")
