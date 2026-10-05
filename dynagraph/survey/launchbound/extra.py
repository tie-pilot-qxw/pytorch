import sys, os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import launchbound as LB
print("\n=== 3D sparse-conv sub-backbone only (no dense BEV head) ===")
def sc3d(dt=2, neigh=12, mode="implicit"):
    stages=[(32291,16,16,3),(16373,32,32,3),(7086,64,64,3),(2781,64,64,3)]
    ops=[]
    for nnz,cin,cout,nl in stages:
        for _ in range(nl):
            feat=nnz*cout*dt
            if mode=="implicit":
                ops += [(1,nnz*4*2,0,'rulebook'),(1,2*feat,0,'gather'),
                        (1,2*feat,2*nnz*neigh*cin*cout,'implicit GEMM'),
                        (1,2*feat,0,'scatter'),(2,2*feat,0,'BN+ReLU')]
            else:  # legacy gather-mm-scatter, one small GEMM per 3x3x3 offset
                ops += [(1,nnz*4*2,0,'rulebook'),
                        (27,2*feat/9,2*nnz*cin*cout/2,'per-offset GEMM'),
                        (27,2*feat/9,0,'per-offset gather'),
                        (27,2*feat/9,0,'per-offset scatter'),(2,2*feat,0,'BN+ReLU')]
    return ops
for mode in ("implicit","legacy"):
    for ch in (5.0, 19.0):
        LB.C_HOST["x"]=ch*1e-6
        r=LB.analyse("",sc3d(mode=mode),"x",verbose=False)
        rho=41703/32291
        v1=(rho*r['Tbar']*1e6+0.8)/(r['Tbar']*1e6+0.8)
        print(f"  spconv {mode:8s} c_host={ch:4.1f}us  N={r['N']:4d}  Tbar={r['Tbar']*1e6:6.2f}us  "
              f"GPU={r['Tsum']*1e3:6.3f}ms  cudagraph={r['speedup']:.2f}x  "
              f"vs pad-to-max(1.29x)={v1:.2f}x  headroom=min={min(r['speedup'],v1):.2f}x")

print("\n=== MACE: DynaGraph headroom over pad-to-max static graph (independent of c_host) ===")
for lab,N,E,Emax in [("N=21 aspirin",21,301,318),("N=42 AcAla3",42,935,1086),
                     ("N=118 AT-AT",118,2900,3562),("N=370 nanotube",370,15400,16384)]:
    for ch in (5.0,19.0):
        LB.C_HOST["x"]=ch*1e-6
        r=LB.analyse("",LB.mace_step(N,E),"x",verbose=False)
        rho=Emax/E; u=r['Tbar']*1e6
        v1=(rho*u+0.8)/(u+0.8)
        if ch==5.0: print(f"  {lab:16s} Tbar={u:5.2f}us rho={rho:.3f}", end="")
        print(f" | cudagraph@{ch:.0f}us={r['speedup']:5.2f}x", end="")
    print(f" | vs pad-to-max={v1:.3f}x  -> HEADROOM {min(r['speedup'],v1):.3f}x")
