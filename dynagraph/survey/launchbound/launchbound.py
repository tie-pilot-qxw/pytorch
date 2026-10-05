#!/usr/bin/env python3
"""CPU-only launch-boundness model.  No GPU is touched.

Machine constants are taken from measurements already made on THIS box
(see docs/notes/FEASIBILITY.md and mega-kernel/h100-roofline/,
which is not included in this repo):
  BW_peak_copy   2871 GB/s   (TMA copy, 132 SM, h100-roofline/README.md)
  FLOPS_peak_bf16 856 TFLOPS (h100-roofline/README.md)
  eager tiny-kernel wall     2.05 us   (docs/notes/FEASIBILITY.md, "FATAL 4 is downgraded too -- but the conclusion is not what the skeptic said", grid_zero table, n=1024)
  eager 8.4 MB pointwise     6.09 us   (same table, n=1,048,576 fp32 r+w)
  eager 33.5 MB pointwise   20.43 us   (same table, n=4,194,304)
  graph serial node overhead 0.78-0.80 us/node (docs/notes/FEASIBILITY.md, "2. All measured data")
Fit of the grid_zero table:  t_eager(bytes) ~= 1.95 us + bytes / 1800 GB/s
=> pure-GPU execution part   t_exec(bytes)  ~= max(1.0 us, bytes / 1800 GB/s)
"""
import math

BW      = 1800e9      # B/s effective streaming bandwidth (this box, fitted)
FLOPS   = 400e12      # bf16, 47% of the 856 TF measured peak: realistic for bmm/linear
FLOPS_SMALL = 100e12  # for matmuls with < 1 GFLOP (wave quantisation, small K)
T_EXEC_FLOOR = 1.0e-6 # s, pure GPU execution of a trivial kernel
G_EAGER = 2.0e-6      # s, GPU-visible per-kernel gap in eager (grid_zero tiny kernel)
G_GRAPH = 0.8e-6      # s, measured graph node overhead on this box

C_HOST = {            # host CPU seconds to issue one kernel
    "inductor_cpp": 2.5e-6,   # compiled, C++ wrapper, no python per kernel
    "python_eager": 5.0e-6,   # PyGraph: "a typical CUDA kernel launch takes ~5-10 us"
    "python_heavy": 8.0e-6,   # python-heavy model code (OpenFold-style indexing)
}

def t_kernel(bytes_moved, flops=0.0):
    f = FLOPS if flops > 1e9 else FLOPS_SMALL
    return max(T_EXEC_FLOOR, bytes_moved / BW, flops / f)

def analyse(name, ops, c_host_key="python_eager", verbose=True):
    """ops: list of (count, bytes, flops, label)"""
    N = sum(c for c, _, _, _ in ops)
    Tsum = sum(c * t_kernel(b, f) for c, b, f, _ in ops)
    c_host = C_HOST[c_host_key]
    eager = max(N * c_host, Tsum + N * G_EAGER)
    graph = Tsum + N * G_GRAPH
    Tbar = Tsum / N
    host_bound = N * c_host > Tsum + N * G_EAGER
    if verbose:
        print(f"\n### {name}   [c_host={c_host*1e6:.1f} us]")
        print(f"  N kernels/step        {N}")
        print(f"  sum GPU exec          {Tsum*1e3:.3f} ms")
        print(f"  mean kernel Tbar      {Tbar*1e6:.2f} us")
        print(f"  host time N*c_host    {N*c_host*1e3:.3f} ms")
        print(f"  eager step            {eager*1e3:.3f} ms   ({'HOST/LAUNCH-BOUND' if host_bound else 'GPU-BOUND'})")
        print(f"  cudagraph step        {graph*1e3:.3f} ms")
        print(f"  ==> cudagraph speedup {eager/graph:.2f}x")
        big = sorted(ops, key=lambda o: -o[0]*t_kernel(o[1], o[2]))[:4]
        for c, b, f, lab in big:
            print(f"      {lab:34s} x{c:<5d} {t_kernel(b,f)*1e6:7.2f} us each"
                  f"  ({100*c*t_kernel(b,f)/Tsum:4.1f}% of GPU)")
    return dict(name=name, N=N, Tsum=Tsum, Tbar=Tbar, eager=eager, graph=graph,
                speedup=eager/graph, host_bound=host_bound)

print("="*78)
print("CROSSOVER: cudagraph pays only when mean kernel duration is small")
print("="*78)
for key in ("inductor_cpp", "python_eager", "python_heavy"):
    c = C_HOST[key]
    # launch-bound iff Tbar + G_EAGER < c_host
    Tstar = c - G_EAGER
    # >10% gain iff (Tbar+G_EAGER)/(Tbar+G_GRAPH) > 1.10  (gap-only benefit)
    # solve: Tbar < (G_EAGER - 1.1*G_GRAPH)/0.1
    T10 = (G_EAGER - 1.10*G_GRAPH)/0.10
    T2x = None
    print(f"  {key:14s} c_host={c*1e6:.1f}us -> launch-bound iff Tbar < {Tstar*1e6:.1f} us; "
          f"gap-only >=10% iff Tbar < {T10*1e6:.1f} us")


# ---------------------------------------------------------------- AF3 Pairformer
def af3_pairformer_block(L, cz=128, cs=384, h=4, fused_tri_attn=True, dt=2):
    """One AF3 Pairformer block.  Returns op list (count, bytes, flops, label).
    cz=128 pair channels, cs=384 single, h=4 heads x 32 dim, SwiGLU transition n=4.
    fused_tri_attn=True models trifast / cuEquivariance (no L^3 logits materialised)."""
    P  = L*L*cz*dt              # pair tensor bytes
    S  = L*cs*dt                # single tensor bytes
    ops = []
    for _ in range(2):          # TriangleMultiplication outgoing + incoming
        ops += [
            (1, 2*P,        0,                'trimul: LayerNorm'),
            (2, P + 2*P,    2*L*L*cz*2*cz,    'trimul: Linear cz->2cz (proj,gate)'),
            (2, 2*(2*P),    0,                'trimul: sigmoid gate'),
            (2, 3*P,        0,                'trimul: mul a*gate'),
            (1, 5*P,        2*cz*L**3,        'trimul: einsum ikc,jkc->ijc'),
            (1, 2*P,        0,                'trimul: LayerNorm out'),
            (2, 2*P,        2*L*L*cz*cz,      'trimul: Linear out / gate'),
            (1, 3*P,        0,                'trimul: gate mul'),
            (1, 3*P,        0,                'trimul: residual add'),
        ]
    for _ in range(2):          # TriangleAttention starting + ending node
        logits = L**3*h*dt
        if fused_tri_attn:
            attn_ops = [(1, 4*P, 4*cz*L**3, 'triattn: FUSED flash-triangle')]
        else:
            attn_ops = [
                (1, 2*P + logits, 2*cz*L**3, 'triattn: QK^T (materialised L^3)'),
                (1, 2*logits,     0,         'triattn: softmax over L^3'),
                (1, logits + 2*P, 2*cz*L**3, 'triattn: A@V'),
            ]
        ops += [
            (1, 2*P,      0,                    'triattn: LayerNorm'),
            (1, P + 3*P,  2*L*L*cz*3*cz,        'triattn: Linear qkv'),
            (1, P + L*L*h*dt, 2*L*L*cz*h,       'triattn: Linear pair bias'),
            (1, 2*P,      2*L*L*cz*cz,          'triattn: Linear gate'),
            (1, 2*P,      0,                    'triattn: sigmoid gate'),
        ] + attn_ops + [
            (1, 3*P,      0,                    'triattn: gate mul'),
            (1, 2*P,      2*L*L*cz*cz,          'triattn: Linear out'),
            (1, 3*P,      0,                    'triattn: residual add'),
        ]
    ops += [                     # PairTransition (SwiGLU n=4)
        (1, 2*P,       0,                   'pairtrans: LayerNorm'),
        (1, P + 8*P,   2*L*L*cz*8*cz,       'pairtrans: Linear up (SwiGLU)'),
        (1, 8*P + 4*P, 0,                   'pairtrans: swish*gate'),
        (1, 4*P + P,   2*L*L*4*cz*cz,       'pairtrans: Linear down'),
        (1, 3*P,       0,                   'pairtrans: residual'),
    ]
    # single track: AttentionPairBias(s) + Transition(s) -- tiny tensors, ~25 kernels
    ops += [
        (1, 2*S,       0,                  'single: LayerNorm'),
        (3, 2*S,       2*L*cs*cs,          'single: Linear qkv'),
        (1, P + L*L*16*dt, 2*L*L*cz*16,    'single: pair->bias proj'),
        (1, 2*L*L*16*dt + L*cs*dt, 2*L*L*cs, 'single: attn (L x L)'),
        (1, 2*S,       2*L*cs*cs,          'single: out proj'),
        (4, 2*S,       0,                  'single: gate/sigmoid/mul/add'),
        (1, 2*S,       0,                  'singletrans: LayerNorm'),
        (2, 4*S,       2*L*cs*4*cs,        'singletrans: Linear up/down'),
        (3, 4*S,       0,                  'singletrans: act/mul/add'),
    ]
    return ops

print("\n" + "="*78)
print("WORKLOAD 1: AlphaFold3-style Pairformer  (48 blocks, 1 trunk pass)")
print("="*78)
for L in (64, 128, 256, 384, 768, 1024, 2048):
    ops = af3_pairformer_block(L)
    ops48 = [(c*48, b, f, lab) for c, b, f, lab in ops]
    r = analyse(f"Pairformer x48, L={L} (fused tri-attn)", ops48,
                "python_eager", verbose=(L in (128, 384, 2048)))
    if L not in (128, 384, 2048):
        print(f"  L={L:5d}: N={r['N']:6d}  Tbar={r['Tbar']*1e6:7.2f}us  "
              f"GPU={r['Tsum']*1e3:8.2f}ms  eager={r['eager']*1e3:8.2f}ms  "
              f"speedup={r['speedup']:.2f}x  {'LAUNCH-BOUND' if r['host_bound'] else 'gpu-bound'}")


# -------------------------------------------------- weight the AF result by the real proteome
print("\n" + "="*78)
print("Is any real protein short enough for cudagraph to matter?")
print("UniProt human reference proteome UP000005640 (same file the project used)")
print("="*78)
import gzip, os, bisect
FA = os.path.join(os.environ.get("DG_DATA", "/workspace/_deps/data"), "human_proteome.fasta.gz")
Ls = []
cur = 0
with gzip.open(FA, "rt") as f:
    for line in f:
        if line.startswith(">"):
            if cur: Ls.append(cur)
            cur = 0
        else:
            cur += len(line.strip())
if cur: Ls.append(cur)
Ls.sort()
n = len(Ls)
print(f"  {n} sequences, p50={Ls[n//2]}, mean={sum(Ls)/n:.0f}, max={Ls[-1]}")

# cudagraph speedup as a function of L (one trunk pass), tabulated then interpolated
grid = [16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 36000]
tab = {}
for L in grid:
    ops = [(c*48, b, f, lab) for c, b, f, lab in af3_pairformer_block(L)]
    tab[L] = analyse(f"L={L}", ops, "python_eager", verbose=False)
print("\n  L        Tbar(us)   GPU(ms)   cudagraph speedup")
for L in grid:
    r = tab[L]
    print(f"  {L:5d}  {r['Tbar']*1e6:9.2f} {r['Tsum']*1e3:10.2f}   {r['speedup']:.3f}x")

def speedup_at(L):
    i = bisect.bisect_left(grid, L)
    return tab[grid[min(i, len(grid)-1)]]["speedup"]
def gputime_at(L):
    i = bisect.bisect_left(grid, L)
    return tab[grid[min(i, len(grid)-1)]]["Tsum"]

for thr in (64, 128, 160, 256):
    k = bisect.bisect_right(Ls, thr)
    work = sum(gputime_at(L) for L in Ls)
    work_below = sum(gputime_at(L) for L in Ls[:k])
    print(f"  L <= {thr:4d}:  {k:7d} seqs = {100*k/n:5.2f}% of sequences, "
          f"but only {100*work_below/work:6.3f}% of total trunk GPU work")

# what does a whole-proteome fold cost, and what would cudagraph save?
tot_eager = sum(tab[grid[min(bisect.bisect_left(grid, L), len(grid)-1)]]["eager"] for L in Ls)
tot_graph = sum(tab[grid[min(bisect.bisect_left(grid, L), len(grid)-1)]]["graph"] for L in Ls)
print(f"\n  whole-proteome trunk pass:  eager {tot_eager/3600:.2f} GPU-h, "
      f"cudagraph {tot_graph/3600:.2f} GPU-h  -> {tot_eager/tot_graph:.4f}x overall")
print("  (contrast: pad-to-max K=1 costs 2590x at p=2 / 16186x at p=3; "
      "K=8 buckets still 2.75x / 7.37x -- FINDINGS.md)")


print("\n" + "="*78)
print("SENSITIVITY: op-inflation factor phi")
print("  My inventory counts ARCHITECTURAL kernels (~3168 per trunk pass).")
print("  ScaleFold measured >150,000 operators per AlphaFold2 TRAINING step")
print("  (18,147 math-bound + 97,749 memory-bound kernels + 34,991 memory ops).")
print("  Real code launches phi x more kernels than the architecture implies")
print("  (chunking, transposes/permutes, dropout, per-head slicing, fwd+bwd+opt).")
print("  Same GPU work, phi x the kernels => Tbar / phi.  How far does that move it?")
print("="*78)
def inflate(ops, phi):
    return [(c*phi, b/phi, f/phi, lab) for c, b, f, lab in ops]
print("  phi |  L=128   L=256   L=384   L=1024  | whole-proteome")
for phi in (1, 3, 10, 30):
    row = []
    tabp = {}
    for L in grid:
        o = inflate([(c*48, b, f, lab) for c, b, f, lab in af3_pairformer_block(L)], phi)
        tabp[L] = analyse("", o, "python_eager", verbose=False)
    for L in (128, 256, 384, 1024):
        row.append(f"{tabp[L]['speedup']:.2f}x")
    te = sum(tabp[grid[min(bisect.bisect_left(grid, L), len(grid)-1)]]["eager"] for L in Ls)
    tg = sum(tabp[grid[min(bisect.bisect_left(grid, L), len(grid)-1)]]["graph"] for L in Ls)
    print(f"  {phi:3d} | " + "  ".join(f"{x:>6s}" for x in row) + f"  |  {te/tg:.3f}x")


# ------------------------------------------------------------------ MACE / MLIP
print("\n" + "="*78)
print("WORKLOAD 2: MACE-style MLIP  (2 interaction layers, 128 ch, lmax=3, corr order 3)")
print("  atom/edge counts from the project's own CPU measurement")
# results.log is not included in this repo
print("  (survey/workload_shapes/results.log, MD17/MD22 real trajectories)")
print("="*78)
LMAX = 3
paths = [(l1,l2,l3) for l1 in range(LMAX+1) for l2 in range(LMAX+1)
         for l3 in range(abs(l1-l2), min(LMAX, l1+l2)+1)]
NPATH = len(paths)
print(f"  e3nn TensorProduct instructions for lmax={LMAX} x lmax={LMAX} -> lmax={LMAX}: {NPATH}")

def mace_step(N, E, C=128, lmax=3, dt=4, nlayers=2, bwd_factor=2.2):
    ir = (lmax+1)**2                      # 16 irrep components
    NF = N*C*ir*dt                        # node feature bytes
    EF = E*C*ir*dt                        # edge feature bytes
    ops = []
    for _ in range(nlayers):
        ops += [
            (4, E*3*dt*2,    0,            'edge vectors / lengths / normalise'),
            (3, E*8*dt*2,    0,            'Bessel radial basis (8)'),
            (2, E*dt*2,      0,            'polynomial cutoff'),
            (6, E*64*dt*2,   2*E*64*64,    'radial MLP 8->64->64->n_w'),
            (8, E*ir*dt*2,   0,            'spherical harmonics l=0..3'),
            (1, 2*NF,        2*N*(C*ir)*C, 'node Linear (irrep-blocked)'),
            (1, NF + EF,     0,            'gather node feats -> edges'),
            (NPATH, 2*EF/NPATH*2, 2*E*C*ir, f'TensorProduct path (x{NPATH})'),
            (1, E*C*dt*2,    2*E*C*NPATH,  'radial weight multiply'),
            (2, EF + NF,     0,            'scatter_sum edges -> nodes'),
            (3*NPATH, 2*NF/NPATH*2, 2*N*C*ir, f'symmetric contraction (corr 3)'),
            (3, 2*NF,        2*N*(C*ir)*C, 'Linear / reshape / residual'),
        ]
    ops += [(2, 2*N*dt, 0, 'readout / energy sum')]
    # backward (forces = autograd through everything)
    ops = ops + [(int(math.ceil(c*(bwd_factor-1))), b, f, lab+' [bwd]') for c,b,f,lab in ops]
    return ops

for label, N, E, shapes in [
    ("MD17 aspirin      N=21", 21, 301, "pairs 284..318, 18 distinct shapes/3000 frames, 24% consec-change"),
    ("MD22 AcAla3       N=42", 42, 935, "pairs 784..1086, 146 distinct, 73% consec-change"),
    ("MD22 AT-AT-CG-CG N=118", 118, 2900, "pairs 2244..3562, 486 distinct, 84% consec-change"),
    ("MD22 nanotube    N=370", 370, 15400, "pairs 14460..16384, 668 distinct, 94% consec-change"),
    ("condensed phase  N=4000", 4000, 4000*45, "(not measured by the project; typical MD box)"),
]:
    r = analyse(f"MACE {label}  [{shapes}]", mace_step(N, E), "python_eager", verbose=False)
    print(f"  {label:24s} E={E:6d}  N_kern={r['N']:5d}  Tbar={r['Tbar']*1e6:7.2f}us  "
          f"GPU={r['Tsum']*1e3:7.3f}ms  eager={r['eager']*1e3:7.3f}ms  "
          f"speedup={r['speedup']:.2f}x  {'LAUNCH-BOUND' if r['host_bound'] else 'gpu-bound'}")
    print(f"      shape space: {shapes}")

# ---------------------------------------------------------- SECOND sparse conv
print("\n" + "="*78)
print("WORKLOAD 3: SECOND sparse-conv 3D backbone on KITTI")
# results.log is not included in this repo
print("  active voxel counts are the project's own measurement (results.log)")
print("="*78)
def second_backbone(v1=32291, v2=16373, v3=7086, v4=2781, dt=2, neigh=12):
    stages = [(v1,16,16,3),(v2,32,32,3),(v3,64,64,3),(v4,64,64,3)]
    ops = []
    for nnz, cin, cout, nlayer in stages:
        for _ in range(nlayer):
            feat = nnz*cout*dt
            ops += [
                (1, nnz*4*2,  0,                        f'rulebook/hash (nnz={nnz})'),
                (1, 2*feat,   0,                        'gather'),
                (1, 2*feat,   2*nnz*neigh*cin*cout,     'implicit GEMM'),
                (1, 2*feat,   0,                        'scatter-add'),
                (2, 2*feat,   0,                        'BN + ReLU'),
            ]
    # dense BEV backbone: 2D convs on [200,176] with 128/256 ch
    for ch, k in ((128,6),(256,6)):
        ops += [(k, 2*200*176*ch*dt, 2*200*176*ch*ch*9, f'BEV conv {ch}ch'),
                (2*k, 2*200*176*ch*dt, 0, f'BEV BN/ReLU {ch}ch')]
    ops += [(10, 2*120000*4, 0, 'voxelisation / scatter')]
    return ops
r = analyse("SECOND 3D+BEV backbone, KITTI mean frame", second_backbone(), "python_eager")
print("  shape space: 433/433 distinct 4-stage voxel tuples; pad-to-max = 41703/32291 = 1.29x at p=1")

# ------------------------------------------------------------------ GraphSAGE
print("\n" + "="*78)
print("WORKLOAD 4: 3-layer GraphSAGE, B=1024, ogbn-arxiv, fanout [15,10,5]")
print("  node/edge counts from the project's own sampler measurement")
print("="*78)
def sage(layers, dt=4, bwd=2.2):
    ops = []
    for nsrc, ndst, nedge, cin, cout in layers:
        ops += [
            (1, nedge*cin*dt + nsrc*cin*dt, 0,                'index_select src feats'),
            (2, 2*nedge*cin*dt,             0,                'scatter_add / segment mean'),
            (2, 2*ndst*cout*dt,             2*ndst*cin*cout,  'lin_l / lin_r GEMM'),
            (3, 2*ndst*cout*dt,             0,                'add / relu / dropout'),
        ]
    ops += [(4, 2*1024*40*dt, 0, 'loss + softmax')]
    ops = ops + [(int(math.ceil(c*(bwd-1))), b, f, lab+' [bwd]') for c,b,f,lab in ops]
    ops += [(4, 2*500000*dt, 0, 'fused Adam (foreach)')]
    return ops
# fanout [15,10,5]: measured for [15,10] -> L1 7000 edges/5700 new nodes, L2 47000 edges/26700 new
layers = [(30000, 29000, 150000, 128, 256),   # layer 0 (deepest hop)
          (29000, 15400,  47000, 256, 256),
          (15400,  1024,   7000, 256, 256)]
r = analyse("GraphSAGE 3L B=1024 (train step)", sage(layers), "python_eager")
print("  shape space: 200/200 distinct shape tuples in 200 steps; "
      "pad-to-max over 6 axes ~ 1.1x (edge counts vary 6498..7486 etc.)")


print("\n" + "="*78)
print("THE STRUCTURAL ARGUMENT: DynaGraph must beat BOTH baselines at once")
print("="*78)
print("""  Per-kernel units (divide everything by N).  Tbar = mean GPU kernel duration.
    B0  eager, exact shapes       :  max(c_host, Tbar + g_eager)
    B1  pad-to-max, ONE static graph (dynamic=False):  rho*Tbar + g_graph
    B3  DynaGraph: exact shape, one reusable graph  :  Tbar + g_graph
  DynaGraph's value = min(B0, B1) / B3.  It must beat the better of the two.
    - beating B0 needs LAUNCH-bound  (Tbar small)
    - beating B1 needs padding to cost real time, i.e. GPU-bound (Tbar large)
  Those are the same knob pointing opposite ways.  The envelope:""")
c, ge, gg = 5.0, 2.0, 0.8
print(f"\n  Tbar(us) |" + "".join(f"  rho={r:<5.2f}" for r in (1.06,1.29,2.75,7.37,100.0)))
for u in (0.3,0.5,0.8,1.0,1.5,2.0,3.0,5.0,8.0,15.0,30.0,100.0,1000.0):
    row = []
    for rho in (1.06,1.29,2.75,7.37,100.0):
        v = min(max(c, u+ge), rho*u+gg)/(u+gg)
        row.append(f"   {v:5.2f}x")
    print(f"  {u:8.1f} |" + "".join(row))
print("\n  best achievable value of DynaGraph over min(B0,B1), maximised over Tbar:")
for rho in (1.06,1.29,2.75,7.37,100.0):
    best = max(((min(max(c,u+ge), rho*u+gg)/(u+gg)), u)
               for u in [x/100 for x in range(5, 200000)])
    print(f"    rho={rho:7.2f}  ->  {best[0]:.2f}x   at Tbar={best[1]:.2f} us")

print("\n  where the surveyed workloads actually sit:")
rows = [
 ("MACE N=21   (MD17 aspirin)",   1.01, 318/301),
 ("MACE N=42   (AcAla3)",         1.07, 1086/935),
 ("MACE N=118  (AT-AT-CG-CG)",    1.39, 3562/2900),
 ("MACE N=370  (nanotube)",       4.20, 16384/15400),
 ("SECOND sparse conv, KITTI",   10.68, 41703/32291),
 ("GraphSAGE 3L B=1024",         23.21, 1.10),
 ("AF3 Pairformer L=64",          1.81, 100.0),
 ("AF3 Pairformer L=128",         5.86, 100.0),
 ("AF3 Pairformer L=384",        50.05, 100.0),
 ("AF3 Pairformer L=2048",     1751.89, 100.0),
]
print(f"  {'workload':32s} {'Tbar(us)':>9s} {'rho':>7s} {'vs eager':>9s} {'vs pad-max':>11s} {'MIN':>7s}")
for name, u, rho in rows:
    v0 = max(c, u+ge)/(u+gg)
    v1 = (rho*u+gg)/(u+gg)
    print(f"  {name:32s} {u:9.2f} {rho:7.2f} {v0:8.2f}x {v1:10.2f}x {min(v0,v1):6.2f}x")


print("\n" + "="*78)
print("THE DECISIVE NUMBER: fold the whole human proteome, three ways")
print("  (same kernel-time model everywhere, so padding a short seq to its bucket")
print("   correctly makes that run *less* launch-bound, not more)")
print("="*78)
def af_time(L, mode):
    i = bisect.bisect_left(grid, L)
    return tab[grid[min(i, len(grid)-1)]][mode]
AF3_BUCKETS = [256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120]
def bucket_of(L, bks):
    i = bisect.bisect_left(bks, L)
    return bks[i] if i < len(bks) else L          # above top bucket: no padding possible
for label, bks in [("AF3 default 13 buckets", AF3_BUCKETS),
                   ("8 work-weighted buckets", [256,512,1024,1536,2048,3072,4608,Ls[-1]]),
                   ("K=1 pad-to-max", [Ls[-1]])]:
    B0 = sum(af_time(L, "eager") for L in Ls)              # eager, exact shapes
    B1 = sum(af_time(bucket_of(L, bks), "graph") for L in Ls)  # bucketed static graphs
    B3 = sum(af_time(L, "graph") for L in Ls)              # DynaGraph
    print(f"  {label:24s}  eager {B0/3600:7.2f} GPU-h | "
          f"static-graph+pad {B1/3600:9.2f} GPU-h | DynaGraph {B3/3600:7.2f} GPU-h")
    print(f"  {'':24s}  DynaGraph vs eager {B0/B3:.3f}x, vs this padding scheme {B1/B3:.3f}x"
          f"  ->  headroom over the better baseline = {min(B0,B1)/B3:.3f}x")


print("\n" + "="*78)
print("SENSITIVITY on c_host (the one parameter I could not measure end-to-end)")
print("  local CPU-only measurement of dispatch+autograd+alloc alone: 2.0-2.4 us/op")
print("  PyGraph paper: 'a typical CUDA kernel launch takes ~5-10 microseconds'")
print("  PyGraph DALLE2 bs=1: 740 kernels, 3.4ms GPU, 14ms e2e on H100")
print("    => implied host cost 14ms/740 = 18.9 us per kernel (compiled, python)")
print("="*78)
print(f"  {'c_host':>7s} | {'MACE N=118':>11s} {'KITTI':>8s} {'SAGE':>7s} "
      f"{'AF3 L=256':>10s} {'AF3 L=1024':>11s} | {'proteome-wide':>13s}")
for ch in (2.5, 5.0, 10.0, 19.0):
    C_HOST["x"] = ch*1e-6
    out = []
    for nm, ops in [("MACE118", mace_step(118, 2900)),
                    ("KITTI", second_backbone()),
                    ("SAGE", sage(layers))]:
        out.append(f"{analyse('', ops, 'x', verbose=False)['speedup']:.2f}x")
    tabx = {}
    for L in grid:
        o = [(c*48, b, f, lab) for c, b, f, lab in af3_pairformer_block(L)]
        tabx[L] = analyse("", o, "x", verbose=False)
    out.append(f"{tabx[256]['speedup']:.2f}x")
    out.append(f"{tabx[1024]['speedup']:.2f}x")
    B0 = sum(tabx[grid[min(bisect.bisect_left(grid, L), len(grid)-1)]]["eager"] for L in Ls)
    B3 = sum(tabx[grid[min(bisect.bisect_left(grid, L), len(grid)-1)]]["graph"] for L in Ls)
    print(f"  {ch:6.1f}us | {out[0]:>11s} {out[1]:>8s} {out[2]:>7s} "
          f"{out[3]:>10s} {out[4]:>11s} | {B0/B3:12.3f}x")
