# Q2: is the per-replay philox offset increment frozen at CAPTURE-TIME shape?
import torch, struct
torch.manual_seed(0)

def off():
    st = torch.cuda.get_rng_state()          # 16 bytes: seed(8) + offset(8)
    return struct.unpack('<q', bytes(st[8:16].tolist()))[0]

def capture(n):
    x = torch.randn(n, device='cuda')
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): torch.nn.functional.dropout(x, 0.5, True)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = torch.nn.functional.dropout(x, 0.5, True)
    return g, x, y

print(f"{'capture n':>12} {'offset advance / replay':>24}")
incs = {}
for n in [1024, 4096, 1<<16, 1<<20, 1<<22]:
    g, x, y = capture(n)
    torch.cuda.synchronize(); a = off()
    g.replay(); torch.cuda.synchronize(); b = off()
    g.replay(); torch.cuda.synchronize(); c = off()
    incs[n] = (b-a, c-b)
    print(f"{n:>12} {b-a:>12} , {c-b:>10}")
    del g, x, y
torch.cuda.empty_cache()

print("\n-> increment scales with capture-time n?",
      "YES (frozen at capture shape)" if len(set(v[0] for v in incs.values()))>1 else "NO (constant)")

# do two replays actually produce DIFFERENT masks? (graph-safe RNG sanity)
g, x, y = capture(1<<16)
g.replay(); torch.cuda.synchronize(); m1 = (y!=0).clone()
g.replay(); torch.cuda.synchronize(); m2 = (y!=0).clone()
print("two replays give different dropout masks:", not torch.equal(m1, m2),
      f"(overlap {100*(m1==m2).float().mean().item():.1f}%, chance=50%)")
del g,x,y,m1,m2; torch.cuda.empty_cache()
