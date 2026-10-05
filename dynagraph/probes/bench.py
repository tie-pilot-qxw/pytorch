#!/usr/bin/env python3
"""Four-way comparison: re-record / eager / pad2max / DynaGraph, through the public switches, on a real shape stream.

    CUDA_VISIBLE_DEVICES=6 python bench.py --regime launch --steps 256

**The card must be exclusive, and you must watch the CPU load.** The launch-bound regime measures **CPU launch overhead**,
and the CPU is shared by the whole machine -- other people's jobs, even on other cards, compete for CPU through their host processes.
We have measured the same unmodified baseline differing by 7x between two runs (A re-record 47 ms vs 331 ms),
so a single wall-clock number cannot be trusted.

But **high load itself is not contamination, it is the real operating condition** -- in training, data loading and multiple processes
already saturate the CPU, and the whole point of CUDA Graphs is to cut CPU launch overhead. High load is exactly the scenario where
it should shine, and should not be excluded.

The real problem is that **the load drifts**: if one config runs for ten minutes before the next one starts, the machine is in a different state.
So the remedy is **interleaving**: the compiled artifacts of all four configs stay alive at the same time, the steady-state regime runs them in turn,
each config is repeated N times and we take the minimum. The minimum is robust to interference (interference only makes a run slower),
and the spread is reported alongside. The load average is recorded in the results because it is part of the operating condition.

To keep all four alive at once we cannot call `torch._dynamo.reset()` midway, so each config uses its **own
model class** -- Dynamo caches by code object, and different instances of the same class share compiled artifacts,
so A and D would land in the same cudagraph region and share fn_cache, making the comparison meaningless.

The shape stream comes from **real data**: the length distribution of the 147520 sequences in the UniProt human reference proteome
(p50 349, p90 990), not made-up numbers. Take the [32, MAXLEN] range and sample by frequency of occurrence.

Four configs:
  A re-record  torch.compile(dynamic=True, mode="reduce-overhead") -- the upstream status quo,
               records one graph per distinct shape
  B eager      not compiled
  C pad2max    compiled + every input padded to the longest length in the stream -- one graph, but does wasted work
  D DynaGraph  switch turned on

The criteria include two positive controls; without either one this benchmark cannot be trusted:
  * **D must actually be served** (0 recordings). If it fell back, we are timing A and reporting it as D's result.
  * **D's numerics must match A's.** Fast but wrong is meaningless.

Two numbers are reported because they answer different questions:
  * **cold**: run the whole stream from scratch, including A's recording cost. This is the real cost --
    under a long-tail distribution new shapes keep showing up.
  * **warm**: run the same stream again. A's graphs are all recorded by then; this regime is the most favorable to A.

D is also run in two orders:
  * **max-first**: the largest comes first. Under the fixed layout (dynagraph_layout=fixed) the arena slots and the
    input store are sized by the first shape, so this avoids a rebuild midway.
  * **natural order**: whatever order the random draw gives. Under the dynamic layout (default) the arena and the input store
    grow on demand without a rebuild; this regime measures the cost of growing and (when there are externs) of re-harvesting.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import subprocess
import sys
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

HERE = os.path.dirname(os.path.abspath(__file__))
FASTA = os.path.join(os.environ.get("DG_DATA", "/workspace/_deps/data"),
                     "human_proteome.fasta.gz")


def real_lengths(lo: int, hi: int) -> list[int]:
    """Real sequence lengths, clipped to [lo, hi]."""
    out, cur = [], 0
    with gzip.open(FASTA, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if lo <= cur <= hi:
                    out.append(cur)
                cur = 0
            else:
                cur += len(line.strip())
    if lo <= cur <= hi:
        out.append(cur)
    return out


def gpu_is_idle() -> tuple[bool, str]:
    """Only exclusive use counts: if other processes are running, wall-clock is meaningless."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not vis:
        return False, "CUDA_VISIBLE_DEVICES is not set, cannot tell which card is being measured"
    try:
        q = subprocess.run(
            ["nvidia-smi", "-i", vis,
             "--query-gpu=memory.used,utilization.gpu,clocks.sm,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        return False, f"nvidia-smi unavailable: {e}"
    if q.returncode != 0:
        return False, f"nvidia-smi failed: {q.stderr.strip()[:120]}"
    mem, util, clk, pw = (x.strip() for x in q.stdout.strip().split(","))
    busy = float(mem) > 1024 or float(util) > 5
    return (not busy), f"memory {mem} MiB, utilization {util}%, SM {clk} MHz, power {pw} W"


def build(regime: str, d_model: int, tag: str = "x"):
    import torch

    class Block(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.a = torch.nn.Linear(d, d)
            self.b = torch.nn.Linear(d, d)

        def forward(self, x):
            h = torch.relu(self.a(x))
            h = h - h.mean(dim=-1, keepdim=True)
            h = self.b(h) * torch.sigmoid(h)
            return x + h

    class M(torch.nn.Module):
        def __init__(self, d, n):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(d) for _ in range(n)])

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x.sum(dim=-1)

    # launch-bound: many layers, each small, time goes into launches; gpu-bound: few layers, each large.
    n_blocks = 12 if regime == "launch" else 3
    torch.manual_seed(0)
    # One separate subclass per config: Dynamo caches by code object, and a shared class would make the four
    # configs share compiled artifacts and the cudagraph region, so D would just pick up the graphs A recorded.
    cls = type(f"M_{tag}", (M,), {})
    return cls(d_model, n_blocks).cuda().eval()


def make_input(L, d_model, pad_to=None):
    """The input is determined by the length alone.

    The two streams have different orders; the same length must get the same input for cross-order numeric comparison to hold.
    With the global RNG, whichever step runs first would shift the whole sequence after it.
    """
    import torch

    g = torch.Generator(device="cuda")
    g.manual_seed(L)
    n = pad_to or L
    x = torch.zeros(n, d_model, device="cuda")
    x[:L] = torch.randn(L, d_model, device="cuda", generator=g)
    return x


def best_of(f, stream, d_model, pad_to, reps):
    """Run the same stream repeatedly, return (min seconds, median, max).

    Take the minimum rather than the mean: interference only makes a run slower, never faster, so the minimum is closest to
    "the real cost without interference". The spread is reported too; if it is too wide, this batch of numbers is unusable.
    """
    ts = [timed_stream(f, stream, d_model, pad_to)[0] for _ in range(reps)]
    ts.sort()
    return ts[0], ts[len(ts) // 2], ts[-1]


def timed_stream(f, stream, d_model, pad_to=None, collect=False):
    """Run one shape stream, return (seconds, optional outputs)."""
    import torch

    outs = [] if collect else None
    # Build the inputs up front: input construction time is the same for all four configs, but counting it dilutes the difference,
    # and the launch-bound regime is exactly about measuring that small difference.
    xs = [make_input(L, d_model, pad_to) for L in stream]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for L, x in zip(stream, xs):
        with torch.no_grad():
            r = f(x)
        if collect:
            outs.append(r[:L].float().cpu().clone())
    torch.cuda.synchronize()
    return time.perf_counter() - t0, outs


def prepare(name, tag, regime, d_model, stream, dynagraph, compiled, pad_to,
            gemm_backends="TRITON", share_autotune=False):
    """Compile + one cold run, return a handle that can be timed repeatedly.

    Does not call torch._dynamo.reset(): the four configs must be alive at the same time to interleave.
    """
    import torch
    import torch._inductor.config as ic
    import torch._inductor.cudagraph_trees as ct

    # With caches on, every row past the first reuses the first row's autotune
    # choices, so all rows run the same GEMM kernels: under host load the
    # autotuner picks differently per compile, which shows up as a fake
    # difference between rows (see docs/METHODOLOGY.md).
    ic.force_disable_caches = not share_autotune
    # `dynagraph` is False, or the update mode ("device" / "host") to build with.
    ic.triton.dynagraph = bool(dynagraph)
    if dynagraph:
        ic.triton.dynagraph_update = dynagraph
    ic.max_autotune_gemm = True
    # DynaGraph rejects extern_kernels, so for it to apply, GEMMs must be routed to Triton.
    # All four configs use the same backend so that A vs D is fair; what this constraint itself costs
    # is measured separately with --gemm-backends ATEN,TRITON (in that regime D falls back with extern-launch).
    ic.max_autotune_gemm_backends = gemm_backends

    import logging

    msgs: list[str] = []

    class Grab(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    grab = Grab()
    lg = logging.getLogger("torch._inductor.dynagraph")
    lg.setLevel(logging.INFO)
    lg.addHandler(grab)

    n_rec = {"v": 0}
    orig = ct.CUDAGraphTreeManager.record_function

    def spy(self, *a, **kw):
        n_rec["v"] += 1
        return orig(self, *a, **kw)

    ct.CUDAGraphTreeManager.record_function = spy
    try:
        m = build(regime, d_model, tag)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        f = torch.compile(m, dynamic=True, mode="reduce-overhead") if compiled else m
        # On the first call of each shape cudagraph_trees only does an eager warmup; it records on the second.
        # The cold regime wants exactly this full cost, so the stream runs twice as-is: first pass cold, second warm.
        # First call **this config's own first shape** twice to get compilation out of the way.
        # Compilation is the same fixed cost for A/C/D (`dynamic=True` compiles once); leaving it in the timing
        # would completely hide the difference between A and D -- compilation measured ~10 s, while the first-pass re-record is only tens of ms.
        # Warm up with the first shape rather than the largest shape: that is the real first-call behavior,
        # and warming up with the largest shape would mask DynaGraph's input-headroom problem.
        timed_stream(f, stream[:1], d_model, pad_to)
        timed_stream(f, stream[:1], d_model, pad_to)
        n_rec["v"] = 0
        cold, _ = timed_stream(f, stream, d_model, pad_to)
        rec_cold = n_rec["v"]
        # Memory is the axis **unaffected by CPU load**, and the most solid gain here:
        # re-recording keeps a graph memory pool per distinct shape, which cannot hold up under a real long-tail distribution;
        # DynaGraph is always one graph + one arena.
        peak = torch.cuda.max_memory_reserved()
    finally:
        ct.CUDAGraphTreeManager.record_function = orig
        lg.removeHandler(grab)
    tags = sorted({t.split("[", 1)[1].split("]", 1)[0]
                   for t in msgs if t.startswith("DynaGraph fallback [")})
    detail = [t[:200] for t in msgs if t.startswith("DynaGraph fallback [")][:3]
    return {"name": name, "fn": f, "model": m, "dynagraph": dynagraph,
            "pad_to": pad_to, "stream": stream, "cold": cold,
            "peak_gib": peak / 2**30, "tags": tags, "detail": detail,
            "rec_cold": rec_cold, "times": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("launch", "gpu"), default="launch")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--distinct", type=int, default=64,
                    help="number of distinct shapes in the stream; upstream records one graph for each")
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--dmodel", type=int, default=0)
    ap.add_argument("--headroom", type=float, default=0.0,
                    help="override dynagraph_headroom; 0 means leave it unchanged")
    ap.add_argument("--out", default="")
    ap.add_argument("--reps", type=int, default=5,
                    help="how many warm repetitions to take the minimum over")
    ap.add_argument("--gemm-backends", default="TRITON",
                    help="max_autotune_gemm_backends; ATEN,TRITON routes GEMMs to "
                         "cuBLAS, and D then falls back with extern-launch -- this regime measures "
                         "the cost of the \"Triton only\" constraint itself")
    ap.add_argument("--partition-extern", action="store_true",
                    help="enable dynagraph_partition_extern: split at extern kernels, "
                         "so D can be served even when GEMMs go to cuBLAS (at the cost of a fragmented graph)")
    ap.add_argument("--extern-child", action="store_true",
                    help="enable dynagraph_extern_child: extern calls stay in the graph as child-graph "
                         "nodes, harvested lazily per shape and swapped in, without splitting")
    ap.add_argument("--allow-shared", action="store_true",
                    help="run even if the card is not clean (the times are reference only and cannot go into conclusions)")
    ap.add_argument("--rows", default="",
                    help="run only these rows (by tag, e.g. A,D,F); empty = all")
    ap.add_argument("--share-autotune", action="store_true",
                    help="do not disable caches: all configs reuse the same autotune choices, GEMM kernels match, numerics are bitwise comparable")
    a = ap.parse_args()

    ok, why = gpu_is_idle()
    la = os.getloadavg()
    print(f"  card state: {why}")
    print(f"  host load average: {la[0]:.1f} / {la[1]:.1f} / {la[2]:.1f}"
          f" ({os.cpu_count()} CPU cores) -- the launch-bound regime measures CPU launch overhead, "
          f"do not trust single-run numbers when load is high")
    if not ok and not a.allow_shared:
        print("  FAIL card is not exclusive, refusing to time. Add --allow-shared to force it, "
              "but those numbers cannot go into conclusions.")
        return 1

    import random

    import torch
    import torch._inductor.config as ic

    if a.headroom:
        ic.triton.dynagraph_headroom = a.headroom
    ic.triton.dynagraph_partition_extern = a.partition_extern
    ic.triton.dynagraph_extern_child = a.extern_child

    d_model = a.dmodel or (128 if a.regime == "launch" else 1024)

    lens = real_lengths(32, a.maxlen)
    print(f"  real sequences: {len(lens)} (length 32..{a.maxlen}), "
          f"median {int(statistics.median(lens))}")
    rng = random.Random(0)
    # Sample by real frequency until we have enough. Do not use sorted(...)[:n] -- that takes the smallest n,
    # which cuts off the whole long tail and badly underestimates the cost of pad2max.
    pool: list[int] = []
    seen = set()
    while len(pool) < a.distinct and len(seen) < len(set(lens)):
        v = rng.choice(lens)
        if v not in seen:
            seen.add(v)
            pool.append(v)
    stream = [rng.choice(pool) for _ in range(a.steps)]
    mx = max(stream)
    # DynaGraph's static input buffers are allocated from the first shape, so prepare another stream with the max first.
    stream_maxfirst = [mx] + [s for s in stream if s != mx]
    waste = sum(mx for _ in stream) / sum(stream)
    print(f"  shape stream: {a.steps} steps, {len(set(stream))} distinct lengths, "
          f"longest {mx}, pad2max wasted work {waste:.2f}x")
    print(f"  regime={a.regime}  d_model={d_model}  "
          f"gemm_backends={a.gemm_backends}  partition_extern={a.partition_extern}"
          f"  extern_child={a.extern_child}\n")

    # ---- First compile all four configs and cold-run each once; only then start interleaved timing ----
    specs = (
        ("A re-record", "A", False, True, None, stream),
        ("B eager", "B", False, False, None, stream),
        ("C pad2max", "C", False, True, mx, stream),
        ("D DynaGraph(max-first)", "D", "device", True, None, stream_maxfirst),
        ("D DynaGraph host(max-first)", "F", "host", True, None, stream_maxfirst),
        ("D DynaGraph(natural)", "E", "device", True, None, stream),
    )
    rows = []
    if a.rows:
        keep = set(a.rows.split(","))
        specs = tuple(sp for sp in specs if sp[1] in keep)
    for name, tag, dg, comp, pad, stm in specs:
        r = prepare(name, tag, a.regime, d_model, stm, dg, comp, pad,
                    a.gemm_backends, a.share_autotune)
        rows.append(r)
        print(f"  {r['name']:<22} first pass (incl. re-record) {r['cold'] * 1e3:8.1f} ms   "
              f"memory {r['peak_gib']:.2f} GiB   first-pass recordings {r['rec_cold']}"
              f"   {r['tags'] or ''}")

    # ---- Interleave: the four configs take turns, so however the load drifts, it drifts on everyone ----
    print(f"\n  interleaved timing, {a.reps} rounds...")
    for _ in range(a.reps):
        for r in rows:
            ic.triton.dynagraph = bool(r["dynagraph"])
            t, _ = timed_stream(r["fn"], r["stream"], d_model, r["pad_to"])
            r["times"].append(t)
    for r in rows:
        ts = sorted(r["times"])
        r["warm"], r["warm_med"], r["warm_max"] = ts[0], ts[len(ts) // 2], ts[-1]

    # ---- Numeric cross-check: D(max-first) must match A on the same shape ----
    ref_row = next((r for r in rows if r["name"] == "A re-record"), rows[0])
    ic.triton.dynagraph = bool(ref_row["dynagraph"])
    _, ref = timed_stream(ref_row["fn"], stream, d_model, None, collect=True)
    by_len = {}
    for L, o in zip(stream, ref):
        by_len.setdefault(L, o)
    for r in rows:
        if not r["name"].endswith("(max-first)"):
            continue
        ic.triton.dynagraph = bool(r["dynagraph"])
        _, got = timed_stream(r["fn"], r["stream"], d_model, None, collect=True)
        worst = 0.0
        for L, o in zip(r["stream"], got):
            b = by_len.get(L)
            if b is not None and b.shape == o.shape:
                worst = max(worst, (b - o).abs().max().item()
                            / max(b.abs().max().item(), 1e-9))
        r["rel_vs_A"] = worst

    print()
    for r in rows:
        spread = r["warm_max"] / max(r["warm"], 1e-9)
        print(f"  {r['name']:<22} steady min {r['warm'] * 1e3:7.1f} "
              f"median {r['warm_med'] * 1e3:7.1f} max {r['warm_max'] * 1e3:7.1f} ms "
              f"(x{spread:.1f})"
              + (f"   vs A {r['rel_vs_A']:.1e}" if "rel_vs_A" in r else ""))

    # The memory axis is independent of CPU load, so it can be reported even when the times cannot be trusted.
    a_row = next((r for r in rows if r["name"] == "A re-record"), None)
    d_row = next((r for r in rows if r["name"].endswith("(max-first)")), None)
    if a_row and d_row and d_row["rec_cold"] == 0:
        print(f"\n  graph count: re-record {a_row['rec_cold']} vs DynaGraph "
              f"{d_row['rec_cold']} ({len(set(stream))} distinct shapes)")
        # The ratio can go either way, so spell out the direction: for a small model the headroom of the arena and
        # the input buffers is a fixed cost that outweighs what is saved by "recording N fewer graphs".
        ratio = a_row["peak_gib"] / max(d_row["peak_gib"], 1e-9)
        who = "DynaGraph saves" if ratio > 1 else "DynaGraph actually uses more"
        print(f"  peak memory: re-record {a_row['peak_gib']:.2f} GiB vs DynaGraph "
              f"{d_row['peak_gib']:.2f} GiB -- {who} "
              f"{max(ratio, 1 / ratio):.2f}x")
        print("  (this one is unaffected by CPU load)")

    # ---- Positive control: D must actually be served, otherwise we are timing A ----
    print()
    bad = 0
    for r in rows:
        if not r["name"].startswith("D "):
            continue
        served = r["rec_cold"] == 0
        mark = "ok served" if served else f"FAIL fell back (recorded {r['rec_cold']} times)"
        print(f"  {r['name']}: {mark}  tags {r['tags'] or '(none)'}")
        for d_ in (r["detail"] if not served else []):
            print(f"      {d_}")
        if r["name"].endswith("(max-first)"):
            bad += not served
            if "rel_vs_A" in r:
                good = r["rel_vs_A"] < 1e-3
                print(f"    numerics vs A: {r['rel_vs_A']:.1e} {'ok' if good else 'FAIL'}")
                bad += not good

    base = {r["name"]: r for r in rows}
    d = base.get("D DynaGraph(max-first)")
    if d and d["rec_cold"] == 0:
        others = [base[k] for k in ("A re-record", "B eager", "C pad2max") if k in base]
        # First pass = the cost of each new shape's first appearance. Compilation was done in the warmup, so here
        # A pays for re-recording and D pays zero -- this is the part DynaGraph actually eliminates.
        print(f"\n  first pass (compilation already warmed up; what remains is the re-record cost):")
        print(f"       re-record {base['A re-record']['cold'] * 1e3:.1f} ms "
              f"({base['A re-record']['rec_cold']} graphs recorded)"
              f" vs DynaGraph {d['cold'] * 1e3:.1f} ms (0 graphs)"
              f" -- {base['A re-record']['cold'] / max(d['cold'], 1e-9):.2f}x")
        best_warm = min(o["warm"] for o in others)
        worst_spread = max(r["warm_max"] / max(r["warm"], 1e-9) for r in rows)
        print(f"  steady state (interleaved, min over {a.reps} rounds): DynaGraph "
              f"{d['warm'] * 1e3:.1f} ms, best of A/B/C {best_warm * 1e3:.1f} ms "
              f"-- {best_warm / d['warm']:.2f}x")
        print("       steady state is not where DynaGraph is expected to win: both sides replay one graph, "
              "and it additionally pays for input copies and layout lookups.")
        if worst_spread > 1.3:
            print(f"  WARN one config's max/min spread reaches {worst_spread:.1f}x, "
                  f"the machine has interference, treat this batch of numbers as reference only")

    print(f"\n  card state after the run: {gpu_is_idle()[1]}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"regime": a.regime, "d_model": d_model, "steps": a.steps,
                       "loadavg": la, "reps": a.reps,
                       "distinct": len(set(stream)), "maxlen": mx,
                       "pad_waste": waste,
                       # fn/model are torch objects that json cannot serialize; keep only the numbers.
                       "rows": [{k: v for k, v in r.items()
                                 if k not in ("fn", "model")} for r in rows]}, fh,
                      ensure_ascii=False, indent=2)
        print(f"  results written to {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
