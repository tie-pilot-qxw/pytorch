#!/usr/bin/env python3
"""(B) Diff the parameter struct of the same kernel across different M."""
import json, struct, sys, os, collections

d = json.load(open(os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "_wf_cublas_sweep.json")))

# group by (op,dtype,func)
groups = collections.defaultdict(list)
for key, recs in d.items():
    op, dt, m = key.split("|")
    M = int(m[1:])
    for i, r in enumerate(recs):
        groups[(op, dt, r["func"], r["name"][:60], i)].append((M, r))

def diff_bytes(a, b):
    """Return [(off, len, a_bytes, b_bytes)] for each contiguous differing run"""
    segs = []
    n = min(len(a), len(b))
    i = 0
    while i < n:
        if a[i] != b[i]:
            j = i
            while j < n and a[j] != b[j]:
                j += 1
            segs.append((i, j - i, a[i:j], b[i:j]))
            i = j
        else:
            i += 1
    return segs

def interp(off, ba, bb):
    outs = []
    for L, fmt, nm in ((4, "<i", "i32"), (4, "<I", "u32"), (8, "<q", "i64"), (4, "<f", "f32")):
        if len(ba) >= L and len(bb) >= L:
            try:
                va = struct.unpack(fmt, ba[:L])[0]
                vb = struct.unpack(fmt, bb[:L])[0]
                outs.append(f"{nm}:{va}->{vb}")
            except Exception:
                pass
    return "  ".join(outs)

for gk, items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    op, dt, func, name, nodeidx = gk
    if len(items) < 2:
        continue
    items.sort()
    print(f"\n### {op} {dt} node#{nodeidx} func=0x{func:x}  {name}")
    print(f"    seen at M = {[m for m,_ in items]}")
    base_M, base = items[0]
    bb = bytes.fromhex(base["params_hex"][0])
    for M, r in items[1:]:
        cb = bytes.fromhex(r["params_hex"][0])
        segs = diff_bytes(bb, cb)
        tot = sum(s[1] for s in segs)
        print(f"  M={base_M} vs M={M}: struct {len(bb)}B, {len(segs)} differing runs, {tot}B total  "
              f"grid {base['grid']}->{r['grid']}")
        for off, ln, a, b in segs:
            print(f"      +{off:<5} {ln:<4}B  {a.hex()} -> {b.hex()}   {interp(off,a,b)}")
