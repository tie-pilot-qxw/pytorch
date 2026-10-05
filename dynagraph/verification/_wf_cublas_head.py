#!/usr/bin/env python3
"""Decode the first 64 bytes of the cutlass Params struct as int32 to confirm offset0=M, offset12=mtiles."""
import json, os, struct
# produced by _wf_cublas_sweep.py
d = json.load(open(os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "_wf_cublas_sweep.json")))
for prefix in ("addmm|fp32", "mm|fp32"):
    print(f"\n===== {prefix}  (cutlass Params first 64B as int32) =====")
    print("        " + " ".join(f"@{o:<10}" for o in range(0, 64, 4)))
    for k, recs in d.items():
        if not k.startswith(prefix):
            continue
        r = recs[0]
        if "cutlass" not in r["name"]:
            continue
        b = bytes.fromhex(r["params_hex"][0])
        ints = struct.unpack_from("<16i", b, 0)
        M = int(k.split("M")[-1])
        print(f"M={M:<6}" + " ".join(f"{v:<11}" for v in ints) + f"  grid={r['grid']}")
