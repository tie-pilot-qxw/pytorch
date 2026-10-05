#!/usr/bin/env python3
"""Timing for the cond cases; needs an exclusive card.

Rules (see docs/METHODOLOGY.md): interleave the configs instead of finishing one before the next;
count the processes on the card before starting and after finishing, and discard the batch if anyone joined midway.
"""
import argparse, os, subprocess, statistics, sys, time

p = argparse.ArgumentParser()
p.add_argument("--cases", default="cond_pointwise,cond_two_outputs,cond_uneven,cond_then_item")
p.add_argument("--rounds", type=int, default=3, help="number of interleaved rounds over the configs")
p.add_argument("--card", type=int, required=True)
a = p.parse_args()


def procs_on(card):
    uu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                        capture_output=True, text=True).stdout
    uid = next((l.split(", ")[1].strip() for l in uu.splitlines()
                if l.split(",")[0].strip() == str(card)), None)
    ap = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                        capture_output=True, text=True).stdout
    return sum(1 for l in ap.splitlines() if uid and uid in l)


before = procs_on(a.card)
if before:
    print(f"  card {a.card} already has {before} processes, not exclusive, not measuring")
    sys.exit(1)

env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.card))
rows = {}
for r in range(a.rounds):
    for case in a.cases.split(","):
        out = subprocess.run(
            [sys.executable, "-u", "probe_unbacked_device.py", "--time", "--cases", case],
            capture_output=True, text=True, env=env, timeout=3600,
        ).stdout
        line = next(
            (l for l in out.splitlines() if l.strip().startswith(case) and "DG=" in l), ""
        )
        rows.setdefault(case, []).append(line)
        print(f"  round {r}  {line.strip()[:200]}", flush=True)

after = procs_on(a.card)
print(f"  at finish card {a.card} has {after} processes" + (" (someone joined midway, batch discarded)" if after else " (exclusive throughout)"))
sys.exit(0 if after == 0 else 1)
