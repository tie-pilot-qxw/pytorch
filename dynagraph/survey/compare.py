#!/usr/bin/env python3
"""
Compare the two result sets side by side; this is the core output of the survey.

  python compare.py results_default.jsonl results_capture.jsonl

Left is the stock PyTorch config: data dependence causes a Dynamo graph break, so Inductor never sees it.
Right is with capture on: data dependence reaches Inductor, so we can see how many pieces and why.
Looking at either side alone leads to the wrong conclusion.
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict

SUITE = {"tv": "torchvision", "tvdet": "detection/seg", "timm": "TIMM", "hf": "HuggingFace"}


def load(p):
    return {json.loads(l)["model"]: json.loads(l) for l in open(p) if l.strip()}


def parts(r):
    return r.get("n_partitions_max") or (1 if r.get("ok") else None)


def main(pa, pb):
    A, B = load(pa), load(pb)
    models = sorted(set(A) | set(B))
    print(f"default config {len(A)} rows   capture config {len(B)} rows   union {len(models)}\n")

    hdr = f"{'':<28}{'default (stock)':>16}{'data-dep capture':>20}"
    print(hdr); print("-" * len(hdr))

    def row(label, fa, fb):
        print(f"{label:<28}{fa:>16}{fb:>20}")

    def frac(d, pred):
        rs = list(d.values())
        if not rs:
            return "-"
        n = sum(1 for r in rs if pred(r))
        return f"{n} ({100 * n / len(rs):.0f}%)"

    row("compiled OK", frac(A, lambda r: r.get("ok")), frac(B, lambda r: r.get("ok")))
    okA = {k: v for k, v in A.items() if v.get("ok")}
    okB = {k: v for k, v in B.items() if v.get("ok")}
    row("  whole-graph cudagraph",
        frac(okA, lambda r: (parts(r) or 1) <= 1), frac(okB, lambda r: (parts(r) or 1) <= 1))
    row("  split into pieces",
        frac(okA, lambda r: (parts(r) or 1) > 1), frac(okB, lambda r: (parts(r) or 1) > 1))
    row("has Dynamo graph breaks",
        frac(okA, lambda r: bool(r.get("dynamo_graph_breaks"))),
        frac(okB, lambda r: bool(r.get("dynamo_graph_breaks"))))
    row("cudagraph skipped entirely",
        frac(okA, lambda r: bool(r.get("cudagraph_skips"))),
        frac(okB, lambda r: bool(r.get("cudagraph_skips"))))

    # models that go from "compiles" to "fails to compile" once capture is on
    regress = [m for m in models
               if A.get(m, {}).get("ok") and not B.get(m, {}).get("ok")]
    if regress:
        print(f"\n{len(regress)} fail to compile only after capture is on (they crash once data dependence reaches Inductor)")
        for t, n in Counter(B[m].get("error_type") for m in regress).most_common():
            print(f"  {n:>4}  {t}")
        for m in regress[:8]:
            print(f"    {m:<42} {B[m].get('error_type')}")

    # partitions that only show up once capture is on
    newly = [m for m in models
             if (parts(A.get(m, {})) or 1) <= 1 and (parts(B.get(m, {})) or 1) > 1]
    if newly:
        print(f"\n{len(newly)} look fine by default and are only shown to be split once capture is on")
        for m in newly[:12]:
            r = B[m]
            print(f"    {m:<42} {parts(r)} pieces  {r.get('partition_reason_counts')}")

    # by reason
    for name, d in (("default config", A), ("capture config", B)):
        pm, pn = Counter(), Counter()
        for r in d.values():
            if not r.get("ok"):
                continue
            for slug, n in (r.get("partition_reason_counts") or {}).items():
                pm[slug] += 1; pn[slug] += n
        if pm:
            print(f"\n{name} partition reasons (models / nodes)")
            for slug, m in pm.most_common():
                print(f"  {m:>4} / {pn[slug]:>6}   {slug}")

    # by suite
    print("\nby suite (capture config, fraction with whole-graph cudagraph)")
    bys = defaultdict(list)
    for m, r in okB.items():
        bys[m.split(":", 1)[0]].append(r)
    for s, rs in sorted(bys.items()):
        n = sum(1 for r in rs if (parts(r) or 1) <= 1)
        print(f"  {SUITE.get(s, s):<14} {n}/{len(rs)}  ({100 * n / len(rs):.0f}%)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
