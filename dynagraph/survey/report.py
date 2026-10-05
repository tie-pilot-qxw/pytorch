#!/usr/bin/env python3
"""
Aggregate runner.py's jsonl into a coverage table.

The output deliberately mirrors Table 1 of PyTorch 2 (ASPLOS'24): grouped by suite,
with that paper's "graph break distribution" replaced by "cudagraph partition distribution".
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict

SUITE_NAME = {
    "tv": "torchvision cls", "tvdet": "torchvision det/seg",
    "timm": "TIMM", "hf": "HuggingFace", "builtin": "synthetic probes",
}


def suite_of(spec: str) -> str:
    return spec.split(":", 1)[0]


def bucket(n):
    if n is None or n <= 1:
        return "whole"
    if n <= 9:
        return "2~9 segs"
    return ">=10 segs"


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_suite = defaultdict(list)
    for r in rows:
        by_suite[suite_of(r["model"])].append(r)

    # state data completeness first: with real tensors a row is only complete if ran=True;
    # the fake path crashes after the first subgraph, and those records carry the data_maybe_truncated flag.
    trunc = [r for r in rows if r.get("data_maybe_truncated")]
    ran = [r for r in rows if r.get("ran")]
    print(f"total {len(rows)} models   ran to completion {len(ran)}   "
          f"possibly truncated {len(trunc)}")
    if trunc:
        print(f"  ** {len(trunc)} rows come from fake mode and may only count the first subgraph; "
              f"treat these numbers as lower bounds only **")
    print()

    # ---- Table 1 layout: one column per suite
    suites = [s for s in ("tv", "tvdet", "timm", "hf", "builtin") if s in by_suite]
    hdr = f"{'':<26}" + "".join(f"{SUITE_NAME.get(s, s):>22}" for s in suites)
    print(hdr)
    print("-" * len(hdr))

    def line(label, fn):
        print(f"{label:<26}" + "".join(f"{fn(by_suite[s]):>22}" for s in suites))

    def frac(rs, pred):
        if not rs:
            return "-"
        n = sum(1 for r in rs if pred(r))
        return f"{n} ({100 * n / len(rs):.0f}%)"

    line("attempted", lambda rs: str(len(rs)))
    line("compiled OK", lambda rs: frac(rs, lambda r: r.get("ok")))
    ok_only = {s: [r for r in by_suite[s] if r.get("ok")] for s in suites}

    def line_ok(label, fn):
        print(f"{label:<26}" + "".join(f"{fn(ok_only[s]):>22}" for s in suites))

    line_ok("whole graph cudagraphable", lambda rs: frac(rs, lambda r: bucket(r.get("n_partitions_max")) == "whole"))
    line_ok("split into 2~9 segs", lambda rs: frac(rs, lambda r: bucket(r.get("n_partitions_max")) == "2~9 segs"))
    line_ok("split into >=10 segs", lambda rs: frac(rs, lambda r: bucket(r.get("n_partitions_max")) == ">=10 segs"))
    # two different metrics, report both:
    #   cudagraph_skips                 whole graph gave up (counter)
    #   n_partitions_skipping_cudagraph some **segments** gave up (graph_partition signature flag)
    # the latter is much finer-grained and more telling: a split model is often not "one graph per segment"; some segments never enter a graph at all.
    line_ok("skips cudagraph (whole)", lambda rs: frac(rs, lambda r: bool(r.get("cudagraph_skips"))))
    line_ok("skips cudagraph (a seg)", lambda rs: frac(rs, lambda r: bool(r.get("n_partitions_skipping_cudagraph"))))
    line_ok("has Dynamo graph break", lambda rs: frac(rs, lambda r: bool(r.get("dynamo_graph_breaks"))))

    # ---- partition reason breakdown
    print("\npartition reasons (models with this reason / nodes involved)")
    per_model, per_node = Counter(), Counter()
    for r in rows:
        if not r.get("ok"):
            continue
        for slug, n in (r.get("partition_reason_counts") or {}).items():
            per_model[slug] += 1
            per_node[slug] += n
    if per_model:
        for slug, m in per_model.most_common():
            print(f"  {m:>4} models / {per_node[slug]:>6} nodes   {slug}")
    else:
        print("  (no model was split)")

    # ---- reasons the whole graph gave up
    skips = Counter()
    for r in rows:
        for s2, n in (r.get("skip_reason_slugs") or {}).items():
            skips[s2] += n
    if skips:
        print("\nreasons the whole graph skipped cudagraph")
        for s2, n in skips.most_common(12):
            print(f"  {n:>4}  {s2}")

    # ---- Dynamo graph break reasons
    gb = Counter()
    for r in rows:
        for s2, n in (r.get("dynamo_break_reasons") or {}).items():
            gb[s2[:90]] += n
    if gb:
        print("\nDynamo graph break reasons (layer 1)")
        for s2, n in gb.most_common(12):
            print(f"  {n:>4}  {s2}")

    # ---- failures
    bad = [r for r in rows if not r.get("ok")]
    if bad:
        print(f"\n{len(bad)} with no data")
        for t, n in Counter(r.get("error_type") for r in bad).most_common():
            print(f"  {n:>4}  {t}")
        # list timeouts and OOMs in full: they are not random failures,
        # slow and large may both correlate with "many partitions"; hiding them is selection bias.
        for kind, pred in (("timeout", lambda r: r.get("error_type") == "Timeout"),
                           ("out of memory", lambda r: "OutOfMemory" in str(r.get("error_type"))
                            or "out of memory" in str(r.get("error", "")).lower()),
                           ("subprocess crashed", lambda r: r.get("error_type") == "NoResult")):
            hit = [r["model"] for r in bad if pred(r)]
            if hit:
                print(f"  {kind} ({len(hit)}, all listed):")
                for m in hit:
                    print(f"      {m}")
        other = [r for r in bad if r.get("error_type")
                 not in ("Timeout", "NoResult") and "OutOfMemory" not in str(r.get("error_type"))]
        if other:
            print("  other failure examples:")
            for r in other[:5]:
                print(f"      {r['model']:<40} {r.get('error_type')}: {str(r.get('error'))[:70]}")

    # ---- self-check: did the harness run in the expected mode
    apis = Counter(r.get("scheduler_api") for r in rows if r.get("ok"))
    print(f"\nself-check  scheduler_api={dict(apis)}"
          f"  fake={Counter(r.get('fake') for r in rows)}")
    if "bool" in apis:
        print("  ** some models ran on an old torch that returns bool; those rows have no partition reasons **")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")
