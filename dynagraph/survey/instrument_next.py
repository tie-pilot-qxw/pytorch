"""
Record every "cannot go into cudagraph" decision in one torch.compile run, with its reason.

Line numbers refer to pytorch-main @ 71b32515a2. Four hook points:

  1. Scheduler.should_partition          scheduler.py:11175-11227
     main returns a reason string (str|None); the NVIDIA-packaged 2.11 only returns bool. Both are supported,
     decided by the actual return value, not the signature.
     Note: **it is called ~3 times per node per compile** (once each from reorder_for_minimizing_partition
     and other places), so it must be deduplicated by (graph_id, node name), otherwise counts are 3x too high.

  2. Scheduler.graph_partition
     Returns (partitions, signatures). This is the only place that gives the **real partition count**:
     counters["inductor"]["cudagraph_partitions"] is only incremented when len(partitions) > 1
     (scheduler.py:11979), so an empty counter does not mean "not split"; it may be exactly 1 segment.
     skip_cudagraph in signatures tells which segments still gave up cudagraph in the end.

  3. log_cudagraph_skip_and_bump_counter cudagraph_utils.py:404-413
     Reasons the whole graph gives up cudagraph; a separate mechanism from partitioning. Four modules each
     import this name, so each one must be rebound; patching only the definition does nothing.
     Upstream double-prefixes the message as "skipping cudagraphs due to skipping cudagraphs due to
     ..."; normalization strips that.

  4. Dynamo graph breaks                 counters["graph_break"]
     Data dependence is blocked at this layer by default; not recording it would misjudge a model as "no partition problem".
"""
from __future__ import annotations

import re
from collections import Counter

_state: dict = {}
_installed = False

_SKIP_MODULES = (
    "torch._inductor.cudagraph_utils",
    "torch._inductor.output_code",
    "torch._inductor.compile_fx",
    "torch._inductor.cudagraph_trees",
)


# ---------------------------------------------------------------- reason normalization
_RULES = [
    (r"^partition includes all ops when cudagraphs is disabled$", "cudagraphs_disabled"),
    (r"^custom partition op: ", "custom_partition_op"),
    (r"^unbacked binding ops$", "unbacked_binding"),
    (r"^uses cudagraph-unsafe unbacked symint", "unsafe_unbacked_symint"),
    (r"^CUDAGraph-unsafe custom ops$", "cudagraph_unsafe_custom_op"),
    (r"^DeviceCopy ops$", "device_copy"),
    (r"^Switch ops$", "switch"),
    (r"^dynamic shape ops$", "dynamic_shape"),
    (r"^cpu ops$", "cpu_ops"),
    (r"\bops$", "non_gpu_ops"),
]


def normalize(reason: str) -> str:
    r = reason.strip()
    for pat, slug in _RULES:
        if re.search(pat, r):
            return slug
    return "other:" + re.sub(r"[^a-z_]+", "_", r.lower())[:40]


def normalize_skip(msg: str) -> str:
    m = re.sub(r"(skipping cudagraphs due to )+", "", msg).strip()
    m = m.split("Found from")[0].strip().rstrip(".")
    return re.sub(r"\s+", " ", m)[:100]


# ---------------------------------------------------------------- install
def reset() -> None:
    keep = {k: _state.get(k) for k in ("_rebound", "api") if k in _state}
    _state.clear()
    _state.update(
        seen=set(),        # dedup by (graph_id, node_name), cancels the ~3 calls per node
        reasons={},        # (graph_id, node_name) -> reason string
        skip_msgs=[],
        raw_calls=0,
        origins=[],        # (reason slug, origin op) -- used to split the attribution
        partitions=[],     # per graph_partition call: (segment count, segments that gave up cudagraph)
    )
    _state.update(keep)


def install() -> None:
    global _installed
    if _installed:
        return

    import importlib
    import torch._inductor.scheduler as S
    import torch._inductor.cudagraph_utils as CU

    reset()

    # ---- 1. should_partition
    orig_sp = S.Scheduler.should_partition
    # fallback: when no node gets partitioned out, every return value is None, so the return values alone cannot tell the API shape.
    try:
        import inspect
        _ann = str(inspect.signature(orig_sp).return_annotation)
        _state["api"] = "str|None" if "str" in _ann else "bool"
    except Exception:
        pass

    def patched_sp(self, node, *a, **kw):
        out = orig_sp(self, node, *a, **kw)
        _state["raw_calls"] = _state.get("raw_calls", 0) + 1
        if isinstance(out, str):
            _state["api"] = "str|None"
        elif isinstance(out, bool) and _state.get("api") != "str|None":
            _state["api"] = "bool"
        if out:
            try:
                from torch._inductor.virtualized import V
                gid = getattr(V.graph, "graph_id", None)
            except Exception:
                gid = None
            try:
                name = node.get_name()
            except Exception:
                name = id(node)
            key = (gid, name)
            if key not in _state["seen"]:
                _state["seen"].add(key)
                reason = out if isinstance(out, str) else "<bool-true-no-reason>"
                _state["reasons"][key] = reason
                # record the origin op. Without it there is no way to answer "how many of these cpu_ops are
                # one-off cases fixable upstream like nn.ReLU6, and how many are structural".
                try:
                    ir = getattr(node, "node", None)
                    org = getattr(ir, "origins", None)
                    if org:
                        tgt = sorted({str(getattr(o, "target", o)) for o in org})[:3]
                        _state.setdefault("origins", []).append(
                            (normalize(reason), ",".join(tgt)))
                except Exception:
                    pass
        return out

    S.Scheduler.should_partition = patched_sp

    # ---- 2. graph_partition: the only reliable source of the partition count
    if hasattr(S.Scheduler, "graph_partition"):
        orig_gp = S.Scheduler.graph_partition

        def patched_gp(self, *a, **kw):
            out = orig_gp(self, *a, **kw)
            try:
                parts, sigs = out
                n_skip = sum(1 for s in sigs if getattr(s, "skip_cudagraph", False))
                _state.setdefault("partitions", []).append((len(parts), n_skip))
            except Exception:
                pass
            return out

        S.Scheduler.graph_partition = patched_gp

    # ---- 3. cudagraph skip: every module that imported it must be rebound
    orig_skip = CU.log_cudagraph_skip_and_bump_counter

    def patched_skip(msg):
        _state.setdefault("skip_msgs", []).append(msg)
        return orig_skip(msg)

    rebound = []
    for modname in _SKIP_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(mod, "log_cudagraph_skip_and_bump_counter"):
            mod.log_cudagraph_skip_and_bump_counter = patched_skip
            rebound.append(modname.rsplit(".", 1)[-1])
    _state["_rebound"] = rebound
    _installed = True


# ---------------------------------------------------------------- read results
def _dedup(xs, limit=12):
    seen, out = set(), []
    for x in xs:
        k = str(x)[:140]
        if k not in seen:
            seen.add(k)
            out.append(k)
        if len(out) >= limit:
            break
    return out


def snapshot() -> dict:
    from torch._dynamo.utils import counters

    ind = counters.get("inductor", {})
    gb = counters.get("graph_break", {})
    reasons = list(_state.get("reasons", {}).values())
    slugs = [normalize(r) for r in reasons]
    parts = _state.get("partitions", [])
    skips = _state.get("skip_msgs", [])

    return {
        # layer 1: Dynamo
        "dynamo_graph_breaks": sum(gb.values()),
        "dynamo_break_reasons": dict(sorted(gb.items(), key=lambda kv: -kv[1])[:10]),
        # layer 2: Inductor node-level reasons
        "n_nodes_not_cudagraphable": len(reasons),
        "partition_reason_counts": dict(Counter(slugs)),
        "partition_reason_samples": _dedup(reasons),
        "partition_origins": dict(
            Counter(f"{slug} <- {org}" for slug, org in _state.get("origins", []))
        ),
        # the partition count comes from the graph_partition return value
        "n_partitions_observed": [p[0] for p in parts],
        "n_partitions_max": max((p[0] for p in parts), default=None),
        "n_partitions_skipping_cudagraph": sum(p[1] for p in parts) if parts else None,
        "n_partitions_counter": ind.get("cudagraph_partitions"),
        # layer 3: whole graph given up
        "cudagraph_skips": ind.get("cudagraph_skips"),
        "skip_reason_slugs": dict(Counter(normalize_skip(m) for m in skips)),
        "skip_reason_samples": _dedup(skips),
        # environment self-check
        "scheduler_api": _state.get("api", "unknown"),
        "should_partition_raw_calls": _state.get("raw_calls", 0),
        "skip_hook_rebound": _state.get("_rebound", []),
    }
