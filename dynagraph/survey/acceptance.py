#!/usr/bin/env python3
"""One-off acceptance check after the self-built main is installed: all four items must pass, otherwise the survey data cannot be trusted."""
import os, sys, traceback
import torch

print("=" * 66)
print("1) Which torch is in use")
print(f"   version : {torch.__version__}")
print(f"   path    : {torch.__file__}")
ok1 = "2.15" in torch.__version__ and "dist-packages" not in torch.__file__
print(f"   -> {'OK self-built main' if ok1 else '** not the self-built main **'}")

print("=" * 66)
print("2) Does should_partition return a reason string")
import inspect
import torch._inductor.scheduler as S
sig = inspect.signature(S.Scheduler.should_partition)
ret = sig.return_annotation
print(f"   signature: {sig}")
ok2 = "str" in str(ret)
print(f"   -> {'OK reasons available' if ok2 else '** returns bool only, cannot tally why **'}")

print("=" * 66)
print("3) Can the naive fake tensor path reach the Scheduler")
import torch._inductor.config as ic, torch._dynamo.config as dc
ic.force_disable_caches = True
dc.capture_dynamic_output_shape_ops = True
dc.capture_scalar_outputs = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instrument import install, reset, snapshot
install(); reset()

ok3 = False
try:
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    import torch.nn as nn

    class M(nn.Module):
        def __init__(s):
            super().__init__(); s.a = nn.Linear(256, 256)
        def forward(s, x):
            y = s.a(x).relu()
            return y[torch.nonzero(y[:, 0] > 0).squeeze(-1)]

    fm = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
    with fm, torch.device("cuda"):
        m = M()
        x = torch.randn(64, 256)
        torch.compile(m, dynamic=True, mode="reduce-overhead")(x)
    snap = snapshot()
    ok3 = bool(snap["n_partitions_observed"])
    print(f"   non-graphable nodes : {snap['n_nodes_not_cudagraphable']}")
    print(f"   reasons             : {snap['partition_reason_counts']}")
    print(f"   reason samples      : {snap['partition_reason_samples'][:3]}")
    print(f"   partitions          : {snap['n_partitions_observed']}")
    print(f"   API                 : {snap['scheduler_api']}")
except Exception as e:
    # In fake mode compilation runs to completion; only the cudagraph **run** phase fails.
    # We do not need that part, so it passes as long as graph_partition ran.
    snap = snapshot()
    ok3 = bool(snap["n_partitions_observed"])
    print(f"   run-phase error     : {type(e).__name__}  (expected)")
    print(f"   non-graphable nodes : {snap['n_nodes_not_cudagraphable']}")
    print(f"   reasons             : {snap['partition_reason_counts']}")
    print(f"   reason samples      : {snap['partition_reason_samples'][:3]}")
    print(f"   partitions          : {snap['n_partitions_observed']}")
    print(f"   API                 : {snap['scheduler_api']}")
print(f"   -> {'OK compile-time data complete' if ok3 else '** fake path broken **'}")

print("=" * 66)
print("4) GPU memory cost")
alloc = torch.cuda.memory_allocated() // 1024
resv = torch.cuda.memory_reserved() // (1024 * 1024)
print(f"   memory_allocated : {alloc} KiB   (expect 0)")
print(f"   memory_reserved  : {resv} MiB")
ok4 = alloc == 0
print(f"   -> {'OK zero tensor memory' if ok4 else '** real tensors were allocated **'}")

print("=" * 66)
print("5) Do the model libraries still work on the new torch")
ok5 = True
for name in ("timm", "transformers", "torchvision"):
    try:
        mod = __import__(name)
        print(f"   {name:<14} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        ok5 = False
        print(f"   {name:<14} ** {type(e).__name__}: {str(e)[:120]}")
try:
    os.environ["DYNAGRAPH_DEVICE"] = "cpu"
    import importlib, models
    importlib.reload(models)
    for spec in ("tv:resnet18", "timm:resnet50", "hf:BertForMaskedLM"):
        m, a, k = models.build(spec)
        print(f"   build {spec:<22} OK")
except Exception as e:
    ok5 = False
    print(f"   ** build failed: {type(e).__name__}: {str(e)[:200]}")
print(f"   -> {'OK' if ok5 else '** model libraries unusable **'}")

print("=" * 66)
allok = ok1 and ok2 and ok3 and ok4 and ok5
print("Acceptance:", "all passed" if allok else "some items failed, see above")
sys.exit(0 if allok else 1)
