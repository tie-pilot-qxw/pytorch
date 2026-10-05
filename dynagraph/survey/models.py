"""
model spec -> (module, example_inputs)

specs look like "timm:resnet50" / "hf:BertForMaskedLM" / "tv:resnet18" / "tvdet:..." / "builtin:..."

Deliberately does not depend on pytorch-main/benchmarks/dynamo/common.py: that module requires a matching
torch version at import time (it references torch._dynamo.utils.copy_dynamo_tensor_attributes, which only main has),
and it drags in a pile of global state. Only its core model-construction recipe is copied here:
for HuggingFace, `model_cls.config_class()` + `model_cls(config)`, **fully offline, never touches the hub**.

Batch size is always small: the survey tallies compile-time partition decisions, which do not depend on batch,
and under dynamic=True the batch gets symbolized anyway. Small batches just keep GPU memory use down.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import os

# The real survey uses cuda. When validating lists, set DYNAGRAPH_DEVICE=cpu/meta to avoid occupying a card.
DEV = os.environ.get("DYNAGRAPH_DEVICE", "cuda")
# runner sets this before build. Only detection-model construction depends on it (training needs targets).
TRAIN = False


def device_ctx():
    """
    Callers must wrap build() in `with models.device_ctx():`.

    Cannot .to(DEV) inside build: under FakeTensorMode the model params are already FakeTensors,
    and .to() then goes through torch.utils.swap_tensors and fails with
    "RuntimeError: _apply(): Couldn't swap Conv2d.weight".
    The torch.device context makes params get created on the target device from the start, which works both fake and non-fake.
    """
    return torch.device(DEV)
BS = 2
SEQ = 128


# ---------------------------------------------------------------- builtin probes
class _Plain(nn.Module):
    def __init__(self):
        super().__init__(); self.a = nn.Linear(256, 256); self.b = nn.Linear(256, 256)
    def forward(self, x):
        return self.b(self.a(x).relu())


class _DataDependent(nn.Module):
    def __init__(self):
        super().__init__(); self.a = nn.Linear(256, 256)
    def forward(self, x):
        y = self.a(x).relu()
        return y[y.sum(-1) > 0]


class _Nonzero(nn.Module):
    def __init__(self):
        super().__init__(); self.a = nn.Linear(256, 256)
    def forward(self, x):
        y = self.a(x)
        return y[torch.nonzero(y[:, 0] > 0).squeeze(-1)]


class _CpuOp(nn.Module):
    """Drops to CPU midway: triggers the non-gpu-ops partition reason, which is independent of unbacked"""
    def __init__(self):
        super().__init__(); self.a = nn.Linear(256, 256)
    def forward(self, x):
        y = self.a(x)
        s = y.sum().cpu().item()
        return y * s


class _Item(nn.Module):
    def __init__(self):
        super().__init__(); self.a = nn.Linear(256, 256)
    def forward(self, x):
        y = self.a(x)
        n = int(y.shape[0] * 0 + y.argmax().item() % 8 + 1)
        return y[:, :n * 32]


_BUILTIN = {
    "plain": _Plain, "datadep": _DataDependent, "nonzero": _Nonzero,
    "cpuop": _CpuOp, "item": _Item,
}


# ---------------------------------------------------------------- suites
def _build_builtin(name):
    return _BUILTIN[name](), (torch.randn(64, 256),), {}


def _build_tv(name):
    import torchvision.models as tvm
    return (getattr(tvm, name)(weights=None).eval(),
            (torch.randn(BS, 3, 224, 224),), {})


def _build_tvdet(name):
    # Detection/segmentation models contain NMS / topk / boolean indexing, a natural source of data dependence
    import torchvision.models.detection as tvd
    m = getattr(tvd, name)(weights=None, weights_backbone=None)
    img = torch.randn(3, 224, 224)
    if not TRAIN:
        return m.eval(), ([img],), {}
    # In training mode torchvision detection models require targets:
    #   ssd.py:330  torch._assert(False, "targets should not be none when in training mode")
    # Without it, all 7 tvdet models in the --train round abort.
    # boxes must be [N,4] with x2>x1, y2>y1 (degenerate boxes are caught by another assertion).
    m.train()
    # Must be **2 images**: ssdlite320's BatchNorm throws at batch=1
    # "Expected more than 1 value per channel when training" (one layer's feature map shrinks to 1x1).
    # Labels are always 1: keypointrcnn defaults to num_classes=2 (background + person), so 2 is out of range.
    def _t():
        return {
            "boxes": torch.tensor([[10.0, 10.0, 80.0, 80.0],
                                   [100.0, 100.0, 180.0, 180.0]]),
            "labels": torch.ones(2, dtype=torch.int64),
            # The mask / keypoint variants each need one more key; extra keys are harmless, missing ones crash.
            # keypoints is [N, K, 3], K is the default 17, the third column is visibility; it must be 1 to give valid points.
            "masks": torch.ones(2, 224, 224, dtype=torch.uint8),
            "keypoints": torch.cat(
                [torch.full((2, 17, 2), 50.0), torch.ones(2, 17, 1)], dim=-1),
        }
    return m, ([img, torch.randn_like(img)], [_t(), _t()]), {}


def _build_timm(name):
    import timm
    m = timm.create_model(name, pretrained=False).eval()
    size = 224
    cfg = getattr(m, "default_cfg", None) or {}
    if isinstance(cfg.get("input_size"), (tuple, list)) and len(cfg["input_size"]) == 3:
        size = cfg["input_size"][-1]
    return m, (torch.randn(BS, 3, size, size),), {}


def _build_hf(name):
    import transformers
    cls = getattr(transformers, name, None)
    if cls is None:
        raise ValueError(f"transformers has no {name} (5.x may have renamed/removed it)")
    config = cls.config_class()
    # Some models need a pad token when BS > 1
    if name.startswith(("Roberta", "Marian")) or "SequenceClassification" in name:
        config.pad_token_id = 0
    model = cls(config).eval()

    vocab = getattr(config, "vocab_size", 30522)
    seq = min(SEQ, getattr(config, "max_position_embeddings", SEQ) or SEQ)
    ids = torch.randint(0, vocab, (BS, seq))
    kwargs = {"input_ids": ids}
    if getattr(config, "is_encoder_decoder", False):
        kwargs["decoder_input_ids"] = ids.clone()
    return model, (), kwargs


_BUILDERS = {
    "builtin": _build_builtin, "tv": _build_tv, "tvdet": _build_tvdet,
    "timm": _build_timm, "hf": _build_hf,
}


def build(spec: str):
    """Always returns the triple (model, args, kwargs)."""
    kind, _, name = spec.partition(":")
    if kind not in _BUILDERS:
        raise ValueError(f"unknown spec prefix: {spec!r}")
    return _BUILDERS[kind](name)
