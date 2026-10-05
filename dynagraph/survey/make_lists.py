#!/usr/bin/env python3
"""
Generate our model spec list from the lists in pytorch-main/benchmarks/dynamo.

List formats:
  huggingface_models_list.txt   "AlbertForMaskedLM,8"        comma
  timm_models_list.txt          "adv_inception_v3 128"       space
  all_torchbench_models_list.txt"BERT_pytorch,128"           comma (torchbench not installed, recorded only)

Batch size is always dropped: the survey counts compile-time partition decisions, which do not depend on batch; see models.py.
"""
from __future__ import annotations
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = "/workspace/pytorch-main/benchmarks/dynamo"
if not os.path.isdir(BENCH):
    # this repo is the PyTorch checkout: DG/survey -> ../../benchmarks/dynamo
    BENCH = os.path.normpath(os.path.join(HERE, "..", "..", "benchmarks", "dynamo"))

# The paper-era (v2.1.0) lists are much more complete than the local checkout: HF 51 vs 39, TIMM 61 vs 18.
# A local copy lives under lists/, so we do not depend on a temp directory.
LISTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lists")

# torchvision: covers classification + detection/segmentation. The latter use NMS / topk / boolean indexing internally,
# a natural source of unbacked symints -- exactly the kind of model we want to count.
TV_CLASSIFY = [
    "resnet18", "resnet50", "resnext50_32x4d", "wide_resnet50_2",
    "densenet121", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
    "efficientnet_b0", "efficientnet_v2_s", "convnext_tiny", "regnet_y_400mf",
    "shufflenet_v2_x1_0", "squeezenet1_1", "vgg16", "alexnet",
    "vit_b_16", "swin_t", "maxvit_t", "inception_v3",
]
TV_DETECT = [
    "fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_fpn",
    "retinanet_resnet50_fpn", "ssd300_vgg16", "ssdlite320_mobilenet_v3_large",
    "maskrcnn_resnet50_fpn", "keypointrcnn_resnet50_fpn",
]

# no longer present in transformers 5.x; skip
HF_REMOVED = {
    "Speech2Text2ForCausalLM",   # removed in 5.x, no replacement
    "XLNetLMHeadModel",          # needs perm_mask/target_mapping; the generic input builder does not apply
}

# names from the paper-era lists are spelled differently in the new library versions
RENAME = {
    "CamemBert": "CamembertForMaskedLM",   # used to be an alias in EXTRA_MODELS
    "SelecSls42b": "selecsls42b",          # timm is all lowercase now
}
# these go through AutoConfig.from_pretrained, unavailable offline; skip for now
HF_NEEDS_HUB = {
    "AllenaiLongformerBase", "T5Small", "DistillGPT2",
    "GoogleFnet", "YituTechConvBert", "DebertaV2ForMaskedLM",
}


def read_list(fname, sep, base=None):
    path = os.path.join(base or BENCH, fname)
    if not os.path.exists(path):
        print(f"  (missing {fname})", file=sys.stderr)
        return []
    out = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        name = ln.split(sep)[0].strip()
        name = RENAME.get(name, name)
        # names containing "/" are hub model IDs (from huggingface_llm_models.py); they need a download and are often 7B/20B,
        # not suited to a coverage survey; skip.
        if "/" in name:
            continue
        out.append(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models_all.txt")
    ap.add_argument("--suites", default="tv,tvdet,timm,hf")
    a = ap.parse_args()
    want = set(a.suites.split(","))

    specs, stats = [], {}

    if "tv" in want:
        specs += [f"tv:{m}" for m in TV_CLASSIFY]
        stats["torchvision classify"] = len(TV_CLASSIFY)
    if "tvdet" in want:
        specs += [f"tvdet:{m}" for m in TV_DETECT]
        stats["torchvision det/seg"] = len(TV_DETECT)
    if "timm" in want:
        t = read_list("v21_timm_models_list.txt", " ", LISTS) or \
            read_list("timm_models_list.txt", " ")
        specs += [f"timm:{m}" for m in t]
        stats["timm"] = len(t)
    if "hf" in want:
        raw = read_list("v21_hf_list.txt", ",", LISTS) or \
              read_list("huggingface_models_list.txt", ",")
        h = [m for m in raw
             if m not in HF_REMOVED and m not in HF_NEEDS_HUB]
        specs += [f"hf:{m}" for m in h]
        stats["huggingface"] = len(h)

    with open(a.out, "w") as f:
        f.write("\n".join(specs) + "\n")

    for k, v in stats.items():
        print(f"  {k:<24} {v}")
    print(f"total {len(specs)} -> {a.out}")


if __name__ == "__main__":
    main()
