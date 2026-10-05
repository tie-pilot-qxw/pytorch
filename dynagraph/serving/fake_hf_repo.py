"""Materialize a HF repo with random weights: small files downloaded as is, every *.safetensors rebuilt
from its header (fetched with a Range request) with random values. No weights downloaded.

  python fake_hf_repo.py Tongyi-MAI/Z-Image-Turbo /fake/Z-Image-Turbo
"""
import json
import os
import struct
import sys

import requests
import torch
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from safetensors.torch import save_file

repo, out = sys.argv[1], sys.argv[2]
api = HfApi()
files = api.list_repo_files(repo)
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
      "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8, "BOOL": torch.bool, "I8": torch.int8}
tok = os.environ.get("HF_TOKEN")
hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
g = torch.Generator().manual_seed(0)
for f in files:
    dst = os.path.join(out, f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not f.endswith(".safetensors"):
        if f.endswith((".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".msgpack", ".h5")) or f.startswith("assets/"):
            continue
        p = hf_hub_download(repo, f)
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(p), dst)
        continue
    url = hf_hub_url(repo, f)
    n = struct.unpack("<Q", requests.get(url, headers={**hdr, "Range": "bytes=0-7"}, allow_redirects=True).content)[0]
    meta = json.loads(requests.get(url, headers={**hdr, "Range": f"bytes=8-{7 + n}"}, allow_redirects=True).content)
    meta.pop("__metadata__", None)
    ts = {}
    for name, m in meta.items():
        dt = DT[m["dtype"]]
        if dt.is_floating_point:
            t = torch.randn(m["shape"], generator=g, dtype=torch.float32).mul_(0.02).to(dt)
        else:
            t = torch.zeros(m["shape"], dtype=dt)
        ts[name] = t
    save_file(ts, dst)
    print(f"{f}: {len(ts)} tensors, {sum(t.numel() * t.element_size() for t in ts.values()) / 2**30:.2f} GiB", flush=True)
