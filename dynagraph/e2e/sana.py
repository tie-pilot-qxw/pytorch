"""SANA 1.5 1.6B DiT (diffusers SanaTransformer2DModel, random weights, bf16) served the way SGLang serves it.

Each request picks one of SANA's 33 aspect-ratio bins at 1024px (ASPECT_RATIO_1024_BIN), runs STEPS denoising
steps with CFG (batch 2), all at that one shape. Text: TEXT=fixed pads to 300 tokens like the diffusers pipeline;
TEXT=var draws a length in [8, 300] per request (SGLang keeps the real length, bucketed).

  python e2e/sana.py --modes eager,compile,trees,dg,pad,oracle
Per-forward times come from harness; per-request totals (what a user waits for) are printed at the end.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import harness
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--requests", type=int, default=24)
ap.add_argument("--warm-requests", type=int, default=3)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--layers", type=int, default=20)
ap.add_argument("--modes", default="eager,compile,trees,dg,pad,oracle")
a = ap.parse_args()
TEXT = os.environ.get("TEXT", "fixed")
if os.environ.get("CUDNN_SDP") != "1":
    torch.backends.cuda.enable_cudnn_sdp(False)

CFG = dict(attention_bias=False, attention_head_dim=32, caption_channels=2304, cross_attention_dim=2240,
           cross_attention_head_dim=112, dropout=0.0, guidance_embeds=False, in_channels=32, interpolation_scale=None,
           mlp_ratio=2.5, norm_elementwise_affine=False, norm_eps=1e-6, num_attention_heads=70,
           num_cross_attention_heads=20, num_layers=a.layers, out_channels=32, patch_size=1,
           qk_norm="rms_norm_across_heads", sample_size=32)


def requests():
    from diffusers.pipelines.pixart_alpha import pipeline_pixart_alpha as pp

    BINS = getattr(pp, f"ASPECT_RATIO_{os.environ.get('BIN', '1024')}_BIN")

    rng = random.Random(0)
    bins = [(int(h), int(w)) for h, w in BINS.values()]
    out = []
    for r in range(a.requests):
        h, w = rng.choice(bins)
        L = 300 if TEXT == "fixed" else rng.randint(8, 300)
        g = torch.Generator().manual_seed(r)
        enc = (torch.randn(2, L, 2304, generator=g) * 0.5).to(torch.bfloat16).cuda()
        mask = torch.ones(2, L, dtype=torch.long)
        mask[:, rng.randint(max(1, L // 3), L):] = 0  # real prompt shorter than the padded length
        mask = mask.cuda()
        lat = torch.randn(2, 32, h // 32, w // 32, generator=g).to(torch.bfloat16).cuda()
        for s in range(a.steps):
            t = torch.full((2,), 999.0 * (1 - s / a.steps)).cuda()
            out.append((lat, enc, mask, t, r))
    return out


def pad(bs):
    """Every request to the largest latent (H and W separately) and the longest text; reference only (the padded
    image tokens take part in linear attention, so the output differs)."""
    H = max(b[0].shape[2] for b in bs)
    W = max(b[0].shape[3] for b in bs)
    L = max(b[1].shape[1] for b in bs)
    out = []
    for lat, enc, mask, t, r in bs:
        lp = lat.new_zeros(2, 32, H, W)
        lp[:, :, : lat.shape[2], : lat.shape[3]] = lat
        ep = enc.new_zeros(2, L, enc.shape[2])
        ep[:, : enc.shape[1]] = enc
        mp = mask.new_zeros(2, L)
        mp[:, : mask.shape[1]] = mask
        out.append((lp, ep, mp, t, r))
    return out


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, lat, enc, mask, t):
        return self.m(hidden_states=lat, encoder_hidden_states=enc, encoder_attention_mask=mask, timestep=t,
                      return_dict=False)[0]


def make():
    from diffusers import SanaTransformer2DModel

    torch.manual_seed(0)
    m = Wrap(SanaTransformer2DModel(**CFG)).to(torch.bfloat16).cuda().eval()

    def step(f, b):
        lat, enc, mask, t, _ = b
        with torch.no_grad():
            y = f(lat, enc, mask, t)
        return y[:, :, :2, :2].float().mean()

    return m, step


def per_request(ts, reqs):
    tot = {}
    for t, r in zip(ts, reqs):
        tot[r] = tot.get(r, 0.0) + t
    return list(tot.values())


def oracle(bs):
    """compile(dynamic=True), one CUDA graph per request shape, replay only: the ceiling for one graph patched per
    shape (no dynamo / AOT front end, no recording)."""
    model, step = make()
    torch._dynamo.reset()
    f = torch.compile(model, dynamic=True)
    for b in bs[:2]:
        step(f, b)
    seen, ts = {}, []
    for b in bs:
        key = (tuple(b[0].shape), tuple(b[1].shape))
        if key not in seen:
            st = [x.clone() for x in b[:4]] + [b[4]]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    step(f, st)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step(f, st)
            seen = {key: g}  # one live graph at a time
        g = seen[key]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    bs = requests()
    reqs = [b[4] for b in bs]
    warm = a.warm_requests * a.steps
    shapes = {(tuple(b[0].shape), tuple(b[1].shape)) for b in bs}
    new = {(tuple(b[0].shape), tuple(b[1].shape)) for b in bs[warm:]} - {(tuple(b[0].shape), tuple(b[1].shape)) for b in bs[:warm]}
    print(f"sana BIN={os.environ.get('BIN', '1024')}: {a.requests} requests x {a.steps} steps, {len(shapes)} shapes, {len(new)} unseen in the new segment, TEXT={TEXT}", flush=True)
    modes = [m for m in a.modes.split(",") if m]
    res = harness.run(make, bs, [m for m in modes if m != "oracle"], warm=warm, label="sana", pad=pad)
    summary = {}
    for m, r in res.items():
        pr = per_request(r["per_seg"]["new"], reqs[warm:])
        prr = per_request(r["per_seg"]["replay"], reqs[warm:])
        summary[m] = dict(new=pr, replay=prr)
        print(f"[sana] {m:<7} per request ({a.steps} steps): new median {statistics.median(pr) * 1e3:8.1f} ms  mean {statistics.mean(pr) * 1e3:8.1f} ms  "
              f"max {max(pr) * 1e3:9.1f} ms | replay median {statistics.median(prr) * 1e3:8.1f} ms", flush=True)
    if "oracle" in modes:
        ts = oracle(bs[warm:])
        pr = per_request(ts, reqs[warm:])
        summary["oracle"] = dict(new=pr)
        print(f"[sana] oracle  per request ({a.steps} steps): median {statistics.median(pr) * 1e3:8.1f} ms  mean {statistics.mean(pr) * 1e3:8.1f} ms", flush=True)
    os.makedirs(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), exist_ok=True)
    json.dump(summary, open(os.environ.get("OUT", os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "sana.json")), "w"))


if __name__ == "__main__":
    main()
