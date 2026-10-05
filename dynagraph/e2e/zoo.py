"""GraCE's (OSDI'26) HF workloads, but with the shape changing every step: can PT2 still use CUDA Graphs under dynamic shapes?

GraCE uses TorchBench / HuggingFace / TIMM models, each at one fixed batch size, and compares PyTorch2 with and without
CUDA Graphs. Here we take the same HF models (randomly initialized, default config or GraCE's sizes), run inference, random (B, L) every step:
  B picked from --bs, L log-uniform on [--lmin, --lmax] -- what request lengths look like in serving
Modes are described in harness: eager / compile(dynamic, no graph) / trees(reduce-overhead, one recording per new shape) /
pad(everything padded to (max B, max L), one static graph) / dg.

  AMP=1 python e2e/zoo.py --model bert --modes eager,compile,trees,pad
"""
import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import harness
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="bert")
ap.add_argument("--batches", type=int, default=100)
ap.add_argument("--warm", type=int, default=20)
ap.add_argument("--bs", default="1")
ap.add_argument("--lmin", type=int, default=16)
ap.add_argument("--lmax", type=int, default=512)
ap.add_argument("--modes", default="eager,compile,trees,pad")
a = ap.parse_args()
if os.environ.get("CUDNN_SDP") != "1":
    # cuDNN SDPA builds an execution plan per new shape (~5 ms per layer on H100): a library cost that
    # swamps everything else on new shapes and that no graph scheme removes. Off by default.
    torch.backends.cuda.enable_cudnn_sdp(False)


def _cfg():
    import transformers as t

    kw = dict(attn_implementation=os.environ.get("ATTN", "sdpa"))
    m = a.model
    # (config, model class, encoder-decoder?)
    if m == "bert":
        return t.BertConfig(**kw), t.BertForMaskedLM, False
    if m == "distilgpt2":
        return t.GPT2Config(n_layer=6, **kw), t.GPT2LMHeadModel, False
    if m == "xlnet":
        # GraCE: XLNetLMHeadModel; xlnet-base size
        return t.XLNetConfig(d_model=768, n_layer=12, n_head=12, d_inner=3072), t.XLNetLMHeadModel, False
    if m == "mt5":
        return t.MT5Config(), t.MT5ForConditionalGeneration, True
    if m == "t5":
        return t.T5Config(), t.T5ForConditionalGeneration, True
    if m == "mobilebert":
        return t.MobileBertConfig(), t.MobileBertForQuestionAnswering, False
    if m == "debertav2":
        # deberta-v3-base size
        return t.DebertaV2Config(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                                 intermediate_size=3072, relative_attention=True, position_buckets=256,
                                 pos_att_type=["p2c", "c2p"], max_relative_positions=-1,
                                 position_biased_input=False), t.DebertaV2ForQuestionAnswering, False
    if m == "deberta":
        return t.DebertaConfig(), t.DebertaForMaskedLM, False
    if m == "blenderbot":
        return t.BlenderbotSmallConfig(), t.BlenderbotSmallForConditionalGeneration, True
    if m == "albert":
        return t.AlbertConfig(hidden_size=768, num_attention_heads=12, intermediate_size=3072), t.AlbertForMaskedLM, False
    if m == "electra":
        return t.ElectraConfig(), t.ElectraForMaskedLM, False
    raise SystemExit(f"unknown model {m}")


CFG, CLS, ENCDEC = _cfg()
VOCAB = min(getattr(CFG, "vocab_size", 30000), 30000)


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, mask):
        kw = dict(input_ids=ids, attention_mask=mask)
        if ENCDEC:
            kw["decoder_input_ids"] = ids
            kw["decoder_attention_mask"] = mask
        out = self.m(**kw)
        return out.logits if hasattr(out, "logits") else out.start_logits


def batches():
    rng = random.Random(0)
    bs = [int(x) for x in a.bs.split(",")]
    out = []
    for _ in range(a.batches):
        B = rng.choice(bs)
        L = int(round(math.exp(rng.uniform(math.log(a.lmin), math.log(a.lmax)))))
        ids = torch.randint(5, VOCAB, (B, L), generator=torch.Generator().manual_seed(len(out)))
        # Rows have different real lengths; mask out the tail (in-row padding is HF's own convention, unrelated to shape padding)
        lens = [rng.randint(max(1, L // 2), L) for _ in range(B)]
        lens[0] = L
        mask = torch.zeros(B, L, dtype=torch.long)
        for i, n in enumerate(lens):
            mask[i, :n] = 1
        out.append((ids.cuda(), mask.cuda()))
    return out


def pad(bs, bucket=False):
    B = max(x.shape[0] for x, _ in bs)
    L = max(x.shape[1] for x, _ in bs)
    out = []
    for x, m in bs:
        if bucket:
            # vLLM-style: each dim up to the next power of two, one static graph per bucket
            B, L = (1 << (n - 1).bit_length() for n in x.shape)
        xp = x.new_zeros(B, L)
        xp[: x.shape[0], : x.shape[1]] = x
        mp = m.new_zeros(B, L)
        mp[: m.shape[0], : m.shape[1]] = m
        mp[x.shape[0]:, 0] = 1
        out.append((xp, mp))
    return out


def main():
    bs = batches()
    shapes = [tuple(x.shape) for x, _ in bs]
    toks = [b * l for b, l in shapes]
    B = max(s[0] for s in shapes)
    L = max(s[1] for s in shapes)
    print(f"{a.model}: {len(bs)} steps, {len(set(shapes))} distinct (B, L), new-segment shapes unseen in warmup: "
          f"{len(set(shapes[a.warm:]) - set(shapes[:a.warm]))}; padding to ({B}, {L}) gives "
          f"{B * L / (sum(toks) / len(toks)):.1f}x the mean token count", flush=True)

    def make():
        torch.manual_seed(0)
        m = Wrap(CLS(CFG)).cuda().eval()

        def step(f, b):
            # In pad mode the output includes the padded rows and columns, so the loss difference is only a reference
            x, mk = b
            with torch.no_grad(), harness.amp():
                y = f(x, mk)
            return y.reshape(y.shape[0], -1)[:, :64].float().mean()

        return m, step

    modes = [x for x in a.modes.split(",") if x]
    harness.run(make, bs, [x for x in modes if x not in ("oracle", "bucket")], warm=a.warm, label=a.model, pad=pad)
    if "bucket" in modes:
        # one recompile + one recording per bucket, like pad but per bucket
        torch._dynamo.config.recompile_limit = 256
        torch._dynamo.config.cache_size_limit = 256
        harness.run(make, bs, ["pad"], warm=a.warm, label=f"{a.model}/bucket", pad=lambda b: pad(b, bucket=True))
    if "oracle" in modes:
        oracle(make, bs[a.warm:])


def oracle(make, bs):
    """Upper bound for DynaGraph: the same compile(dynamic=True) code, one graph captured per shape, timing replay only.
    No recording, no padding, no DG overhead of its own -- the fastest that patching one graph per shape can possibly be."""
    import time

    model, step = make()
    torch._dynamo.reset()
    f = torch.compile(model, dynamic=True)
    for b in bs[:3]:
        step(f, b)
    ts = []
    for x, mk in bs:
        xs, ms = x.clone(), mk.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                step(f, (xs, ms))
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step(f, (xs, ms))
        g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / 10)
        del g
    ts.sort()
    print(f"[{a.model}] oracle  one graph per shape, replay only: median {ts[len(ts) // 2] * 1e3:.2f} ms  mean {sum(ts) / len(ts) * 1e3:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
