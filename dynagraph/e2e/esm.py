"""Variable-length training of a protein language model: ESM-2 (HF transformers' EsmForMaskedLM, random init), MLM, UniProt human proteome.

The default config is ESM-2 t12 35M (12 layers, hidden 480, 20 heads, rotary). Batching follows fairseq / ESM:
sequences longer than 1022 are randomly cropped to 1022, sorted by length, cut into batches by a token budget
(B x L <= max_tokens), and the batch order is shuffled.
So B and L both change every step: a batch of short sequences has hundreds of rows, a batch of long ones a few --
a 2-D shape space, and the waste of padding to the global (max B, max L) (K=1) is the "superlinear cost x heavy-tailed
distribution" cell from the original survey.
attention defaults to flex (Inductor's Triton template, tier 1); --attn sdpa uses aten's mem-efficient kernel (extern).
Usage: AMP=1 GEMM=deepgemm PYTHONPATH=${DG_DEPS:-/workspace/_deps}/deepgemm-src:$PYTHONPATH python e2e/esm.py
Data: $DG_DATA/human_proteome.fasta.gz (DG_DATA defaults to /workspace/_deps/data).
"""
import argparse
import gzip
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import harness
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, default=100)
ap.add_argument("--warm", type=int, default=20)
ap.add_argument("--max-tokens", type=int, default=8192)
ap.add_argument("--max-len", type=int, default=1022)
ap.add_argument("--layers", type=int, default=12)
ap.add_argument("--hidden", type=int, default=480)
ap.add_argument("--heads", type=int, default=20)
ap.add_argument("--attn", default="flex", choices=("flex", "sdpa", "eager"))
ap.add_argument("--modes", default=",".join(harness.MODES))
a = ap.parse_args()

# ESM alphabet (fair-esm): <cls> <pad> <eos> <unk>, the residues, then <mask>.
TOKS = ["<cls>", "<pad>", "<eos>", "<unk>"] + list("LAGVSERTIDPKQNFYMHWCXBUZO.-") + ["<null_1>", "<mask>"]
IDX = {t: i for i, t in enumerate(TOKS)}
CLS, PAD, EOS, UNK, MASK = 0, 1, 2, 3, IDX["<mask>"]


def read_fasta(path):
    seqs, cur = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def load():
    rng = random.Random(0)
    seqs = read_fasta(os.path.join(os.environ.get("DG_DATA", "/workspace/_deps/data"), "human_proteome.fasta.gz"))
    rng.shuffle(seqs)
    crops = []
    for s in seqs:
        if len(s) > a.max_len:
            st = rng.randrange(len(s) - a.max_len + 1)
            s = s[st: st + a.max_len]
        crops.append(s)
    # fairseq-style: sort a shuffled chunk by length, cut by token budget, shuffle the batches
    out, chunk = [], sorted(crops[:20000], key=len)
    cur = []
    for s in chunk:
        L = len(s) + 2
        if cur and (len(cur) + 1) * max(L, len(cur[-1]) + 2) > a.max_tokens:
            out.append(cur)
            cur = []
        cur.append(s)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    batches = []
    g = torch.Generator().manual_seed(1)
    for group in out[: a.batches]:
        L = max(len(s) for s in group) + 2
        ids = torch.full((len(group), L), PAD, dtype=torch.long)
        for i, s in enumerate(group):
            ids[i, : len(s) + 2] = torch.tensor([CLS] + [IDX.get(c, UNK) for c in s] + [EOS])
        attn = (ids != PAD).long()
        # MLM: 15% of residues; of those 80% <mask>, 10% random, 10% kept (BERT / ESM)
        res = (ids > EOS) & (ids != PAD)
        pick = (torch.rand(ids.shape, generator=g) < 0.15) & res
        labels = torch.where(pick, ids, torch.full_like(ids, -100))
        r = torch.rand(ids.shape, generator=g)
        inp = ids.clone()
        inp[pick & (r < 0.8)] = MASK
        rnd = pick & (r >= 0.8) & (r < 0.9)
        inp[rnd] = torch.randint(4, 29, (int(rnd.sum()),), generator=g)
        batches.append((inp.cuda(), attn.cuda(), labels.cuda()))
    return batches


def make_model():
    from transformers import EsmConfig, EsmForMaskedLM

    torch.manual_seed(0)
    cfg = EsmConfig(vocab_size=len(TOKS), hidden_size=a.hidden, num_hidden_layers=a.layers,
                    num_attention_heads=a.heads, intermediate_size=4 * a.hidden,
                    max_position_embeddings=a.max_len + 4, position_embedding_type="rotary",
                    pad_token_id=PAD, mask_token_id=MASK, token_dropout=False,
                    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                    attn_implementation={"flex": "flex_attention"}.get(a.attn, a.attn))
    return EsmForMaskedLM(cfg).cuda().train()


def pad(batches):
    """K=1: every batch to (max B, max L) over the run; extra rows / columns are padding (masked, no labels)."""
    B = max(x.shape[0] for x, _, _ in batches)
    L = max(x.shape[1] for x, _, _ in batches)
    out = []
    for x, m, y in batches:
        xp = x.new_full((B, L), PAD)
        xp[: x.shape[0], : x.shape[1]] = x
        mp = m.new_zeros(B, L)
        mp[: m.shape[0], : m.shape[1]] = m
        # a fully padded row would make every key masked; give it one visible token
        mp[x.shape[0]:, 0] = 1
        yp = y.new_full((B, L), -100)
        yp[: y.shape[0], : y.shape[1]] = y
        out.append((xp, mp, yp))
    return out


def main():
    batches = load()
    shapes = [tuple(x.shape) for x, _, _ in batches]
    toks = [b * l for b, l in shapes]
    B = max(s[0] for s in shapes)
    L = max(s[1] for s in shapes)
    print(f"{len(batches)} batches, {len(set(shapes))} distinct (B, L); B {min(s[0] for s in shapes)}..{B}, "
          f"L {min(s[1] for s in shapes)}..{L}; padded to ({B}, {L}) the token count is {B * L / (sum(toks) / len(toks)):.1f}x the mean, "
          f"attention work (B*L^2) {B * L * L / (sum(b * l * l for b, l in shapes) / len(shapes)):.1f}x")

    def make():
        m = make_model()
        opt = torch.optim.AdamW(m.parameters(), lr=4e-4)

        def step(f, b):
            x, mask, y = b
            with harness.amp():
                loss = f(input_ids=x, attention_mask=mask, labels=y).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            return loss.detach()

        return m, step

    harness.run(make, batches, a.modes.split(","), warm=a.warm, label=f"esm-{a.attn}", pad=pad)


if __name__ == "__main__":
    main()
