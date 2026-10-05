#!/usr/bin/env python3
r"""A real variable-length training step on two cards: transformers Llama (small config) + DDP or FSDP2, forward + backward + AdamW,
with seqlen changing every step and different on each of the two ranks.

    CUDA_VISIBLE_DEVICES=3,5 torchrun --standalone --nproc_per_node=2 probe_train_ddp.py --parallel ddp
    CUDA_VISIBLE_DEVICES=3,5 torchrun --standalone --nproc_per_node=2 probe_train_ddp.py --parallel fsdp [--trace-hooks]

DynaGraph off and on each run the same shape stream (same seed, same data), and each rank compares on its own: recording count, fallback tags,
whether each step's loss matches, per-step time; rank 0 summarizes. DDP's all-reduce lives in an autograd hook, outside the region; with FSDP2's default
(`skip_fsdp_hooks=True`) the hooks are outside the region and each layer is one frame called N times (the multi-GPU version of lane); `--trace-hooks`
traces all-gather / reduce-scatter into the graph, where they become NCCL child sites.
"""
import argparse, os, sys, time, logging
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
import torch, torch._dynamo, torch._inductor.config as ic
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--parallel", default="ddp", choices=("ddp", "fsdp"))
ap.add_argument("--trace-hooks", action="store_true", help="FSDP2: trace the hooks into the graph (skip_fsdp_hooks=False)")
ap.add_argument("--passes", type=int, default=3)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--only", default="", help="off / on")
ap.add_argument("--noise", action="store_true", help="DG off twice: the drift the backward and the collectives alone give")
a = ap.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)

logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("torch._inductor.dynagraph"); lg.setLevel(logging.INFO)
tags: list[str] = []
class _Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "fallback [" in m:
            tags.append(m.split("[", 1)[1].split("]", 1)[0] + (": " + m.split("]: ", 1)[1][:80] if "]: " in m else ""))
lg.addHandler(_Grab())

SHAPES = [64, 128, 96, 256, 32, 160, 128, 224]
V = 1024

def make_model():
    torch.manual_seed(0)
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=V, hidden_size=a.hidden, intermediate_size=a.hidden * 2,
                      num_hidden_layers=a.layers, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=512, attn_implementation="sdpa", use_cache=False)
    return LlamaForCausalLM(cfg)

def batch(L, step):
    g = torch.Generator(device="cuda"); g.manual_seed(1000 + step * world + rank)
    return torch.randint(0, V, (a.batch, L), device="cuda", generator=g)

def wrap(m):
    if a.parallel == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(m, device_ids=[rank]), m
    from torch.distributed.fsdp import fully_shard
    for layer in m.model.layers:
        fully_shard(layer)
    fully_shard(m)
    return m, m

def run(dynagraph: bool):
    torch._dynamo.reset()
    ic.triton.dynagraph = dynagraph
    ic.triton.dynagraph_extern_child = True
    ic.force_disable_caches = True
    if a.parallel == "fsdp" and hasattr(torch._dynamo.config, "skip_fsdp_hooks"):
        torch._dynamo.config.skip_fsdp_hooks = not a.trace_hooks
    tags.clear()
    from torch._inductor import cudagraph_trees as ct
    n_rec = {"v": 0}
    orig = ct.CUDAGraphNode.__init__
    def rec(self, *x, **kw):
        n_rec["v"] += 1
        return orig(self, *x, **kw)
    ct.CUDAGraphNode.__init__ = rec
    try:
        m = make_model().cuda().train()
        wrapped, inner = wrap(m)
        opt = torch.optim.AdamW(inner.parameters(), lr=1e-3)
        f = torch.compile(wrapped, dynamic=True, mode="reduce-overhead")
        losses, times = [], []
        stream = [SHAPES[(i + rank * 3) % len(SHAPES)] for i in range(a.passes * len(SHAPES))]
        for step, L in enumerate(stream):
            ids = batch(L, step)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = f(input_ids=ids, labels=ids)
            out.loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
            losses.append(out.loss.item())
        return n_rec["v"], losses, times, list(tags)
    finally:
        ct.CUDAGraphNode.__init__ = orig

res = {}
P = len(SHAPES)
for mode in ("off", "on"):
    if a.only and mode != a.only:
        continue
    n, losses, times, tg = run(mode == "on" and not a.noise)
    res[mode] = (n, losses, times, tg)
    meds = [sorted(times[i:i + P])[len(times[i:i + P]) // 2] * 1e3 for i in range(0, len(times), P)]
    line = f"  rank {rank} {a.parallel} DG={mode:<3} recordings {n:<3} median step per pass {' / '.join(f'{x:.1f}' for x in meds)} ms  loss[-1] {losses[-1]:.4f}"
    if tg:
        seen = {}
        for t in tg: seen[t.split(":")[0]] = seen.get(t.split(":")[0], 0) + 1
        line += f"\n    tags {seen}"
    print(line, flush=True)
    dist.barrier()
ok = True
if "off" in res and "on" in res:
    lo, ln = res["off"][1], res["on"][1]
    diffs = [abs(x - y) / max(abs(x), 1e-6) for x, y in zip(lo, ln)]
    worst = max(diffs)
    ok = worst < 1e-5
    print(f"  rank {rank} max per-step loss rel diff {worst:.2e}  recordings {res['off'][0]} -> {res['on'][0]}", flush=True)
    if worst > 1e-6:
        print("  per step: " + " ".join(f"{d:.0e}" for d in diffs), flush=True)
flags = torch.tensor([int(ok)], device="cuda")
dist.all_reduce(flags, op=dist.ReduceOp.MIN)
if rank == 0:
    print("  all passed" if int(flags.item()) else "  FAILED", flush=True)
dist.destroy_process_group()
sys.exit(0 if int(flags.item()) else 1)
