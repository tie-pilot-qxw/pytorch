# Elastic parallelism: how much of it can DynaGraph take on (2026-09-19)

It started from a question: elastic parallelism today relies on tricks to get around CUDA Graphs;
can our approach support it in one go?

Material: the paper Xinwei provided (Overleaf `6a0e78889e6d3b1b07334494`, elastic CP on SGLang),
the works it cites, LoongServe / ShiftParallelism / NanoCP / Llumnix,
and the SGLang elastic EP implementation mounted in Xinwei's `xinwei_vllm_omni` container
(`/scr/dataset/xinwei/code/serving/sglang`, already read).

## How others get around it

The paper's own words (`04_Method.tex`, "CUDA Graph compatibility"):

> All layout metadata required by elasticity is maintained as graph-resident
> data. Changing the active parallel width therefore updates runtime mappings
> rather than the captured execution structure.

Concretely, it uses a two-level KV layout: a token is first mapped to a **configuration-independent** virtual bucket
(`bucket(t) = t mod B`, `B = LCM(supported CP widths)`), and the bucket is then mapped to a physical rank
according to the current configuration. A resize only changes the bucket->rank ownership; token->bucket never moves.

**Why this avoids re-recording**: paged attention already addresses indirectly through the block table,
and loop bounds are read from device memory. So the grid stays the same, the kernel parameters stay the same, and only the mapping table in device memory changes.

**Why this is not general**: it requires **the operators to already be written with indirect addressing**. A GEMM's grid genuinely depends on
M, and no mapping table can make it change by itself. So systems on this path either pad shapes to a fixed value,
or give up on cudagraphs where shapes really do change -- Xinwei's vllm-omni does the latter.

The paper itself states a restriction:

> We restrict supported CP configurations to nested power-of-two group sizes

This restriction exists precisely to keep shapes and mappings regular.

## SGLang's elastic EP: the same trick, and its cost is visible

The code is in the SGLang mounted in the `xinwei_vllm_omni` container:
`/scr/dataset/xinwei/code/serving/sglang/python/sglang/srt/elastic_ep/elastic_ep.py`
(backend mooncake, `--elastic-ep-backend mooncake`).

`ElasticEPState.active_ranks` is an **int32 tensor on the device**
(`torch.ones(world_size, dtype=int32, device=cuda)`) that marks which ranks are still alive,
and it is passed straight into the dispatcher:

```python
active_ranks = ElasticEPStateManager.instance().active_ranks
buffer.dispatch(hidden_states, topk_ids, active_ranks,
                self.num_max_dispatch_tokens_per_rank, ...)
```

It is **the same trick** as the paper: put the variable part into a device-memory tensor and do not re-record the graph.
But here the cost of the trick is directly visible -- `num_max_dispatch_tokens_per_rank`
is a **fixed maximum**, i.e. **pad2max along the communication dimension**:
no matter how many tokens there actually are, every dispatch moves the maximum amount.
Once the MoE router gets skewed, this waste becomes considerable.

So "don't use cudagraphs" and "use cudagraphs but pad to the max" are two exits from the same problem:
the former avoids the waste but loses the graph, the latter keeps the graph but does useless work all the time.
**What DynaGraph aims to fill is exactly the middle path: keep a single graph while letting sizes actually change.**

Note that a **boolean-style mask** like `active_ranks` (which ranks are alive) and a
**size** (how many tokens per rank) are two different things: for the former a device-memory tensor is enough,
for the latter a device-memory tensor does not solve it, because the grid and buffer sizes depend on it.
This is yet another instance of the same dividing line.

## vllm-omni's own elasticity: dynamic SP for diffusion, and it indeed does not use cudagraphs

The code is in the `xinwei_vllm_omni` container (**not mounted in; it is in the container's own layer**):
`/workspace/vllm-omni-anon`, entry point `scripts/serve-elastic.sh`.

What it does is **dynamic sequence parallelism for diffusion models**: at denoising step 10, the DiT switches from sp=1 (rank 1)
to sp=2 (ranks 1,2):

```bash
--runtime-v2-dit-step-schedule '[
  {"start":0,"end":10,"group_id":"g_dit_sp1"},
  {"start":10,"end":null,"group_id":"g_dit_sp2"}]'
```

Plus a `GroupFree-Collective` backend built specifically to avoid `torch.distributed.new_group`
("that registration is a logical session") -- because creating a communication group is a collective operation and expensive.

**cudagraph does not appear even once in `vllm_omni/diffusion/runtime_v2/`**
(grep `cuda_?graph` gives zero hits); it only appears on the autoregressive/audio path.

### Why it can only give up on cudagraphs

Three things change at the same time at the switch:

| What changes | Nature |
|---|---|
| Per-rank latent sequence length is **halved** | Shape |
| **Extra ulysses all-to-all** (at sp=1 these nodes do not exist at all) | **Graph topology** |
| Communication group membership changes | Communication |

The key point: **diffusion has no KV cache, and therefore no natural indirect addressing layer like paged attention.**
The trick from the paper and SGLang (moving the variable part into device-memory tensors) **cannot be applied at all** here --
it is all dense GEMM and attention, and the shapes genuinely depend on sequence length.

So this is not implementation laziness; that path does not work for diffusion.
**This workload happens to be exactly the kind that only DynaGraph can handle and "moving metadata" cannot.**

### But there are two hard points

**1. Topology changes are harder than parameter changes.** At sp=1 there are no all-to-all nodes at all.
In principle one could **record one graph for the widest configuration, and when narrower, use `SetEnabled` to turn off the communication nodes
while enlarging the local shapes** -- the planner already has `cudaGraphKernelNodeSetEnabled`.
But this requires the communication nodes to be in the graph and device-updatable, and NCCL nodes are not today.

**2. The more practical one: this path does not go through Inductor at all.**
The DiT uses `torch.compile` only in scattered places such as the autoencoder; `runtime_v2` is a scheduling layer.
**With no compiled artifacts, DynaGraph has nothing to hook into.** To connect it, the DiT would first have to go through `torch.compile`,
and that in itself introduces a pile of shape-specialization problems.

### The cheapest first step

Not a two-GPU experiment, but **single-GPU**: run the sp=1 DiT through
`torch.compile(dynamic=True)` + `TORCHINDUCTOR_DYNAGRAPH=1`,
and see which fallback tags it reports. This step can be done on a single GPU, and it will immediately tell us
how much of a diffusion model's graph DynaGraph cannot handle today.

## Key point: these are two axes with completely different cardinalities

| Axis | Cardinality | Change frequency | What it needs |
|---|---|---|---|
| **Shape** | Unbounded (the human proteome alone has 3058 distinct lengths) | **Per step** | Must be changed on the device side, no host round trip allowed |
| **Parallel width** | Very small (nested power-of-two, CP2/4/8/16, 4 width buckets total) | Minute-scale elastic events | Changing on the host side is sufficient |

**The paper and vllm-omni are dealing with the width axis**, and the width axis can be solved by "moving metadata into device memory".
**DynaGraph fills in the shape axis**, and does not require operator cooperation -- this is a difference in mechanism,
not in degree.

## So how much can be supported

### Already possible

1. **The shape axis itself.** Changes in per-rank token count, sequence length, and batch size
   are exactly what DynaGraph does today, without requiring operators to be written with indirect addressing.
2. **Shape changes caused by width changes.** When CP shrinks from 16 to 8, the per-rank token count doubles --
   in DynaGraph's eyes this is just one shape change.

### Blocked today

1. **Graphs containing NCCL are rejected outright.** `torch.ops._c10d_functional.*` hits
   the `extern-launch` of `unreachable_launch`. **Distributed + DynaGraph has never been run even once.**
2. **The KV cache is not managed by DynaGraph.** It is a persistent pool + block table,
   not an Inductor intermediate buffer, and not in the arena. The paper's bucket layout is still needed --
   the two are complementary, not substitutes.
3. **Inputs must arrive at the maximum shape first.** Static input buffers are allocated from the first-arriving shape times the headroom factor,
   and an input that exceeds this triggers retirement with `input-too-large`. Elastic shrinking happens to make the per-rank token count **larger**,
   so this restriction hurts more in the elastic setting than on a single machine.

## A path nobody uses: modifying the graph on the host side

```c
cudaGraphExecKernelNodeSetParams(exec, node, params)
```

It changes the grid and parameters of any kernel node on an **already-instantiated graph**,
**without needing the device-updatable attribute**, as long as the topology is not touched.
It is in `/usr/local/cuda/include/cuda_runtime_api.h`, and **PyTorch does not use it at all**.

Combined with the table above, the architecture is clear -- **fast path on the device, slow path on the host**:

- Per-step shape changes -> device-side planner (implemented)
- Width changes at each reconfiguration -> host-side `SetParams`; these are minute-scale events, so a host round trip does not matter

Neither path requires operators to be written with indirect addressing; this is the essential difference from the paper's approach.

### Hard points not yet verified

1. **NCCL may switch kernels when the world size changes.** The choice of algorithm/protocol depends on size and topology.
   If the kernel changes, it is no longer a parameter change but a topology change; `SetParams` is not enough, and one needs
   `cudaGraphExecUpdate` to swap the full graph, or one graph per width bucket.
   Fortunately **there are only 4 width buckets**, so one graph per bucket is entirely acceptable -- this is exactly the benefit of a small cardinality.
2. **What is in NCCL's kernel parameters.** When the world size changes, the communicator has to be replaced,
   and the comm state is a device pointer. Whether it can be re-pointed in place has not been checked.
3. **Whether `extern-launch` is the only blocker.** Right now it is rejected already at the `unusable_reason()` stage,
   so capture has never happened, and we do not know whether further problems will show up after it.

## Next step: minimal verification

Two GPUs, one Inductor graph with an all-reduce, answering three things:

1. After temporarily letting `extern-launch` through, does capture succeed? Do the handle counts match?
   (NCCL does not go through the static launcher, so its nodes have **no** handles --
   per the lesson from the `handle-mismatch` entry, the kernel table and the handle table might **both** be missing one entry while
   the counts still match, so this needs to be verified separately.)
2. Whether `cudaGraphExecKernelNodeSetParams` can act on NCCL nodes.
3. After changing the world size, whether the kernel function NCCL picks changes.

This step is very cheap (two GPUs, a few dozen lines), but it decides whether this whole route is viable:
if the answer to #3 is "it changes", then host-side `SetParams` is not enough, and we have to fall back to "one graph per width bucket".

**Note**: this step needs two GPUs, and currently the memory on all eight GPUs is full.

## Positioning of the cited papers

- **LoongServe** (OSDI'24, Elastic Sequence Parallelism) --
  the representative work on changing the degree of parallelism at runtime; elastic sequence parallelism.
- **ShiftParallelism** (2026, Snowflake) -- switches between different parallel configurations.
- **NanoCP** (2026) -- request-level dynamic context parallelism.
- **Llumnix** (OSDI'24) -- request migration and rescheduling, without changing the degree of parallelism.

The paper characterizes this class as "typically treat migration or parallelism
reconfiguration as explicit runtime operations", i.e. they treat reconfiguration as an explicit runtime operation.
**We have none of their code in hand; the above is positioning based only on the paper's citations, and the implementations have not been checked.**
