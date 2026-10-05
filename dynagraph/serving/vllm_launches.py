"""Launch declarations for vLLM operators (`torch.utils._capture_launch`).

What the operator's owner would ship next to the operator. Two layers: the
kernel-level op (`_C_cache_ops::reshape_and_cache_flash`), whose launch
follows from its arguments alone, and the op the compiled model actually
calls (`vllm::unified_kv_cache_update`), which reads the engine's per-step
context and hands the rest down.
"""
import torch
from torch.utils import _capture_launch as cl


def _reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping,
                             kv_cache_dtype, k_scale, v_scale):
    # csrc/libtorch_stable/cache_kernels.cu, reshape_and_cache_flash: the
    # FP8/auto path (nvfp4 has its own dispatch and is not declared here).
    if kv_cache_dtype.startswith("nvfp4"):
        raise cl.Mismatch("nvfp4 kv cache is not declared")
    num_tokens = slot_mapping.size(0)
    num_heads, head_size = key.size(1), key.size(2)
    if num_tokens == 0:
        return []
    return [cl.Launch(
        kernel=("reshape_and_cache_flash", key.dtype, key_cache.dtype, kv_cache_dtype),
        grid=(num_tokens, 1, 1),
        block=(min(num_heads * head_size, 512), 1, 1),
        smem=0,
        args=(key, value, key_cache, value_cache, slot_mapping,
              key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
              key.stride(0), value.stride(0), num_heads, head_size,
              key_cache.size(1), k_scale, v_scale,
              1 if k_scale.numel() > 1 else 0),
    )]


cl.register("_C_cache_ops::reshape_and_cache_flash", _reshape_and_cache_flash,
            prepare=lambda *a, **k: None)   # an AOT kernel: nothing to set up


def _kv_context(layer_name):
    from vllm.model_executor.layers.attention.attention import (
        _resolve_layer_name, get_attention_context)
    _, layer, kv_cache, slot_mapping = get_attention_context(_resolve_layer_name(layer_name))
    return layer, kv_cache, slot_mapping


def _unified_kv_cache_update(key, value, layer_name):
    # vllm/model_executor/layers/attention/attention.py and the FlashAttention
    # backend's do_kv_cache_update.
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    layer, kv_cache, slot_mapping = _kv_context(layer_name)
    if slot_mapping is None:
        return []
    impl = layer.impl
    if type(impl) is not FlashAttentionImpl:
        raise cl.Mismatch(f"{type(impl).__name__}.do_kv_cache_update is not declared")
    key_cache, value_cache = kv_cache.transpose(1, 2).split(impl.head_size, dim=-1)
    return cl.launches_of("_C_cache_ops::reshape_and_cache_flash", key, value,
                          key_cache, value_cache, slot_mapping, impl.kv_cache_dtype,
                          layer._k_scale, layer._v_scale)


def _kv_key(key, value, layer_name):
    # The engine's per-step state this reads: where the slot mapping lives and
    # how many tokens it covers. The cache and the scales are per layer and
    # fixed for the engine's life.
    _, _, slot_mapping = _kv_context(layer_name)
    return None if slot_mapping is None else (slot_mapping.data_ptr(), slot_mapping.size(0))


def _kv_sources(key, value, layer_name):
    layer, kv_cache, _ = _kv_context(layer_name)
    return [kv_cache, layer._k_scale, layer._v_scale]


# Every layer launches the same kernel up to its own cache and scales.
cl.register("vllm::unified_kv_cache_update", _unified_kv_cache_update,
            key=_kv_key, prepare=lambda *a, **k: None,
            template=lambda key, value, layer_name: _kv_key(key, value, layer_name),
            sources=_kv_sources)


# ---------------------------------------------------------------- attention
# FlashAttention 3's launch is a CUTLASS params struct filled by its C++ host
# code, and the kernel is picked by heuristics it does not expose, so the
# declaration is that host code itself: `recorded` captures one call on a side
# stream (host code runs, nothing reaches the GPU).
def _attn_op():
    return torch.ops.vllm.unified_attention_with_output.default


def _attention_launches(*args, **kwargs):
    return _recorded_attention(*args, **kwargs)


def _attn_structure(query, key, value, output, layer_name, *rest, **kw):
    # `_attn_key` without what differs between layers.
    k = _attn_key(query, key, value, output, layer_name, *rest, **kw)
    return None if k is None else k[:4] + k[5:]


def _attn_sources(query, key, value, output, layer_name, *rest, **kw):
    # Per-layer state the kernels are handed besides the arguments.
    from vllm.model_executor.layers.attention.attention import (
        _resolve_layer_name, get_attention_context)
    _, layer, kv_cache, _ = get_attention_context(_resolve_layer_name(layer_name))
    return [kv_cache, layer._k_scale, layer._v_scale, getattr(layer.impl, "sinks", None)]


_recorded_attention = cl.recorded(lambda *a, **k: _attn_op()(*a, **k),
                                  template_key=_attn_structure, sources=_attn_sources)
_attention_launches.records = True


def _attn_key(query, key, value, output, layer_name, *rest, **kw):
    # What the launch depends on besides the operands: the engine's per-step
    # metadata. Under the graph-safe contract (the one vLLM keeps for its own
    # full CUDA graphs) every metadata tensor lives at a fixed address, the
    # split count is capped, and the per-request lengths reach the kernel
    # through those tensors -- so the launch follows from the token counts
    # alone. Outside it, max_seq_len steers FA3's split heuristic and joins
    # the key.
    from vllm.model_executor.layers.attention.attention import (
        _resolve_layer_name, get_attention_context)
    md, layer, kv_cache, _ = get_attention_context(_resolve_layer_name(layer_name))
    if md is None:
        return None
    sm = md.scheduler_metadata
    safe = md.max_num_splits > 0 and sm is not None
    return (
        md.num_actual_tokens, md.max_query_len, md.max_num_splits, md.use_cascade,
        kv_cache.data_ptr(), md.query_start_loc.data_ptr(), md.seq_lens.data_ptr(),
        md.block_table.data_ptr(),
        None if sm is None else (sm.data_ptr(), sm.numel()),
        None if safe else md.max_seq_len,
    )


cl.register("vllm::unified_attention_with_output", _attention_launches,
            key=_attn_key, prepare=lambda *a, **k: None)


def graph_safe_metadata(max_tokens):
    """The engine side of the contract: FA3 metadata the way vLLM builds it
    for its own full CUDA graphs (a resident scheduler_metadata buffer, a capped
    split count), for batches up to `max_tokens`, without vLLM capturing any
    graph itself."""
    from vllm.v1.attention.backends import flash_attn as fa
    from vllm.utils.math_utils import round_up
    init = fa.FlashAttentionMetadataBuilder.__init__

    def patched(self, kv_cache_spec, layer_names, vllm_config, device, *a, **k):
        init(self, kv_cache_spec, layer_names, vllm_config, device, *a, **k)
        if self.use_full_cuda_graph or not self.aot_schedule:
            return
        self.use_full_cuda_graph = True
        self.max_cudagraph_size = max_tokens
        n = max(vllm_config.scheduler_config.max_num_seqs, max_tokens)
        self.scheduler_metadata = torch.zeros(1 + round_up(n, 4) * 4, dtype=torch.int32,
                                              device=self.device)
        self.max_num_splits = self.attention_config.flash_attn_max_num_splits_for_cuda_graph

    fa.FlashAttentionMetadataBuilder.__init__ = patched


# ------------------------------------------------- attention, described by FA3
# The same declaration through the library's own describe entry
# (`_fa3d_C.fwd_describe`, a build of vllm-flash-attn whose launch points
# write into a sink instead of launching; `$DG_DEPS/fa3d-src/hopper/launch_sink.h`).
# vLLM's attention Python runs unchanged with FA3's `fwd` swapped for it, so
# what comes back is FA3's own answer for this call: no capture, no GPU.
_SINK = None
DESCRIBE_TIME = [0.0, 0]


def _fa3d():
    import glob
    import os
    if not hasattr(torch.ops, "_fa3d_C") or not hasattr(torch.ops._fa3d_C, "fwd_describe"):
        so = glob.glob(os.path.join(os.environ.get("FA3D_DIR", os.path.join(os.environ.get("DG_DEPS", "/workspace/_deps"), "fa3d-src/build")), "_fa3d_C*.so"))
        torch.ops.load_library(so[0])
    return torch.ops._fa3d_C.fwd_describe


def _parse_sink(buf, owner):
    # Only the records written, not the whole buffer.
    arr = buf.numpy()
    n, at, out = int(arr[0]), 1, []
    for _ in range(n):
        if arr[at] != 0:
            raise cl.Mismatch("FA3 described a memset; not declared yet")
        func, gx, gy, gz, bx, by, bz, cx, cy, cz, smem, pdl, nbytes = arr[at + 1:at + 14].tolist()
        at += 14
        nw = (nbytes + 7) // 8
        raw = arr[at:at + nw].tobytes()[:nbytes]
        at += nw
        cluster = None if (cx, cy, cz) == (1, 1, 1) else (cx, cy, cz)
        out.append(cl.Launch(func, (gx, gy, gz), (bx, by, bz), smem, (raw,), cluster, owner))
    return out


def _attention_described(*args, **kwargs):
    global _SINK
    describe = _fa3d()
    if _SINK is None:
        _SINK = torch.zeros(1 << 16, dtype=torch.int64)
    held = []

    def fwd(*a, **k):
        import time
        t0 = time.perf_counter()
        r = describe(_SINK, *a, **k)
        DESCRIBE_TIME[0] += time.perf_counter() - t0
        DESCRIBE_TIME[1] += 1
        DESCRIBE_TIME.append(time.perf_counter() - t0)
        held.append(r)
        return r

    ns = torch.ops._vllm_fa3_C
    real = ns.fwd
    setattr(ns, "fwd", fwd)
    try:
        _attn_op()(*args, **kwargs)
    finally:
        setattr(ns, "fwd", real)
    if not held:
        return []   # no metadata this call (a profiling run): nothing launched
    if len(held) != 1:
        raise cl.Mismatch(f"attention made {len(held)} FA3 calls")
    return _parse_sink(_SINK, held)


def use_described_attention():
    cl.register("vllm::unified_attention_with_output", _attention_described,
                key=_attn_key, prepare=lambda *a, **k: None, exact=False,
                template=_attn_structure, sources=_attn_sources)
