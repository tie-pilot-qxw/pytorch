# DynaGraph fallback and external attention state

DynaGraph must not silently hand an unsupported dynamic-region call to ordinary
Inductor CUDA graph trees. This can bypass external launch state needed for
correct replay. The current policy raises at that handoff.

## Why ordinary replay was unsafe here

The compiled vLLM model calls `vllm::unified_attention_with_output`, a custom
operator whose host implementation reads the current vLLM attention context.
FA3 uses that context to choose kernels and construct launch parameters.

The vLLM integration registers a launch description and `_attn_key` with
DynaGraph. The key includes token/query counts, split settings, and metadata
addresses. DynaGraph evaluates it when selecting prepared attention parameters
and updates the graph nodes before replay. A compatible node structure can
still require different parameters. A new node structure triggers DynaGraph's
own capture and validation of the real operator's launches.

Ordinary Inductor graph replay does not evaluate this declaration key. In
`cudagraphify_impl`, the dispatch key consists of Python integer arguments to
the compiled region, typically symbolic sizes. Graph trees additionally manage
input copying, address/storage conditions, lifetimes, and execution paths;
Dynamo guards select the compiled specialization. Those mechanisms do not
account for the external attention context in this integration.

During capture, the attention host code reads the current context and submits
kernel launches. CUDA records them without executing the kernels. During
replay, the recorded GPU operations execute without rerunning that host code.
New tensor contents at stable captured addresses remain visible, but changed
host launch parameters and pointers are not automatically updated.

Thus two calls with the same flattened token count can select an old graph
while requiring different attention parameters. For example, a 32768-token
chunk can contain 16 queries of length 2048, or 16 queries of length 2047 plus
a partial query. Equal flattened sizes do not make the attention contexts
interchangeable.

## Evidence

A small opaque CUDA operator that reads a host-context scalar reproduces the
issue independently of FA3: after forcing DynaGraph build failure, ordinary
replay retains the captured scalar when the context changes.

A forced-fallback Qwen3-0.6B diagnostic also reproduces the mixed-prefill
failure with the corrected attention declarations and a metadata capture budget
of 64. Across 37 forwards, the previous fallback produced 303872 nonfinite
logit elements and top-1 mismatches on forwards 35 and 37. It includes
B=28/L=2047, B=17/L=2048, and B=26/L=2047. Logits were compared against
uncompiled execution after restoring KV and scheduler state.

The earlier declaration fix accounts for profiling output fills and FA3's
scheduler semaphore memset. It eliminates those refusal causes but cannot make
ordinary fallback aware of external launch state.

## Current policy

With DynaGraph enabled, a failed dynamic-region build, runner retirement,
SKIP_SHAPE, or unsuccessful/exhausted REBUILD raises a RuntimeError before
ordinary Inductor graph capture or replay. DynaGraph's own structure-based
recapture and successful rebuilds remain enabled.

This policy applies at the execution handoff, rather than making every internal
refusal diagnostic throw: internal probes can reject one strategy and still
find a supported execution path. The raised error identifies the symbolic-size
key and points to DynaGraph diagnostics for the refusal reason.

Regions without integer inputs do not enter this dynamic runner path; their
existing ordinary capture behavior is unchanged. Explicit capture-size filters
also retain their configured uncaptured execution. Disabling DynaGraph retains
ordinary Inductor behavior. This is a DynaGraph fallback policy, not a general
repair to Inductor's handling of arbitrary external operator state.

A future safe fallback could execute the compiled region without CUDA graph
replay, so its host code reads the context on each call. Supporting ordinary
replay would instead require a contract that guards or updates all external
launch state. Adding metadata keys to a capture cache alone can create many
captures and is not implemented here.

## Regression checks

`dynagraph/probes/test_context_fallback.py` verifies refusal errors before
ordinary capture, successful DynaGraph execution, successful and failed
rebuilds, and ordinary behavior when DynaGraph is disabled. The failure policy
requires no GPU to test.

```sh
LD_LIBRARY_PATH=/home/sunho/pytorch/build/lib .venv/bin/python dynagraph/probes/test_context_fallback.py
```

Local diagnostic scripts, before/after logits, and logs are in
`agent_space/mixed-prefill-debug/`; these scratch artifacts are not committed.
