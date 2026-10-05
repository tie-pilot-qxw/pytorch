#!/bin/bash
# Full DynaGraph regression on one exclusive card. Prints rc + last non-noise line per script.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../setup/use_main.sh"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export TORCHINDUCTOR_DYNAGRAPH_UPDATE=${UPDATE:-auto}
LOGDIR=${LOGDIR:-${DG_OUT:-/tmp/dynagraph_out}/_regress_logs}; mkdir -p "$LOGDIR"
LIST=${LIST:-"test_module test_handle_collection test_arena test_arena_e2e end_to_end test_flag test_verify probe_extern_kernel_fallback probe_multi_symbol probe_grid2d probe_grid2d_symbolic probe_stride_on_symbol probe_multi_partition probe_partition_extern probe_empty_and_tiny probe_early_out probe_extern_child probe_update_persistence probe_conv_child probe_switch_py probe_cond_semantics probe_inplace probe_train probe_noncontig probe_rng_child probe_sdpa probe_grid_types probe_input_patch probe_cat_views probe_unbacked probe_combo probe_user_triton probe_train_step probe_train_varlen probe_train_varlen_fa probe_unbacked_device probe_const_grid_scalar probe_autotuned_user probe_cond_selfcheck probe_unbacked_reduce probe_user_triton_mutate probe_tma probe_tma_real probe_tma_shape probe_graph_budget"}
pass=0; fail=0
for t in $LIST; do
  # Through runpy with the dynagraph logger at INFO, so a "not served" failure
  # leaves its fallback tag in the log.
  timeout 1500 python -c "import logging, runpy, sys; logging.basicConfig(level=logging.WARNING); logging.getLogger('torch._inductor.dynagraph').setLevel(logging.INFO); sys.argv = ['$t.py']; runpy.run_path('$t.py', run_name='__main__')" > "$LOGDIR/$t.log" 2>&1; rc=$?
  last=$(grep -v "^I0\|^W0\|UserWarning\|warn_once\|^ *return \|^$" "$LOGDIR/$t.log" | tail -n 1 | cut -c1-110)
  if [ $rc -eq 0 ]; then pass=$((pass+1)); st=PASS; else fail=$((fail+1)); st=FAIL; fi
  printf "%-32s %s rc=%s  %s\n" "$t" "$st" "$rc" "$last"
done
echo "== pass=$pass fail=$fail"
