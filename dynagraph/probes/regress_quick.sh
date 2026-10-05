#!/bin/bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../setup/use_main.sh"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export TORCHINDUCTOR_DYNAGRAPH_UPDATE=${UPDATE:-auto}
LOGDIR=${LOGDIR:-${DG_OUT:-/tmp/dynagraph_out}/_regress_logs}; mkdir -p $LOGDIR
pass=0; fail=0
for t in ${LIST:-test_arena test_arena_e2e test_flag end_to_end test_verify probe_early_out probe_extern_child probe_conv_child probe_branch_space probe_launch_decl probe_launch_triton probe_register}; do
  # Through runpy with the dynagraph logger at INFO, so a "not served" failure
  # leaves its fallback tag in the log.
  timeout 1500 python -c "import logging, runpy, sys; logging.basicConfig(level=logging.WARNING); logging.getLogger('torch._inductor.dynagraph').setLevel(logging.INFO); sys.argv = ['$t.py']; runpy.run_path('$t.py', run_name='__main__')" > $LOGDIR/$t.log 2>&1; rc=$?
  last=$(grep -av "^I0\|^W0\|UserWarning\|warn_once\|^ *return \|^$" $LOGDIR/$t.log | tail -n 1 | cut -c1-110)
  if [ $rc -eq 0 ]; then pass=$((pass+1)); st=PASS; else fail=$((fail+1)); st=FAIL; fi
  printf "%-32s %s rc=%s  %s\n" "$t" "$st" "$rc" "$last"
done
echo "== pass=$pass fail=$fail"
