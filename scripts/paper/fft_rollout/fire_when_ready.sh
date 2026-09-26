#!/bin/bash
# Mac: wait until an architecture's matched-pretraining run has written its final weights, then fire its rollout on HEX.
#   bash fire_when_ready.sh ARCH [CAP=4]      (leave it running, e.g. under nohup; polls every 5 min)
# Final weights = <run dir>/weights/step<total_steps>_tok* (pretrain.py renames it into place when complete), as prep binds them. Does nothing if
# runs/ARCH already exists (launch.sh refuses too). Lanes go in with NICE=1000 so pretraining resubmits outrank them,
# and nothing is launched while any mp-full-* job is pending or while ~/.sallm_fire/HOLD exists (rm it to allow launches).
set -euo pipefail
ARCH="${1:?ARCH}"; CAP="${2:-4}"
M=/scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/runs
R=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926
q() { grep -v -E '^\*\*|^ *\*|AUP|agree|^$' || true; }
while true; do
  run=$(ssh -o ConnectTimeout=20 hex "ls -d $M/full_${ARCH}_wsd_lr* 2>/dev/null" 2>/dev/null | q)
  if (( $(wc -w <<< "$run") == 1 )); then
    final=$(ssh -o ConnectTimeout=20 hex "t=\$(grep -o '\"total_steps\": [0-9]*' $run/run_meta.json | grep -o '[0-9]*\$'); \
      ls -d $run/weights/step\$(printf %06d \$t)_tok* 2>/dev/null | grep -v '\.tmp\$'" 2>/dev/null | q)
    pending=$(ssh -o ConnectTimeout=20 hex 'squeue -u $USER -h -t PD -o %j' 2>/dev/null | grep -c '^mp-full-' || true)
    if [[ -n "$final" && -e ~/.sallm_fire/HOLD ]]; then
      echo "$(date '+%F %T') final weights ready but ~/.sallm_fire/HOLD exists; waiting"
    elif [[ -n "$final" && "$pending" -gt 0 ]]; then
      echo "$(date '+%F %T') final weights ready but an mp-full-* job is pending; waiting (pretraining first)"
    elif [[ -n "$final" ]] && ssh hex "test -s $final/pytorch_model.bin" 2>/dev/null; then
      echo "$(date '+%F %T') final weights: $final -> launching $ARCH with CAP=$CAP"
      ssh hex "NICE=1000 bash $R/code/fft_rollout/launch.sh $ARCH $run $CAP" 2>&1 | q
      exit 0
    fi
  fi
  sleep 300
done
