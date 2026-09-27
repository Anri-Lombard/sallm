#!/bin/bash
# unit = <stage>:<label>, stage in train
set -euo pipefail
unit="$1"
stage="${unit%%:*}"; label="${unit#*:}"
root=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925
case "$stage:$label" in
  train:gdn_*) exec bash "$root/jobs/gdn_train.sh" "$label" ;;
  train:xlstm_*) exec bash "$root/jobs/xlstm_train.sh" "$label" ;;
  train:mamba2_*) exec bash "$root/jobs/mamba2_train.sh" "$label" ;;
  train:mzansilm_*) exec bash "$root/jobs/mzansilm_train.sh" "$label" ;;
  val:*) exec bash "$root/jobs/stage_val.sh" "$label" ;;
  test:*) exec bash "$root/jobs/stage_test.sh" "$label" ;;
  *) [[ -x "$root/jobs/stage_$stage.sh" ]] && exec bash "$root/jobs/stage_$stage.sh" "$label"; echo "unknown unit $unit" >&2; exit 3 ;;
esac
