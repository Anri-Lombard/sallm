#!/bin/bash
# usage: stage_test.sh <label> [--adapter PATH] [--out PATH]
# Scores TEST with the General protocol on the checkpoint in $R/val/<label>/selection.json -> $R/test/<label>.json,
# or on --adapter PATH (parity checks) -> $R/test/adapter_override/<label>__<tree12>.json. See kit/reselect.py.
set -euo pipefail
label="${1:?label}"; shift
root=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925
export PYTHONDONTWRITEBYTECODE=1
echo "stage_test $label $* host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
/home/lmbanr001/masters/sallm/.venv/bin/python "$root/jobs/kit/reselect.py" test "$label" "$@"
echo "stage_test $label end=$(date -Is)"
