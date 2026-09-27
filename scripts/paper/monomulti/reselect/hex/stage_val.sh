#!/bin/bash
# usage: stage_val.sh <label>
# Scores VALIDATION for every $R/train/<label>/checkpoint-* (ascending step, skips ones already in $R/val/<label>/),
# then writes $R/val/<label>/selection.json. While training is still running (no final_adapter) it refuses unless
# RESELECT_ALLOW_PARTIAL=1: then it scores only checkpoints that have trainer_state.json and writes selection.partial.json.
# Intent val_source: upstream dev.jsonl for {mamba2,mzansilm}_intent_mono_{sot,xho,zul}, train carve otherwise. See kit/reselect.py.
set -euo pipefail
label="${1:?label}"
root=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925
export PYTHONDONTWRITEBYTECODE=1
echo "stage_val $label host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
/home/lmbanr001/masters/sallm/.venv/bin/python "$root/jobs/kit/reselect.py" val "$label"
echo "stage_val $label end=$(date -Is)"
