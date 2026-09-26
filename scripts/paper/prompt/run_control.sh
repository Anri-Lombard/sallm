#!/bin/bash
set -uo pipefail
here=/scratch/alombard/sallm_snapshots/general-prefix-fix-20260924
out=/scratch/alombard/sallm/results/general_prefix_fix_20260924_control
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/scratch/alombard/sallm/hf HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1
export FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda
export PYTHONPATH=/scratch/alombard/sallm_snapshots/downstream-generation-20260914-v8/src/main:/scratch/alombard/sallm_snapshots/full-matrix-execution-20260916-v1
export TMPDIR=/scratch/alombard/sallm/tmp/general_prefix_fix
cd "$here"
for mode in train current; do
  tag=control-multi-intent-mzansilm-$mode; start=$(date +%s)
  echo "START $tag gpu=0 $(date -Is)" >> logs/status.txt
  if /scratch/alombard/sallm/.venv/bin/python run_prefix_eval.py --spec units_control.json --unit-id control-multi-intent-mzansilm --arch mzansilm --mode $mode --packs injongointent_all --output "$out/$tag" > "logs/$tag.log" 2>&1; then
    echo "DONE $tag $(( $(date +%s) - start ))s $(date -Is)" >> logs/status.txt
  else
    echo "FAIL $tag $(( $(date +%s) - start ))s $(date -Is)" >> logs/status.txt
  fi
done
