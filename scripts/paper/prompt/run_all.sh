#!/bin/bash
# usage: run_all.sh GPU "arch:mode arch:mode ..." [packs...]
set -uo pipefail
here=/scratch/alombard/sallm_snapshots/general-prefix-fix-20260924
out=/scratch/alombard/sallm/results/general_prefix_fix_20260924
export CUDA_VISIBLE_DEVICES="$1"; shift
jobs="$1"; shift
export HF_HOME=/scratch/alombard/sallm/hf HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1
export FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda
export PYTHONPATH=/scratch/alombard/sallm_snapshots/downstream-generation-20260914-v8/src/main:/scratch/alombard/sallm_snapshots/full-matrix-execution-20260916-v1
export TMPDIR=/scratch/alombard/sallm/tmp/general_prefix_fix
mkdir -p "$TMPDIR" "$here/logs" "$out"
cd "$here"
for job in $jobs; do
  arch="${job%%:*}"; mode="${job##*:}"
  tag="${arch}-${mode}${SUFFIX:-}"
  start=$(date +%s)
  echo "START $tag gpu=$CUDA_VISIBLE_DEVICES $(date -Is)" >> logs/status.txt
  if /scratch/alombard/sallm/.venv/bin/python run_prefix_eval.py --arch "$arch" --mode "$mode" --output "$out/$tag" ${@:+--packs "$@"} > "logs/$tag.log" 2>&1; then
    echo "DONE $tag $(( $(date +%s) - start ))s $(date -Is)" >> logs/status.txt
  else
    echo "FAIL $tag $(( $(date +%s) - start ))s $(date -Is)" >> logs/status.txt
  fi
done
