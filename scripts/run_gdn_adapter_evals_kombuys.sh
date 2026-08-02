#!/bin/bash

set -euo pipefail

cd /scratch/alombard/sallm
export CUDA_VISIBLE_DEVICES=0
export SALLM_HOME_DIR=/home/alombard
export SALLM_SCRATCH_DIR=/scratch/alombard
export SALLM_REPO_DIR=/scratch/alombard/sallm
export SALLM_EVAL_OUTPUT_ROOT=/scratch/alombard/sallm/results/eval
export SKIP_FINETUNE_STATE_CHECK=1
export HF_TOKEN_FILE=/scratch/alombard/sallm/hf/token

for index in $(seq 0 28); do
  echo "=== GDN adapter evaluation $index/28 ==="
  SLURM_ARRAY_TASK_ID="$index" bash scripts/launch_gdn_adapter_eval_array.sh
done
