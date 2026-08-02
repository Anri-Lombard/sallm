#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7%2
#SBATCH --job-name=news-hpo-r1
#SBATCH --mail-type=FAIL,END

set -euo pipefail

export SCRATCH="${SCRATCH:-/scratch/${USER}}"

array_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
learning_rates=(3.0e-05 8.0e-05 1.5e-04 3.0e-04)

if (( array_index < 4 )); then
  architecture="llama252"
  config="finetune/llama_252m_news_all_hpo_r1"
  trial="$array_index"
else
  architecture="gdn"
  config="finetune/gdn_news_all_hpo_r1"
  trial="$((array_index - 4))"
fi

export SALLM_JOB_NAME="news-hpo-${architecture}-${trial}"
output_root="${SCRATCH}/masters/sallm"

exec bash "${SALLM_REPO_DIR:-$HOME/masters/sallm}/scripts/launch_finetune.sh" \
  "$config" \
  "training.learning_rate=${learning_rates[$trial]}" \
  "training.output_dir=${output_root}/checkpoints/news_hpo_r1/${architecture}/trial_${trial}" \
  "training.logging_dir=${output_root}/logs/news_hpo_r1/${architecture}/trial_${trial}" \
  "training.run_name=${architecture}-news-hpo-r1-trial-${trial}" \
  "wandb.name=${architecture}-news-hpo-r1-trial-${trial}"
