#!/bin/bash
#SBATCH --account=nlpgroup80
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere80:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1%1
#SBATCH --job-name=gdn-news-mono-repair
#SBATCH --mail-type=FAIL,END

set -euo pipefail

languages=(eng xho)
index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
language="${languages[$index]}"
root="${SCRATCH:-/scratch/${USER}}/masters/sallm"

exec bash "${SALLM_REPO_DIR:-$HOME/masters/sallm}/scripts/launch_finetune.sh" \
  finetune/gdn_news_all_hpo_r1 \
  "dataset.languages=[${language}]" \
  "peft.kwargs.r=32" \
  "peft.kwargs.lora_alpha=64" \
  "training.learning_rate=3.0e-05" \
  "training.output_dir=${root}/checkpoints/gdn_news_mono_repair_20260730/${language}" \
  "training.logging_dir=${root}/logs/gdn_news_mono_repair_20260730/${language}" \
  "training.run_name=gdn-news-mono-${language}-repair-20260730" \
  "wandb.name=gdn-news-mono-${language}-repair-20260730"
