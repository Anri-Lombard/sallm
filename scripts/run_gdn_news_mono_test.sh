#!/bin/bash
#SBATCH --account=nlpgroup80
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere80:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1%1
#SBATCH --job-name=gdn-news-mono-test
#SBATCH --mail-type=FAIL,END

set -euo pipefail

languages=(eng xho)
index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
language="${languages[$index]}"
root="${SCRATCH:-/scratch/${USER}}/masters/sallm"
pack="masakhanews_${language}_test_chat"

exec bash "${SALLM_REPO_DIR:-$HOME/masters/sallm}/scripts/launch_evaluation.sh" \
  eval/run_mamba_masakhanews_"${language}" \
  "eval_model.checkpoint=anrilombard/sallm-gated-deltanet-125m-shallowwide-4x40-20260707" \
  "++eval_model.peft_adapter=${root}/checkpoints/gdn_news_mono_repair_20260730/${language}/final_adapter" \
  "++eval_model.merge_lora=false" \
  "++eval_model.tie_word_embeddings=true" \
  "evaluation.task_packs=[${pack}]" \
  "evaluation.output_dir=${root}/results/eval/gdn_news_mono_repair_20260730/${language}" \
  "wandb.name=gdn-news-mono-${language}-repair-test-20260730" \
  "++evaluation.overrides.${pack}.num_fewshot=0" \
  "++evaluation.overrides.${pack}.batch_size=1" \
  "++evaluation.overrides.${pack}.max_batch_size=1"
