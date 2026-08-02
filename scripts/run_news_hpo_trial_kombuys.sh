#!/bin/bash

set -euo pipefail

architecture="${1:?Usage: $0 <llama252|gdn> <trial> [gpu] [microbatch] [gradient_accumulation]}"
trial="${2:?Usage: $0 <llama252|gdn> <trial> [gpu] [microbatch] [gradient_accumulation]}"
gpu="${3:-0}"
microbatch="${4:-8}"
gradient_accumulation="${5:-4}"
learning_rates=(3.0e-05 8.0e-05 1.5e-04 3.0e-04)

if (( trial < 0 || trial >= ${#learning_rates[@]} )); then
  echo "Trial must be between 0 and 3." >&2
  exit 1
fi

case "$architecture" in
  llama252) config="finetune/llama_252m_news_all_hpo_r1" ;;
  gdn) config="finetune/gdn_news_all_hpo_r1" ;;
  *) echo "Unknown architecture: $architecture" >&2; exit 1 ;;
esac

repo="${SALLM_REPO_DIR:-/scratch/alombard/sallm}"
scratch="${SCRATCH:-/scratch/alombard}"
output_root="${scratch}/masters/sallm"
run_id="${architecture}-news-hpo-r1-trial-${trial}-kombuys-gpu${gpu}"

export CUDA_VISIBLE_DEVICES="$gpu"
export SCRATCH="$scratch"
export SALLM_REPO_DIR="$repo"
export HF_HOME="${HF_HOME:-$repo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export HF_TOKEN="${HF_TOKEN:-$(cat "$repo/hf/token" 2>/dev/null || true)}"
export TOKENIZERS_PARALLELISM=true
export WANDB_DIR="${output_root}/wandb"
export WANDB_CACHE_DIR="${scratch}/.cache/wandb"
export WANDB_CONFIG_DIR="${scratch}/.config/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRITON_CACHE_DIR="${scratch}/.triton/cache"
export PYTORCH_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export HYDRA_FULL_ERROR=1

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" \
  "$TRITON_CACHE_DIR" "${output_root}/logs/news_hpo_r1"

cd "$repo"
source .venv/bin/activate

resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args+=("+finetune.training.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}")
fi

exec python -m sallm.main --config-name "$config" \
  "finetune.training.learning_rate=${learning_rates[$trial]}" \
  "finetune.training.per_device_train_batch_size=${microbatch}" \
  "finetune.training.gradient_accumulation_steps=${gradient_accumulation}" \
  "finetune.training.output_dir=${output_root}/checkpoints/news_hpo_r1/${architecture}/trial_${trial}_kombuys_gpu${gpu}" \
  "finetune.training.logging_dir=${output_root}/logs/news_hpo_r1/${architecture}/trial_${trial}_kombuys_gpu${gpu}" \
  "finetune.training.run_name=${run_id}" \
  "finetune.wandb.name=${run_id}" \
  "${resume_args[@]}"
