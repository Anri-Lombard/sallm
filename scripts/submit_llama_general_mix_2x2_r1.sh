#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env.sh"
set_sallm_cluster_env
cd "$SALLM_REPO_DIR"

submit_train() {
  local config="$1"
  local variant="$2"
  shift 2
  sbatch \
    --parsable \
    --account=l40sfree \
    --partition=l40s \
    --gres=gpu:l40s:2 \
    --job-name="gen-${variant}" \
    scripts/launch_finetune.sh "finetune/${config}" "$@"
}

submit_eval() {
  local dependency="$1"
  local local_checkpoint="$2"
  local hub_checkpoint="$3"
  local variant="$4"
  sbatch \
    --parsable \
    --account=nlpgroup \
    --partition=a100 \
    --qos=nlpgroup \
    --gres=gpu:ampere:1 \
    --dependency="afterok:${dependency}" \
    --job-name="eval-gen-${variant}" \
    scripts/launch_evaluation.sh \
    eval/run_llama_general_full_matrix_r1 \
    "eval.eval_model.checkpoint=[${local_checkpoint},${hub_checkpoint}]" \
    "eval.evaluation.output_dir=${SCRATCH}/masters/sallm/results/eval/llama_general_${variant}_full_matrix_r1" \
    "eval.wandb.name=eval-llama-general-${variant}-full-matrix-r1"
}

current_lora=$(submit_train llama_sa_general_currentmix_control_r1 current-lora)
current_full=$(submit_train \
  llama_sa_general_currentmix_control_r1 current-full \
  peft.method=none \
  training.learning_rate=2.0e-05 \
  "training.output_dir=${SCRATCH}/masters/sallm/checkpoints/ft_llama_125m_general_currentmix_control_r1_fullft" \
  "training.logging_dir=${SCRATCH}/masters/sallm/logs/ft_llama_125m_general_currentmix_control_r1_fullft" \
  training.run_name=llama-125m-general-currentmix-control-r1-fullft \
  wandb.name=llama-125m-general-currentmix-control-r1-fullft)
balanced_lora=$(submit_train llama_sa_general_tokenbalanced_r1 balanced-lora)
balanced_full=$(submit_train \
  llama_sa_general_tokenbalanced_r1 balanced-full \
  peft.method=none \
  training.learning_rate=2.0e-05 \
  "training.output_dir=${SCRATCH}/masters/sallm/checkpoints/ft_llama_125m_general_tokenbalanced_r1_fullft" \
  "training.logging_dir=${SCRATCH}/masters/sallm/logs/ft_llama_125m_general_tokenbalanced_r1_fullft" \
  training.run_name=llama-125m-general-tokenbalanced-r1-fullft \
  wandb.name=llama-125m-general-tokenbalanced-r1-fullft)

current_lora_eval=$(submit_eval \
  "$current_lora" \
  "${SCRATCH}/masters/sallm/checkpoints/ft_llama_125m_general_currentmix_control_r1/final_adapter" \
  "anrilombard/sallm-llama-125m-general-currentmix-control-r1" \
  currentmix_lora)
current_full_eval=$(submit_eval \
  "$current_full" \
  "${SCRATCH}/masters/sallm/checkpoints/ft_llama_125m_general_currentmix_control_r1_fullft/final_merged_model" \
  "anrilombard/sallm-llama-125m-general-currentmix-control-r1-merged" \
  currentmix_fullft)
balanced_lora_eval=$(submit_eval \
  "$balanced_lora" \
  "${SCRATCH}/masters/sallm/checkpoints/ft_llama_125m_general_tokenbalanced_r1/final_adapter" \
  "anrilombard/sallm-llama-125m-general-tokenbalanced-r1" \
  tokenbalanced_lora)
balanced_full_eval=$(submit_eval \
  "$balanced_full" \
  "${SCRATCH}/masters/sallm/checkpoints/ft_llama_125m_general_tokenbalanced_r1_fullft/final_merged_model" \
  "anrilombard/sallm-llama-125m-general-tokenbalanced-r1-merged" \
  tokenbalanced_fullft)

printf 'current_lora=%s eval=%s\n' "$current_lora" "$current_lora_eval"
printf 'current_full=%s eval=%s\n' "$current_full" "$current_full_eval"
printf 'balanced_lora=%s eval=%s\n' "$balanced_lora" "$balanced_lora_eval"
printf 'balanced_full=%s eval=%s\n' "$balanced_full" "$balanced_full_eval"
