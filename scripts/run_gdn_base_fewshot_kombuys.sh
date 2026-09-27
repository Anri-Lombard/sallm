#!/bin/bash

set -euo pipefail

MODE="${1:?Usage: $0 <core|tail> <gpu> [shots]}"
GPU="${2:?Usage: $0 <core|tail> <gpu> [shots]}"
SHOTS="${3:-2 3}"

cd /scratch/alombard/sallm
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="$GPU"
export SCRATCH=/scratch/alombard
export SALLM_HOME_DIR=/home/alombard
export SALLM_SCRATCH_DIR=/scratch/alombard
export SALLM_REPO_DIR=/scratch/alombard/sallm
export HF_HOME=/scratch/alombard/sallm/hf
export HF_TOKEN_FILE=/scratch/alombard/sallm/hf/token
export HF_TOKEN
HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
export SKIP_FINETUNE_STATE_CHECK=1
export FLA_DISABLE_BACKEND_DISPATCH=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

CHECKPOINT="anrilombard/sallm-gated-deltanet-125m-shallowwide-4x40-20260707"
ROOT="/scratch/alombard/sallm/results/eval"

run_eval() {
  local shot="$1"
  local label="$2"
  local config="$3"
  local pack="$4"
  shift 4

  local output="${ROOT}/gdn_125m_base_${shot}shot_${label}_r1"
  if [[ -s "${output}/evaluation_summary.json" ]]; then
    echo "SKIP completed: ${shot}-shot ${label}"
    return
  fi

  local args=(
    "eval.eval_model.checkpoint=${CHECKPOINT}"
    "++eval.eval_model.tie_word_embeddings=true"
    "++eval.eval_model.peft_adapter=null"
    "++eval.eval_model.merge_lora=false"
    "eval.evaluation.output_dir=${output}"
    "eval.wandb.name=eval-gdn-125m-base-${shot}shot-${label//_/-}-r1"
    "++eval.evaluation.overrides.${pack}.num_fewshot=${shot}"
    "++eval.evaluation.overrides.${pack}.batch_size=1"
    "++eval.evaluation.overrides.${pack}.max_batch_size=1"
  )

  if [[ "$label" == "masakhanews_all" ]]; then
    args+=("eval.evaluation.task_packs=[masakhanews_all_test_chat]")
  fi

  local extra_pack
  for extra_pack in "$@"; do
    args+=(
      "++eval.evaluation.overrides.${extra_pack}.num_fewshot=${shot}"
      "++eval.evaluation.overrides.${extra_pack}.batch_size=1"
      "++eval.evaluation.overrides.${extra_pack}.max_batch_size=1"
    )
  done

  echo "RUN: ${shot}-shot ${label}"
  python -m sallm.main --config-name "eval/${config}" "${args[@]}"
}

run_generation() {
  local shot="$1"
  local label="$2"
  local config="$3"
  local task_count="$4"

  local output="${ROOT}/gdn_125m_base_${shot}shot_${label}_r1"
  if [[ -s "${output}/evaluation_summary.json" ]]; then
    echo "SKIP completed: ${shot}-shot ${label}"
    return
  fi

  local args=(
    "eval.eval_model.checkpoint=${CHECKPOINT}"
    "++eval.eval_model.tie_word_embeddings=true"
    "++eval.eval_model.peft_adapter=null"
    "++eval.eval_model.merge_lora=false"
    "eval.evaluation.output_dir=${output}"
    "eval.wandb.name=eval-gdn-125m-base-${shot}shot-${label//_/-}-r1"
  )

  local index
  for ((index = 0; index < task_count; index++)); do
    args+=(
      "++eval.evaluation.generation_tasks.${index}.fewshot=${shot}"
      "++eval.evaluation.generation_tasks.${index}.fewshot_split=train"
      "++eval.evaluation.generation_tasks.${index}.fewshot_seed=42"
      "++eval.evaluation.generation_tasks.${index}.fewshot_lang_match=true"
      "++eval.evaluation.generation_tasks.${index}.fewshot_template_mode=SAME"
      "++eval.evaluation.generation_tasks.${index}.fewshot_token_budget=1792"
      "++eval.evaluation.generation_tasks.${index}.prompt_headroom_tokens=256"
    )
  done

  echo "RUN: ${shot}-shot ${label}"
  python -m sallm.main --config-name "eval/${config}" "${args[@]}"
}

for shot in $SHOTS; do
  if [[ "$MODE" == "core" ]]; then
    run_eval "$shot" masakhanews_all run_mamba_masakhanews_all masakhanews_all_test_chat
    run_eval "$shot" masakhaner_all run_mamba_masakhaner_all masakhaner_all
    run_eval "$shot" masakhapos_all run_mamba_masakhapos_all masakhapos_all
    run_eval "$shot" sib_all run_mamba_sib_all sib_all
    run_eval "$shot" injongointent_all run_mamba_injongointent_all injongointent_all
    run_eval "$shot" sa_general_all run_mamba_sa_general_all afrimgsm_sa afrimmlu_sa afrixnli_sa
  elif [[ "$MODE" == "tail" ]]; then
    for lang in afr eng sot ssw tsn tso xho zul; do
      run_eval "$shot" "belebele_${lang}" "run_mamba_belebele_${lang}" "belebele_${lang}"
    done
    run_generation "$shot" t2x_xho run_mamba_t2x_xho 1
    run_generation "$shot" afrihg_all run_mamba_afrihg_all 2
  else
    echo "Unknown mode: $MODE" >&2
    exit 2
  fi
done
