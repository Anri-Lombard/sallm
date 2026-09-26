#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-15%3
#SBATCH --job-name=eval-llama252
#SBATCH --mail-type=FAIL,END

set -euo pipefail

source_repo="${SALLM_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
checkpoint="${SALLM_EVAL_CHECKPOINT:-anrilombard/sallm-llama-252m}"
result_prefix="${SALLM_EVAL_RESULT_PREFIX:-llama_252m_base_0shot}"
run_prefix="${SALLM_EVAL_RUN_PREFIX:-llama-252m-base-0shot}"
num_fewshot="${SALLM_EVAL_NUM_FEWSHOT:-0}"
root="${SCRATCH:-/scratch/$USER}/masters/sallm/results/eval"

if [[ ! "$num_fewshot" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SALLM_EVAL_NUM_FEWSHOT must be a non-negative integer." >&2
  exit 2
fi

if [[ "$num_fewshot" != 0 && "$index" -ge 14 ]]; then
  echo "ERROR: few-shot generation tasks require the dedicated generation runner." >&2
  exit 2
fi

labels=(
  masakhanews_all masakhaner_all masakhapos_all sib_all injongointent_all
  sa_general_all belebele_afr belebele_eng belebele_sot belebele_ssw
  belebele_tsn belebele_tso belebele_xho belebele_zul t2x_xho afrihg_all
)
configs=(
  run_mamba_masakhanews_all run_mamba_masakhaner_all run_mamba_masakhapos_all
  run_mamba_sib_all run_mamba_injongointent_all run_mamba_sa_general_all
  run_mamba_belebele_afr run_mamba_belebele_eng run_mamba_belebele_sot
  run_mamba_belebele_ssw run_mamba_belebele_tsn run_mamba_belebele_tso
  run_mamba_belebele_xho run_mamba_belebele_zul run_mamba_t2x_xho
  run_mamba_afrihg_all
)
packs=(
  masakhanews_all_test_chat masakhaner_all masakhapos_all sib_all
  injongointent_all afrimgsm_sa belebele_afr belebele_eng belebele_sot
  belebele_ssw belebele_tsn belebele_tso belebele_xho belebele_zul '' ''
)

label="${labels[$index]}"
config="${configs[$index]}"
pack="${packs[$index]}"
output="$root/${result_prefix}_${label}_r1"

if [[ -s "$output/evaluation_summary.json" ]]; then
  echo "SKIP completed: $label"
  exit 0
fi

args=(
  "eval.eval_model.checkpoint=$checkpoint"
  "++eval.eval_model.tie_word_embeddings=true"
  "++eval.eval_model.peft_adapter=null"
  "++eval.eval_model.merge_lora=false"
  "eval.evaluation.output_dir=$output"
  "eval.wandb.name=eval-${run_prefix}-${label//_/-}-r1"
)

if [[ -n "$pack" ]]; then
  args+=(
    "++eval.evaluation.overrides.${pack}.num_fewshot=${num_fewshot}"
    "++eval.evaluation.overrides.${pack}.batch_size=8"
    "++eval.evaluation.overrides.${pack}.max_batch_size=16"
    "++eval.evaluation.overrides.${pack}.apply_chat_template=false"
  )
fi

if [[ "$label" == masakhanews_all ]]; then
  args+=("eval.evaluation.task_packs=[masakhanews_all_test_chat]")
elif [[ "$label" == sa_general_all ]]; then
  for extra_pack in afrimmlu_sa afrixnli_sa; do
    args+=(
      "++eval.evaluation.overrides.${extra_pack}.num_fewshot=${num_fewshot}"
      "++eval.evaluation.overrides.${extra_pack}.batch_size=8"
      "++eval.evaluation.overrides.${extra_pack}.max_batch_size=16"
      "++eval.evaluation.overrides.${extra_pack}.apply_chat_template=false"
    )
  done
elif [[ "$label" == t2x_xho ]]; then
  args+=(
    "++eval.evaluation.generation_tasks.0.fewshot=0"
    "++eval.evaluation.generation_tasks.0.prompt_format=raw"
  )
elif [[ "$label" == afrihg_all ]]; then
  args+=(
    "++eval.evaluation.generation_tasks.0.fewshot=0"
    "++eval.evaluation.generation_tasks.1.fewshot=0"
    "++eval.evaluation.generation_tasks.0.prompt_format=raw"
    "++eval.evaluation.generation_tasks.1.prompt_format=raw"
  )
fi

export SALLM_SKIP_MAMBA_KERNEL_CHECK=1
command=(bash "$source_repo/scripts/launch_evaluation.sh" "eval/$config" "${args[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  "${command[@]}"
fi
