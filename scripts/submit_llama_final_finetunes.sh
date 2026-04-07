#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_llama_final_finetunes.sh [options]

Options:
  --account <name>       Slurm account (default: l40sfree)
  --partition <name>     Slurm partition (default: l40s)
  --gres <spec>          GPU request (default: gpu:l40s:1)
  --run-tag <tag>        Suffix for run/output names (default: 2026-04-03)
  --config <name>        Submit only this config (repeatable)
  --dry-run              Print sbatch commands without submitting

This submits:
  - monolingual LLaMA finetunes
  - multilingual LLaMA finetunes
  - the general LLaMA model

It intentionally skips:
  - llama_sa_general_balanced
  - llama_sa_general_instruction
  - llama_sa_general_structured
  - llama_sentiment_ft
  - experimental llama_sa_general_all_v2 / v3 configs
EOF
}

SBATCH_ACCOUNT="l40sfree"
SBATCH_PARTITION="l40s"
SBATCH_GRES="gpu:l40s:1"
RUN_TAG="2026-04-03"
DRY_RUN=0
SELECTED_CONFIGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)
      SBATCH_ACCOUNT="${2:-}"
      shift 2
      ;;
    --partition)
      SBATCH_PARTITION="${2:-}"
      shift 2
      ;;
    --gres)
      SBATCH_GRES="${2:-}"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="${2:-}"
      shift 2
      ;;
    --config)
      SELECTED_CONFIGS+=("${2:-}")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
SCRATCH_ROOT="${SCRATCH:-${PROJECT_SCRATCH:-/scratch/${USER:-lmbanr001}}}"

if [[ "$DRY_RUN" -eq 0 ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on HEX or use --dry-run." >&2
  exit 1
fi

MONOLINGUAL_CONFIGS=(
  finetune/llama_afrihg_xho
  finetune/llama_afrihg_zul
  finetune/llama_injongointent_eng
  finetune/llama_injongointent_sot
  finetune/llama_injongointent_xho
  finetune/llama_injongointent_zul
  finetune/llama_ner_tsn
  finetune/llama_ner_xho
  finetune/llama_ner_zul
  finetune/llama_news_eng
  finetune/llama_news_xho
  finetune/llama_pos_tsn
  finetune/llama_pos_xho
  finetune/llama_pos_zul
  finetune/llama_sib_afr
  finetune/llama_sib_eng
  finetune/llama_sib_nso
  finetune/llama_sib_sot
  finetune/llama_sib_xho
  finetune/llama_sib_zul
  finetune/llama_t2x_xho
)

MULTILINGUAL_CONFIGS=(
  finetune/llama_afrihg_all
  finetune/llama_injongointent_all
  finetune/llama_ner_all
  finetune/llama_news_all
  finetune/llama_pos_all
  finetune/llama_sib_all
)

GENERAL_CONFIGS=(
  finetune/llama_sa_general_all
)

MANIFEST_DIR="outputs/final_submissions"
MANIFEST_PATH="${MANIFEST_DIR}/llama_final_submit_${RUN_TAG}.csv"
mkdir -p "$MANIFEST_DIR"

python3 - "$MANIFEST_PATH" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "category",
            "config",
            "job_id",
            "run_name",
            "output_dir",
            "logging_dir",
            "overrides",
        ]
    )
PY

append_manifest() {
  python3 - "$MANIFEST_PATH" "$@" <<'PY'
import csv
import sys

path = sys.argv[1]
row = sys.argv[2:]
with open(path, "a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow(row)
PY
}

common_overrides() {
  local config_name="$1"
  local stem="${config_name##*/}"
  local run_name="${stem}-${RUN_TAG}"
  local output_dir="${SCRATCH_ROOT}/masters/sallm/checkpoints/final_llama/${RUN_TAG}/${stem}"
  local logging_dir="${SCRATCH_ROOT}/masters/sallm/logs/final_llama/${RUN_TAG}/${stem}"
  printf '%s\n' \
    "++finetune.wandb.name=${run_name}" \
    "++finetune.training.run_name=${run_name}" \
    "++finetune.training.output_dir=${output_dir}" \
    "++finetune.training.logging_dir=${logging_dir}" \
    "++finetune.training.eval_strategy=steps" \
    "++finetune.training.eval_steps=500" \
    "++finetune.training.save_strategy=no" \
    "++finetune.training.gradient_checkpointing=false" \
    "++finetune.hub.enabled=false" \
    "++finetune.hub.push_adapter=false" \
    "++finetune.hub.push_merged=false"
}

family_overrides() {
  local config_name="$1"
  local stem="${config_name##*/}"
  case "$stem" in
    llama_injongointent_*|llama_news_*|llama_sib_*)
      printf '%s\n' \
        "++finetune.dataset.max_seq_length=1024" \
        "++finetune.training.learning_rate=0.00026" \
        "++finetune.training.weight_decay=0.15" \
        "++finetune.training.warmup_ratio=0.1" \
        "++finetune.training.lr_scheduler_type=constant_with_warmup" \
        "++finetune.training.gradient_accumulation_steps=4" \
        "++finetune.training.per_device_train_batch_size=2" \
        "++finetune.training.per_device_eval_batch_size=2" \
        "++finetune.training.num_train_epochs=8"
      ;;
    llama_pos_*|llama_ner_*)
      printf '%s\n' \
        "++finetune.dataset.max_seq_length=1024" \
        "++finetune.training.learning_rate=0.00016" \
        "++finetune.training.weight_decay=0.12" \
        "++finetune.training.warmup_ratio=0.1" \
        "++finetune.training.lr_scheduler_type=constant_with_warmup" \
        "++finetune.training.gradient_accumulation_steps=4" \
        "++finetune.training.per_device_train_batch_size=2" \
        "++finetune.training.per_device_eval_batch_size=2" \
        "++finetune.training.num_train_epochs=10"
      ;;
    llama_afrihg_*)
      printf '%s\n' \
        "++finetune.dataset.max_seq_length=1024" \
        "++finetune.training.learning_rate=0.000295" \
        "++finetune.training.weight_decay=0.104" \
        "++finetune.training.warmup_ratio=0.1" \
        "++finetune.training.lr_scheduler_type=constant_with_warmup" \
        "++finetune.training.gradient_accumulation_steps=4" \
        "++finetune.training.per_device_train_batch_size=4" \
        "++finetune.training.per_device_eval_batch_size=4" \
        "++finetune.training.num_train_epochs=4"
      ;;
    llama_t2x_xho)
      printf '%s\n' \
        "++finetune.dataset.max_seq_length=1024" \
        "++finetune.training.learning_rate=0.00015373792550986017" \
        "++finetune.training.weight_decay=0.1529293154724379" \
        "++finetune.training.warmup_ratio=0.0" \
        "++finetune.training.lr_scheduler_type=linear" \
        "++finetune.training.gradient_accumulation_steps=16" \
        "++finetune.training.per_device_train_batch_size=4" \
        "++finetune.training.per_device_eval_batch_size=4" \
        "++finetune.training.num_train_epochs=4"
      ;;
    llama_sa_general_all)
      printf '%s\n' \
        "++finetune.dataset.max_seq_length=1024" \
        "++finetune.training.learning_rate=0.00026" \
        "++finetune.training.weight_decay=0.15" \
        "++finetune.training.warmup_ratio=0.1" \
        "++finetune.training.lr_scheduler_type=constant_with_warmup" \
        "++finetune.training.gradient_accumulation_steps=4" \
        "++finetune.training.per_device_train_batch_size=2" \
        "++finetune.training.per_device_eval_batch_size=2" \
        "++finetune.training.num_train_epochs=3"
      ;;
    *)
      return 1
      ;;
  esac
}

submit_one() {
  local category="$1"
  local config_name="$2"
  local stem="${config_name##*/}"
  local run_name="${stem}-${RUN_TAG}"
  local output_dir="/scratch/lmbanr001/masters/sallm/checkpoints/final_llama/${RUN_TAG}/${stem}"
  local logging_dir="/scratch/lmbanr001/masters/sallm/logs/final_llama/${RUN_TAG}/${stem}"
  local overrides=()
  local line=""

  while IFS= read -r line; do
    overrides+=("$line")
  done < <(common_overrides "$config_name")

  while IFS= read -r line; do
    overrides+=("$line")
  done < <(family_overrides "$config_name")

  local cmd=(
    sbatch
    "--parsable"
    "--account=${SBATCH_ACCOUNT}"
    "--partition=${SBATCH_PARTITION}"
    "--gres=${SBATCH_GRES}"
    scripts/launch_finetune.sh
    "$config_name"
  )
  cmd+=("${overrides[@]}")

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[DRY-RUN] %q ' "${cmd[@]}"
    printf '\n'
    append_manifest "$category" "$config_name" "DRYRUN" "$run_name" "$output_dir" "$logging_dir" "${overrides[*]}"
    return 0
  fi

  local result
  result=$("${cmd[@]}")
  local job_id="${result%%;*}"
  if [[ -z "$job_id" || ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Failed to parse job id for $config_name" >&2
    echo "$result" >&2
    exit 1
  fi

  echo "$config_name -> $job_id"
  append_manifest "$category" "$config_name" "$job_id" "$run_name" "$output_dir" "$logging_dir" "${overrides[*]}"
}

should_submit() {
  local config_name="$1"
  if [[ "${#SELECTED_CONFIGS[@]}" -eq 0 ]]; then
    return 0
  fi

  local selected=""
  for selected in "${SELECTED_CONFIGS[@]}"; do
    if [[ "$selected" == "$config_name" || "$selected" == "${config_name##*/}" ]]; then
      return 0
    fi
  done
  return 1
}

for config in "${MONOLINGUAL_CONFIGS[@]}"; do
  if should_submit "$config"; then
    submit_one monolingual "$config"
  fi
done

for config in "${MULTILINGUAL_CONFIGS[@]}"; do
  if should_submit "$config"; then
    submit_one multilingual "$config"
  fi
done

for config in "${GENERAL_CONFIGS[@]}"; do
  if should_submit "$config"; then
    submit_one general "$config"
  fi
done

echo "Manifest written to ${MANIFEST_PATH}"
