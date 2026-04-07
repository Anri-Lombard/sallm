#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_llama_final_evals.sh [options]

Options:
  --account <name>       Slurm account (default: l40sfree)
  --partition <name>     Slurm partition (default: l40s)
  --gres <spec>          GPU request (default: gpu:l40s:1)
  --run-tag <tag>        Submission tag used by the finetune manifest (default: 2026-04-03)
  --config <name>        Submit only the named finetune config (repeatable)
  --recover              Submit evals immediately from saved local artifacts when present
  --dry-run              Print sbatch commands without submitting
EOF
}

SBATCH_ACCOUNT="l40sfree"
SBATCH_PARTITION="l40s"
SBATCH_GRES="gpu:l40s:1"
RUN_TAG="2026-04-03"
DRY_RUN=0
RECOVER=0
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
    --recover)
      RECOVER=1
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

MANIFEST_PATH="outputs/final_submissions/llama_final_submit_${RUN_TAG}.csv"
EVAL_MANIFEST_PATH="outputs/final_submissions/llama_final_eval_submit_${RUN_TAG}.csv"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Finetune manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi

python3 - "$EVAL_MANIFEST_PATH" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "config",
            "finetune_job_id",
            "eval_config",
            "eval_job_id",
            "checkpoint",
            "output_dir",
        ]
    )
PY

append_manifest() {
  python3 - "$EVAL_MANIFEST_PATH" "$@" <<'PY'
import csv
import sys

path = sys.argv[1]
row = sys.argv[2:]
with open(path, "a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow(row)
PY
}

map_eval_config() {
  local stem="$1"
  local task=""
  local lang=""

  if [[ "$stem" =~ ^llama_sa_general_(.+)$ ]]; then
    if [[ "$stem" == "llama_sa_general_all" ]]; then
      printf 'run_llama_sa_general_all\n'
      return 0
    fi
    return 1
  elif [[ "$stem" =~ ^llama_([^_]+)_(.+)$ ]]; then
    task="${BASH_REMATCH[1]}"
    lang="${BASH_REMATCH[2]}"
  else
    return 1
  fi

  case "$task" in
    ner) task="masakhaner" ;;
    news) task="masakhanews" ;;
    pos) task="masakhapos" ;;
    sib|injongointent|t2x|afrihg) ;;
    *) return 1 ;;
  esac

  printf 'run_llama_%s_%s\n' "$task" "$lang"
}

tail -n +2 "$MANIFEST_PATH" | while IFS=, read -r category config_name job_id run_name output_dir logging_dir overrides; do
  if [[ "${#SELECTED_CONFIGS[@]}" -gt 0 ]]; then
    matched=0
    for selected in "${SELECTED_CONFIGS[@]}"; do
      if [[ "$config_name" == "$selected" ]]; then
        matched=1
        break
      fi
    done
    if [[ "$matched" -eq 0 ]]; then
      continue
    fi
  fi

  stem="${config_name##*/}"
  eval_cfg="$(map_eval_config "$stem" || true)"
  if [[ -z "$eval_cfg" || ! -f "src/conf/eval/${eval_cfg}.yaml" ]]; then
    echo "skip ${config_name}: no matching eval config"
    continue
  fi

  checkpoint="${output_dir}/final_merged_model"
  if [[ -d "${output_dir}/final_adapter" ]]; then
    checkpoint="${output_dir}/final_adapter"
  elif [[ -d "${output_dir}/final_merged_model" ]]; then
    checkpoint="${output_dir}/final_merged_model"
  fi
  eval_output_dir="${SCRATCH_ROOT}/masters/sallm/results/eval/final_llama/${RUN_TAG}/${stem}"
  eval_wandb_name="eval-${stem}-${RUN_TAG}"

  cmd=(
    sbatch
    "--parsable"
    "--account=${SBATCH_ACCOUNT}"
    "--partition=${SBATCH_PARTITION}"
    "--gres=${SBATCH_GRES}"
    scripts/launch_evaluation.sh
    "eval/${eval_cfg}"
    "++eval.eval_model.checkpoint=${checkpoint}"
    "++eval.evaluation.output_dir=${eval_output_dir}"
    "++eval.wandb.name=${eval_wandb_name}"
  )

  if [[ "$RECOVER" -eq 1 ]]; then
    if [[ ! -d "$checkpoint" ]]; then
      echo "skip ${config_name}: no local artifact at ${checkpoint}"
      continue
    fi
  else
    cmd=(
      sbatch
      "--parsable"
      "--account=${SBATCH_ACCOUNT}"
      "--partition=${SBATCH_PARTITION}"
      "--gres=${SBATCH_GRES}"
      "--dependency=afterok:${job_id}"
      scripts/launch_evaluation.sh
      "eval/${eval_cfg}"
      "++eval.eval_model.checkpoint=${checkpoint}"
      "++eval.evaluation.output_dir=${eval_output_dir}"
      "++eval.wandb.name=${eval_wandb_name}"
    )
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[DRY-RUN] %q ' "${cmd[@]}"
    printf '\n'
    append_manifest "$config_name" "$job_id" "$eval_cfg" "DRYRUN" "$checkpoint" "$eval_output_dir"
    continue
  fi

  result=""
  if ! result="$("${cmd[@]}" 2>&1)"; then
    echo "Failed to submit eval job for $config_name" >&2
    echo "$result" >&2
    continue
  fi
  eval_job_id="${result%%;*}"
  if [[ -z "$eval_job_id" || ! "$eval_job_id" =~ ^[0-9]+$ ]]; then
    echo "Failed to parse eval job id for $config_name" >&2
    echo "$result" >&2
    exit 1
  fi

  echo "${config_name} -> ${eval_cfg} -> ${eval_job_id}"
  append_manifest "$config_name" "$job_id" "$eval_cfg" "$eval_job_id" "$checkpoint" "$eval_output_dir"
done

echo "Manifest written to ${EVAL_MANIFEST_PATH}"
