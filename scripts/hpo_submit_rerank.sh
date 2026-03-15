#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/hpo_submit_rerank.sh --topk-csv <csv> [options]

Options:
  --task <name|all>         Filter task_config rows (default: all)
  --manifest <csv>          Output manifest path
  --account <name>          Pass through to sbatch
  --partition <name>        Pass through to sbatch
  --gres <spec>             GPU request for finetune jobs
  --eval-gres <spec>        GPU request for eval jobs (defaults to --gres)
  --eval-pack-suffix <txt>  Append suffix to the default task pack name
                            Use `_val` for post-HPO rerank selection so we score
                            candidates on validation/dev via lm-eval, not test.
  --eval-output-root <dir>  Base output dir for eval results
  --dry-run                 Print sbatch commands only

Examples:
  bash scripts/hpo_submit_rerank.sh \
    --topk-csv outputs/hpo/llama_pos_combined_topk.csv \
    --task all \
    --account nlpgroup \
    --partition a100 \
    --gres gpu:ampere:2 \
    --eval-gres gpu:ampere:1 \
    --eval-pack-suffix _val \
    --eval-output-root /scratch/lmbanr001/masters/sallm/results/eval/rerank_val \
    --manifest outputs/hpo/llama_pos_rerank_manifest_a100.csv
EOF
}

TOPK_CSV=""
TASK_FILTER="all"
MANIFEST_PATH="outputs/hpo/llama_rerank_manifest.csv"
SBATCH_ACCOUNT=""
SBATCH_PARTITION=""
SBATCH_GRES=""
SBATCH_EVAL_GRES=""
EVAL_PACK_SUFFIX=""
EVAL_OUTPUT_ROOT=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topk-csv)
      TOPK_CSV="${2:-}"
      shift 2
      ;;
    --task)
      TASK_FILTER="${2:-}"
      shift 2
      ;;
    --manifest)
      MANIFEST_PATH="${2:-}"
      shift 2
      ;;
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
    --eval-gres)
      SBATCH_EVAL_GRES="${2:-}"
      shift 2
      ;;
    --eval-pack-suffix)
      EVAL_PACK_SUFFIX="${2:-}"
      shift 2
      ;;
    --eval-output-root)
      EVAL_OUTPUT_ROOT="${2:-}"
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

if [[ -z "$TOPK_CSV" ]]; then
  echo "--topk-csv is required" >&2
  usage
  exit 1
fi
if [[ ! -f "$TOPK_CSV" ]]; then
  echo "top-k CSV not found: $TOPK_CSV" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "$DRY_RUN" -eq 0 ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on HEX or use --dry-run." >&2
  exit 1
fi

SCRATCH_ROOT="${SCRATCH:-/scratch/lmbanr001}"
if [[ -z "$EVAL_OUTPUT_ROOT" ]]; then
  EVAL_OUTPUT_ROOT="${SCRATCH_ROOT}/masters/sallm/results/eval/rerank"
fi
mkdir -p "$(dirname "$MANIFEST_PATH")"

python3 - "$MANIFEST_PATH" <<'PY'
import csv
import sys

path = sys.argv[1]
header = [
    "submitted_at",
    "project",
    "sweep_id",
    "sweep_name",
    "task_config",
    "rank",
    "run_id",
    "run_name",
    "objective_metric",
    "objective_goal",
    "objective_value",
    "topk_metric",
    "parameter_json",
    "override_args",
    "finetune_job_id",
    "eval_job_id",
    "checkpoint_path",
    "eval_output_dir",
    "eval_config",
    "eval_task_pack",
    "status",
    "score",
]
with open(path, "w", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow(header)
PY

map_eval_task() {
  case "$1" in
    ner) echo "masakhaner" ;;
    news) echo "masakhanews" ;;
    pos) echo "masakhapos" ;;
    sib) echo "sib" ;;
    injongointent) echo "injongointent" ;;
    t2x) echo "t2x" ;;
    afrihg) echo "afrihg" ;;
    *) echo "" ;;
  esac
}

infer_task_and_lang() {
  local cfg="$1"
  if [[ "$cfg" =~ ^llama_sa_general_(.+)$ ]]; then
    echo "sa_general ${BASH_REMATCH[1]}"
    return
  fi
  if [[ "$cfg" =~ ^llama_([^_]+)_(.+)$ ]]; then
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
    return
  fi
  echo "unknown unknown"
}

topk_metric_for_cfg() {
  local cfg="$1"
  if [[ "$cfg" == *"_ner_"* ]]; then
    echo "f1"
  elif [[ "$cfg" == *"_pos_"* ]]; then
    echo "token_accuracy"
  else
    echo "eval/loss"
  fi
}

submit_cmd() {
  local __outvar="$1"
  shift
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] $*"
    printf -v "$__outvar" '%s' "DRYRUN"
    return 0
  fi
  local result
  local status=0
  if ! result=$("$@" 2>&1); then
    status=$?
  fi
  printf -v "$__outvar" '%s' "$result"
  return "$status"
}

extract_job_id() {
  local text="$1"
  echo "$text" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}' | tail -n1 || true
}

append_manifest_row() {
  python3 - "$MANIFEST_PATH" "$@" <<'PY'
import csv
import sys

path = sys.argv[1]
row = sys.argv[2:]
row.append("")
with open(path, "a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow(row)
PY
}

PY_ROWS=$(python3 - "$TOPK_CSV" "$TASK_FILTER" <<'PY'
import base64
import csv
import sys

csv_path = sys.argv[1]
task_filter = (sys.argv[2] if len(sys.argv) > 2 else "all").strip().lower()

with open(csv_path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        task = (row.get("task_config") or "").strip()
        if task_filter not in {"", "all"}:
          tf = task_filter
          if tf not in task.lower() and tf != task.lower():
              continue
        fields = [
            row.get("rank", ""),
            row.get("project", ""),
            row.get("sweep_id", ""),
            row.get("sweep_name", ""),
            task,
            row.get("run_id", ""),
            row.get("run_name", ""),
            row.get("objective_metric", ""),
            row.get("objective_goal", ""),
            row.get("objective_value", ""),
            base64.b64encode((row.get("parameter_json", "{}")).encode("utf-8")).decode("ascii"),
            row.get("override_args", ""),
        ]
        print("\t".join(fields))
PY
)

mapfile -t ROWS <<<"$PY_ROWS"
if [[ "${#ROWS[@]}" -eq 0 ]]; then
  echo "No rows selected from $TOPK_CSV for --task $TASK_FILTER"
  exit 0
fi

declare -a FT_SBATCH_ARGS
declare -a EVAL_SBATCH_ARGS
[[ -n "$SBATCH_ACCOUNT" ]] && FT_SBATCH_ARGS+=("--account=$SBATCH_ACCOUNT") && EVAL_SBATCH_ARGS+=("--account=$SBATCH_ACCOUNT")
[[ -n "$SBATCH_PARTITION" ]] && FT_SBATCH_ARGS+=("--partition=$SBATCH_PARTITION") && EVAL_SBATCH_ARGS+=("--partition=$SBATCH_PARTITION")
[[ -n "$SBATCH_GRES" ]] && FT_SBATCH_ARGS+=("--gres=$SBATCH_GRES")
if [[ -n "$SBATCH_EVAL_GRES" ]]; then
  EVAL_SBATCH_ARGS+=("--gres=$SBATCH_EVAL_GRES")
elif [[ -n "$SBATCH_GRES" ]]; then
  EVAL_SBATCH_ARGS+=("--gres=$SBATCH_GRES")
fi

echo "Submitting independent rerank candidates from $TOPK_CSV"
if [[ -z "$EVAL_PACK_SUFFIX" ]]; then
  echo "WARNING: no --eval-pack-suffix provided."
  echo "For clean post-HPO selection, rerank should usually target validation packs (for example `_val`),"
  echo "and the held-out test split should be reserved for the final selected winner only."
fi
echo "Rows selected: ${#ROWS[@]} | dry-run=${DRY_RUN}"

for row in "${ROWS[@]}"; do
  [[ -z "$row" ]] && continue
  IFS=$'\t' read -r rank project sweep_id sweep_name task_config run_id run_name objective_metric objective_goal objective_value param_b64 override_args <<<"$row"

  param_json=$(python3 - "$param_b64" <<'PY'
import base64
import sys
print(base64.b64decode(sys.argv[1]).decode("utf-8"))
PY
)

  PY_PARAM_OVERRIDES=$(python3 - "$param_json" <<'PY'
import json
import re
import sys

def render(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", text):
        return text
    return json.dumps(text)

raw = sys.argv[1].strip()
obj = json.loads(raw) if raw else {}
for key in sorted(obj.keys()):
    # Use Hydra '++' so HPO-selected keys can be added when the current
    # finetune YAML omits them (for example gradient_checkpointing or
    # weight_decay on llama_pos_* configs) while still overriding existing keys.
    print(f"++finetune.{key}={render(obj[key])}")
PY
)

  param_overrides=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && param_overrides+=("$line")
  done <<< "$PY_PARAM_OVERRIDES"

  read -r task lang <<<"$(infer_task_and_lang "$task_config")"
  eval_task="$(map_eval_task "$task")"
  eval_cfg=""
  eval_task_pack=""
  if [[ -n "$eval_task" ]]; then
    eval_cfg="run_llama_${eval_task}_${lang}"
    eval_task_pack="${eval_task}_${lang}"
    if [[ -n "$EVAL_PACK_SUFFIX" ]]; then
      eval_task_pack="${eval_task_pack}${EVAL_PACK_SUFFIX}"
      if [[ ! -f "src/conf/eval/tasks/${eval_task_pack}.yaml" ]]; then
        echo "Missing eval task pack override: src/conf/eval/tasks/${eval_task_pack}.yaml" >&2
        eval_cfg=""
      fi
    fi
  fi

  candidate_root="${rank}_${run_id}"
  output_dir="${SCRATCH_ROOT}/masters/sallm/checkpoints/rerank/${task_config}/${candidate_root}"
  logging_dir="${SCRATCH_ROOT}/masters/sallm/logs/rerank/${task_config}/${candidate_root}"
  eval_output_dir="${EVAL_OUTPUT_ROOT}/${task_config}/${candidate_root}"
  checkpoint_path="${output_dir}/final_merged_model"

  wandb_name="rerank-${task_config//_/-}-${candidate_root}"
  eval_wandb_name="eval-rerank-${task_config//_/-}-${candidate_root}"
  rerank_run_name="rerank_${task_config}_${candidate_root}"

  extra_args=(
    "finetune.training.output_dir=${output_dir}"
    "finetune.training.logging_dir=${logging_dir}"
    "finetune.training.run_name=${rerank_run_name}"
    "finetune.wandb.name=${wandb_name}"
    # Rerank only needs the final adapter; epoch checkpoints burn scratch quota
    # and can poison eval dependencies when a finetune fails mid-save.
    "finetune.training.save_strategy=no"
    "finetune.hub.enabled=false"
  )
  extra_args+=("${param_overrides[@]}")

  ft_submit_out=""
  ft_status=0
  if ! submit_cmd ft_submit_out sbatch "${FT_SBATCH_ARGS[@]}" scripts/launch_finetune.sh "finetune/${task_config}" "${extra_args[@]}"; then
    ft_status=$?
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    ft_job_id="DRYFT${rank}_${run_id}"
  else
    ft_job_id="$(extract_job_id "$ft_submit_out")"
    if [[ -z "$ft_job_id" ]]; then
      echo "Failed to parse finetune job id for ${task_config} rank=${rank}" >&2
      echo "$ft_submit_out" >&2
      [[ "$ft_status" -ne 0 ]] && echo "sbatch exit status: $ft_status" >&2
      ft_job_id=""
    fi
  fi

  eval_job_id=""
  status="submitted"
  if [[ -n "$eval_cfg" ]] && [[ -f "src/conf/eval/${eval_cfg}.yaml" ]] && [[ -n "$ft_job_id" ]]; then
    eval_override_args=(
      "++eval.eval_model.checkpoint=${checkpoint_path}"
      "++eval.evaluation.output_dir=${eval_output_dir}"
      "++eval.wandb.name=${eval_wandb_name}"
    )
    if [[ -n "$eval_task_pack" ]] && [[ -f "src/conf/eval/tasks/${eval_task_pack}.yaml" ]]; then
      eval_override_args+=("++eval.evaluation.task_packs=[${eval_task_pack}]")
    fi
    eval_submit_out=""
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY-RUN] sbatch ${EVAL_SBATCH_ARGS[*]} --dependency=afterok:${ft_job_id} scripts/launch_evaluation.sh eval/${eval_cfg} ${eval_override_args[*]}"
      eval_job_id="DRYEVAL${rank}_${run_id}"
    else
      eval_status=0
      if ! submit_cmd eval_submit_out sbatch "${EVAL_SBATCH_ARGS[@]}" --dependency=afterok:"${ft_job_id}" scripts/launch_evaluation.sh "eval/${eval_cfg}" "${eval_override_args[@]}"; then
        eval_status=$?
      fi
      eval_job_id="$(extract_job_id "$eval_submit_out")"
      if [[ -z "$eval_job_id" ]]; then
        echo "Failed to parse eval job id for ${task_config} rank=${rank}" >&2
        echo "$eval_submit_out" >&2
        [[ "$eval_status" -ne 0 ]] && echo "sbatch exit status: $eval_status" >&2
        status="submitted_eval_failed"
      fi
    fi
  else
    status="submitted_no_eval"
  fi

  topk_metric="$(topk_metric_for_cfg "$task_config")"
  now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  append_manifest_row \
    "$now_utc" "$project" "$sweep_id" "$sweep_name" "$task_config" "$rank" "$run_id" "$run_name" \
    "$objective_metric" "$objective_goal" "$objective_value" "$topk_metric" "$param_json" "$override_args" \
    "$ft_job_id" "$eval_job_id" "$checkpoint_path" "$eval_output_dir" "$eval_cfg" "$eval_task_pack" "$status"

  echo "[$task_config rank=$rank run=$run_id] ft_job=${ft_job_id:-NA} eval_job=${eval_job_id:-NA} eval_cfg=${eval_cfg:-NA} eval_pack=${eval_task_pack:-NA}"
done

echo "Manifest written to: $MANIFEST_PATH"
