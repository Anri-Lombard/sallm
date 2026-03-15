#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_eval_only_from_topk.sh --topk-csv <csv> [options]

Options:
  --task <name|all>         Filter task_config rows (default: all)
  --manifest <csv>          Output manifest path
  --account <name>          Pass through to sbatch
  --partition <name>        Pass through to sbatch
  --gres <spec>             GPU request for eval jobs
  --eval-pack-suffix <txt>  Append suffix to the default task pack name
                            Leave unset for final held-out test evaluation.
                            Use `_val` only if you intentionally want validation/dev lm-eval packs.
  --eval-output-root <dir>  Base output dir for eval results
  --dry-run                 Print sbatch commands only
EOF
}

TOPK_CSV=""
TASK_FILTER="all"
MANIFEST_PATH="outputs/hpo/llama_eval_only_manifest.csv"
SBATCH_ACCOUNT=""
SBATCH_PARTITION=""
SBATCH_GRES=""
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
  if [[ -n "$EVAL_PACK_SUFFIX" ]]; then
    EVAL_OUTPUT_ROOT="${SCRATCH_ROOT}/masters/sallm/results/eval/rerank${EVAL_PACK_SUFFIX}"
  else
    EVAL_OUTPUT_ROOT="${SCRATCH_ROOT}/masters/sallm/results/eval/rerank"
  fi
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
    "checkpoint_path",
    "eval_output_dir",
    "eval_config",
    "eval_task_pack",
    "eval_job_id",
    "status",
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
with open(path, "a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow(row)
PY
}

declare -a EVAL_SBATCH_ARGS
[[ -n "$SBATCH_ACCOUNT" ]] && EVAL_SBATCH_ARGS+=("--account=$SBATCH_ACCOUNT")
[[ -n "$SBATCH_PARTITION" ]] && EVAL_SBATCH_ARGS+=("--partition=$SBATCH_PARTITION")
[[ -n "$SBATCH_GRES" ]] && EVAL_SBATCH_ARGS+=("--gres=$SBATCH_GRES")

echo "Submitting eval-only jobs from $TOPK_CSV"
echo "task filter=${TASK_FILTER} | dry-run=${DRY_RUN}"

python3 - "$TOPK_CSV" "$TASK_FILTER" <<'PY' | \
while IFS=$'\t' read -r rank project sweep_id sweep_name task_config run_id run_name objective_metric objective_goal objective_value; do
import csv
import sys

csv_path = sys.argv[1]
task_filter = (sys.argv[2] if len(sys.argv) > 2 else "all").strip().lower()

with open(csv_path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        task = (row.get("task_config") or "").strip()
        if task_filter not in {"", "all"}:
            if task_filter not in task.lower() and task_filter != task.lower():
                continue
        print("\t".join([
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
        ]))
PY
  [[ -z "$rank" ]] && continue

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
  checkpoint_path="${SCRATCH_ROOT}/masters/sallm/checkpoints/rerank/${task_config}/${candidate_root}/final_adapter"
  eval_output_dir="${EVAL_OUTPUT_ROOT}/${task_config}/${candidate_root}"
  eval_wandb_name="eval-rerank-${task_config//_/-}-${candidate_root}"

  status="submitted"
  eval_job_id=""

  if [[ -z "$eval_cfg" ]] || [[ ! -f "src/conf/eval/${eval_cfg}.yaml" ]]; then
    status="missing_eval_config"
  elif [[ ! -d "$checkpoint_path" ]]; then
    status="missing_checkpoint"
  else
    eval_override_args=(
      "++eval.eval_model.checkpoint=${checkpoint_path}"
      "++eval.evaluation.output_dir=${eval_output_dir}"
      "++eval.wandb.name=${eval_wandb_name}"
    )
    if [[ -n "$eval_task_pack" ]]; then
      eval_override_args+=("++eval.evaluation.task_packs=[${eval_task_pack}]")
    fi

    eval_submit_out=""
    eval_status=0
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY-RUN] sbatch ${EVAL_SBATCH_ARGS[*]} scripts/launch_evaluation.sh eval/${eval_cfg} ${eval_override_args[*]}"
      eval_job_id="DRYEVAL${rank}_${run_id}"
    else
      if ! submit_cmd eval_submit_out sbatch "${EVAL_SBATCH_ARGS[@]}" scripts/launch_evaluation.sh "eval/${eval_cfg}" "${eval_override_args[@]}"; then
        eval_status=$?
      fi
      eval_job_id="$(extract_job_id "$eval_submit_out")"
      if [[ -z "$eval_job_id" ]]; then
        echo "Failed to parse eval job id for ${task_config} rank=${rank}" >&2
        echo "$eval_submit_out" >&2
        [[ "$eval_status" -ne 0 ]] && echo "sbatch exit status: $eval_status" >&2
        status="submit_failed"
      fi
    fi
  fi

  now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  append_manifest_row \
    "$now_utc" "$project" "$sweep_id" "$sweep_name" "$task_config" "$rank" "$run_id" "$run_name" \
    "$objective_metric" "$objective_goal" "$objective_value" "$checkpoint_path" "$eval_output_dir" \
    "$eval_cfg" "$eval_task_pack" "$eval_job_id" "$status"

  echo "[$task_config rank=$rank run=$run_id] eval_job=${eval_job_id:-NA} eval_cfg=${eval_cfg:-NA} eval_pack=${eval_task_pack:-NA} status=$status"
done

echo "Manifest written to: $MANIFEST_PATH"
