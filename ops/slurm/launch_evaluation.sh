#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL,END

set -euo pipefail

CONFIG_NAME="${1:-}"
if [[ -z "$CONFIG_NAME" ]]; then
  echo "Usage: sbatch $0 <config_name_without_yaml> [hydra overrides...]" >&2
  echo "Example: sbatch $0 eval/run eval_model.checkpoint=uctnlp/mzansilm-125m 'evaluation.task_packs=[sib_xho]' wandb.name=eval-sib-xho" >&2
  exit 1
fi
shift
EXTRA_ARGS=("$@")

if [[ "$CONFIG_NAME" != */* ]]; then
  CONFIG_NAME="eval/$CONFIG_NAME"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/auth.sh"
source "$SCRIPT_DIR/lib/env.sh"
set_sallm_cluster_env

RUN_SPEC="$CONFIG_NAME ${EXTRA_ARGS[*]:-}"
setup_sallm_runtime "$RUN_SPEC"

CFG_NAME="${CONFIG_NAME##*/}"
JOB_NAME="eval-${CFG_NAME}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME"
  exec > >(tee -a "$JOB_LOG_DIR/${JOB_NAME}-${SLURM_JOB_ID}.out") 2>&1
fi

python -m sallm.main --config-name "$CONFIG_NAME" "${EXTRA_ARGS[@]}"
