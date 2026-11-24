#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL,END

set -euo pipefail

CFG="${1:-}"
if [[ -z "$CFG" ]]; then
  echo "Usage: sbatch $0 <config_name_without_yaml> [hydra overrides...]" >&2
  echo "Example: sbatch $0 finetune/run finetune/defaults/arch@_global_=hybrid finetune/defaults/datasets@_global_=t2x dataset.subset=xho" >&2
  exit 1
fi
shift
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/auth.sh"
source "$SCRIPT_DIR/lib/env.sh"
set_sallm_cluster_env

RUN_SPEC="$CFG ${EXTRA_ARGS[*]:-}"
setup_sallm_runtime "$RUN_SPEC"

CFG_NAME="${CFG##*/}"
JOB_NAME="ft-${CFG_NAME}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME"
  exec > >(tee -a "$JOB_LOG_DIR/${JOB_NAME}-${SLURM_JOB_ID}.out") 2>&1
fi

NUM_PROCS="$(detect_gpu_count)"
MASTER_PORT=$((29500 + (${SLURM_JOB_ID:-0} % 1000)))

accelerate launch \
  --num_processes "$NUM_PROCS" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "$MASTER_PORT" \
  -m sallm.main --config-name "$CFG" "${EXTRA_ARGS[@]}"
