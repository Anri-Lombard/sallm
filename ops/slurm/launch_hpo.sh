#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=FAIL,END

set -euo pipefail

SWEEP_ARG="${1:-}"
COUNT="${2:-50}"
SWEEP_DIR="src/conf/sweeps"

if [[ -z "$SWEEP_ARG" ]]; then
  echo "Usage: sbatch $0 <sweep_yaml_or_name_without_yaml> [count]" >&2
  echo "Example: sbatch $0 lr_grid 3" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/auth.sh"
source "$SCRIPT_DIR/lib/env.sh"
set_sallm_cluster_env
cd "$SALLM_REPO_DIR"

SWEEP_NAME="${SWEEP_ARG%.yaml}"
SWEEP_NAME="${SWEEP_NAME##*/}"
JOB_NAME="hpo-${SWEEP_NAME}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME"
  mkdir -p logs
  exec > >(tee -a "logs/${JOB_NAME}-${SLURM_JOB_ID}.out") 2>&1
fi

SWEEP_FILE=""
if [[ -f "$SWEEP_ARG" ]]; then
  SWEEP_FILE="$SWEEP_ARG"
elif [[ -f "${SWEEP_DIR}/${SWEEP_ARG}.yaml" ]]; then
  SWEEP_FILE="${SWEEP_DIR}/${SWEEP_ARG}.yaml"
elif [[ "$SWEEP_ARG" == *.yaml && -f "${SWEEP_DIR}/${SWEEP_ARG}" ]]; then
  SWEEP_FILE="${SWEEP_DIR}/${SWEEP_ARG}"
else
  echo "Sweep file not found for arg: $SWEEP_ARG" >&2
  exit 1
fi

setup_sallm_runtime "$(cat "$SWEEP_FILE")"

if [[ -z "${TOKENIZER_PATH:-}" ]]; then
  TOKENIZER_PATH=$(python -m sallm.hpo.run --tokenizer-path)
  if [[ -n "$TOKENIZER_PATH" ]]; then
    export TOKENIZER_PATH
  fi
fi

CREATE_OUT=$(wandb sweep "$SWEEP_FILE" 2>&1)
echo "$CREATE_OUT"

SWEEP_PATH=$(printf "%s\n" "$CREATE_OUT" | sed -nE 's/^.*Run sweep agent with: wandb agent ([^[:space:]]+).*$/\1/p' | tail -n1)
if [[ -z "$SWEEP_PATH" ]]; then
  SWEEP_PATH=$(printf "%s\n" "$CREATE_OUT" | sed -nE 's@.*https?://wandb\.ai/([^/]+)/([^/]+)/sweeps/([A-Za-z0-9_-]+).*@\1/\2/\3@p' | tail -n1)
fi
if [[ -z "$SWEEP_PATH" ]]; then
  echo "Failed to parse sweep ID from wandb output" >&2
  exit 1
fi

NUM_GPUS="$(detect_gpu_count)"
AGENTS_TO_RUN=$NUM_GPUS
if [[ "$COUNT" -lt "$AGENTS_TO_RUN" ]]; then
  AGENTS_TO_RUN="$COUNT"
fi

BASE_PER_AGENT=$((COUNT / AGENTS_TO_RUN))
REMAINDER=$((COUNT % AGENTS_TO_RUN))

echo "Launching $AGENTS_TO_RUN agents across $NUM_GPUS GPUs (total=$COUNT runs)"

PIDS=()
for IDX in $(seq 0 $((AGENTS_TO_RUN - 1))); do
  PER_AGENT=$BASE_PER_AGENT
  if [[ $IDX -lt $REMAINDER ]]; then
    PER_AGENT=$((PER_AGENT + 1))
  fi
  if [[ $PER_AGENT -eq 0 ]]; then
    continue
  fi
  export CUDA_VISIBLE_DEVICES="$IDX"
  echo "GPU $IDX -> wandb agent --count $PER_AGENT $SWEEP_PATH"
  wandb agent --count "$PER_AGENT" "$SWEEP_PATH" &
  PIDS+=("$!")
done

EXIT_CODE=0
for P in "${PIDS[@]}"; do
  if ! wait "$P"; then
    EXIT_CODE=1
  fi
done

exit "$EXIT_CODE"
