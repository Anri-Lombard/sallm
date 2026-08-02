#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2%1
#SBATCH --job-name=mamba-mono-pos-corrected
#SBATCH --mail-type=FAIL,END

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$SCRIPT_DIR/lib/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$HOME/masters/sallm}/scripts"
fi
source "$SCRIPT_DIR/lib/env.sh"
set_sallm_cluster_env

languages=(xho zul tsn)
adapters=(
  anrilombard/sallm-mamba-pos_xho
  anrilombard/sallm-mamba-pos_zul
  anrilombard/sallm-mamba-pos_tsn
)
index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
language="${languages[$index]}"
adapter="${adapters[$index]}"

export HF_TOKEN
HF_TOKEN="$(cat "${HF_TOKEN_FILE:-$SALLM_HOME_DIR/.huggingface/token}")"
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

cd "$SALLM_REPO_DIR"
source .venv/bin/activate

extra_args=()
if [[ -n "${SALLM_POS_MAX_SAMPLES:-}" ]]; then
  extra_args+=(--max-samples-per-lang "$SALLM_POS_MAX_SAMPLES")
fi

python scripts/run_constrained_pos_eval.py \
  --checkpoint anrilombard/sallm-mamba-125m \
  --peft-adapter "$adapter" \
  --languages "$language" \
  --split test \
  --contract tuple \
  --score-mode mean \
  --pad-to-multiple-of 64 \
  "${extra_args[@]}" \
  --output "$SCRATCH/masters/sallm/results/eval/mamba_mono_pos_corrected_20260730/${language}${SALLM_POS_OUTPUT_SUFFIX:-}.json"
