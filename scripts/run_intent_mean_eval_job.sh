#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL,END

set -euo pipefail

ARCHITECTURE="$1"
CHECKPOINT="$2"
CONFIG="$3"
OUTPUT="$4"
ADAPTER="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$SCRIPT_DIR/lib/env.sh" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}/scripts" "$HOME/masters/sallm/scripts"; do
    if [[ -f "$candidate/lib/env.sh" ]]; then
      SCRIPT_DIR="$candidate"
      break
    fi
  done
fi
source "$SCRIPT_DIR/lib/env.sh"
set_sallm_cluster_env
export HF_TOKEN
HF_TOKEN="$(cat "${HF_TOKEN_FILE:-$SALLM_HOME_DIR/.huggingface/token}")"

cd "$SALLM_REPO_DIR"
source .venv/bin/activate

export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

[[ "$ARCHITECTURE" == "mamba" ]] && export MAMBA_SCAN_IMPL=cuda
[[ "$ARCHITECTURE" == "gated_deltanet" ]] && export FLA_DISABLE_BACKEND_DISPATCH=1

ARGS=(
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --architecture "$ARCHITECTURE"
  --output "$OUTPUT"
)
[[ -n "$ADAPTER" ]] && ARGS+=(--adapter "$ADAPTER")

python scripts/run_injongointent_mean_eval.py "${ARGS[@]}"
