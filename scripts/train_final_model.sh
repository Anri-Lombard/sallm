#!/bin/bash
##SBATCH --account=your-slurm-account
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name="sallm-llama-mobilellama"
##SBATCH --mail-user=you@example.com
#SBATCH --mail-type=FAIL,END

set -euo pipefail

CONFIG_NAME="${1:-base/llama_125m}"
shift || true
EXTRA_ARGS=("$@")
CONFIG_BASENAME="${CONFIG_NAME##*/}"

resolve_repo_root() {
  local candidate=""
  for candidate in \
    "${PROJECT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "$(pwd)" \
    "$HOME/masters/sallm"
  do
    [[ -n "$candidate" ]] || continue
    if [[ -f "$candidate/scripts/lib/cluster_env.sh" && -f "$candidate/scripts/lib/auth.sh" ]]; then
      printf '%s\n' "${candidate%/}"
      return 0
    fi
  done
  return 1
}

PROJECT_ROOT="$(resolve_repo_root)" || {
  echo "Could not resolve project root for cluster scripts" >&2
  exit 1
}
export PROJECT_ROOT
LIB_DIR="$PROJECT_ROOT/scripts/lib"
source "$LIB_DIR/cluster_env.sh"
source "$LIB_DIR/auth.sh"
setup_sallm_cluster_env

echo "Using Hydra config: ${CONFIG_NAME}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Hydra overrides: ${EXTRA_ARGS[*]}"
fi

export PYTHONPATH="$SCRATCH/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
load_hf_token || true

echo "Setting up environment..."
module load python/miniconda3-py3.12
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate sallm-uv
set -u

export PATH="$HOME/.local/bin:$PATH"
cd "$PROJECT_ROOT"
uv sync --frozen
source .venv/bin/activate
echo "Environment ready."

export HYDRA_FULL_ERROR=1

NUM_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
if [[ -z "$NUM_GPUS" || "$NUM_GPUS" -le 0 ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
  else
    NUM_GPUS=1
  fi
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_NAME="ft-${CONFIG_BASENAME}"
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME"
  mkdir -p logs
  exec > >(tee -a "logs/${JOB_NAME}-${SLURM_JOB_ID}.out") 2>&1
fi

echo "Launching final training run..."

accelerate launch --num_processes "$NUM_GPUS" --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    -m sallm.main --config-name "$CONFIG_NAME" "${EXTRA_ARGS[@]}"

echo "Final training run finished."
