#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=sallm-pretrain
#SBATCH --mail-type=FAIL,END

set -euo pipefail

CONFIG_NAME="${1:-base/llama_125m}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env.sh"
source "$SCRIPT_DIR/lib/auth.sh"
set_sallm_cluster_env

echo "Using Hydra config: ${CONFIG_NAME}"

export PYTHONPATH="$SCRATCH/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
load_hf_token || true
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

echo "Setting up environment..."
module load python/miniconda3-py3.12
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx sallm-uv; then
  set +u
  conda activate sallm-uv
  set -u
else
  echo "Conda environment sallm-uv is unavailable; using the repository .venv."
fi

export PATH="$SALLM_HOME_DIR/.local/bin:$PATH"
cd "$SALLM_REPO_DIR"
if [[ "$CONFIG_NAME" == *gated_deltanet* ]]; then
  uv sync --frozen --extra pure-gdn
else
  uv sync --frozen
fi
source .venv/bin/activate
echo "Environment ready."

export HYDRA_FULL_ERROR=1

echo "Launching pretraining..."

NUM_PROCESSES="${SLURM_GPUS_ON_NODE:-4}"
accelerate launch --num_processes "$NUM_PROCESSES" --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  -m sallm.main --config-name "$CONFIG_NAME" "$@"

echo "Pretraining finished."
