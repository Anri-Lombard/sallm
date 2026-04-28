#!/bin/bash
#SBATCH --partition=a100
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name="sallm-mamba-resume"
#SBATCH --mail-type=FAIL,END

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env.sh"
set_sallm_cluster_env

export MKL_INTERFACE_LAYER=LP64,INTEL64

CONFIG="base/mamba_125m.yaml"

export PYTHONPATH="$SCRATCH/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate sallm-uv
set -u

export PATH="$SALLM_HOME_DIR/.local/bin:$PATH"
cd "$SALLM_REPO_DIR"
uv sync --frozen --inexact
source .venv/bin/activate

export MAMBA_SCAN_IMPL="cuda"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Resuming training with $CONFIG"
accelerate launch --mixed_precision=bf16 -m sallm.main --config-name "$CONFIG"
