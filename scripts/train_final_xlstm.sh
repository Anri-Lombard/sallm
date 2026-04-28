#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name="sallm-xlstm"
#SBATCH --mail-type=FAIL,END

set -euo pipefail

export MKL_INTERFACE_LAYER=LP64,INTEL64

CONFIG="base/xlstm_125m.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env.sh"
source "$SCRIPT_DIR/lib/auth.sh"
set_sallm_cluster_env

export PYTHONPATH="$SCRATCH/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/hf"
load_hf_token || true
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate sallm-uv
set -u

export PATH="$SALLM_HOME_DIR/.local/bin:$PATH"
cd "$SALLM_REPO_DIR"
uv sync --frozen
source .venv/bin/activate

echo "Launching training with $CONFIG"
accelerate launch --mixed_precision=bf16 -m sallm.main --config-name "$CONFIG"
