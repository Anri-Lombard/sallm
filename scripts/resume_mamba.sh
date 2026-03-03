#!/bin/bash
##SBATCH --account=your-slurm-account
#SBATCH --partition=a100
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name="sallm-mamba-resume"
##SBATCH --mail-user=you@example.com
#SBATCH --mail-type=FAIL,END

set -euo pipefail

export MKL_INTERFACE_LAYER=LP64,INTEL64

CONFIG="base/mamba_125m.yaml"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/cluster_env.sh"
setup_sallm_cluster_env

export PYTHONPATH="$SCRATCH/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate sallm-uv
set -u

export PATH="$HOME/.local/bin:$PATH"
cd "$PROJECT_ROOT"
uv sync --frozen --inexact
source .venv/bin/activate

export MAMBA_SCAN_IMPL="cuda"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Resuming training with $CONFIG"
accelerate launch --mixed_precision=bf16 -m sallm.main --config-name "$CONFIG"
