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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/cluster_env.sh"
source "$SCRIPT_DIR/lib/auth.sh"
setup_sallm_cluster_env

echo "Using Hydra config: ${CONFIG_NAME}"

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

echo "Launching final training run..."

accelerate launch --num_processes 4 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    -m sallm.main --config-name "$CONFIG_NAME"

echo "Final training run finished."
