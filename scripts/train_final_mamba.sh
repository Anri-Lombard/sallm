#!/bin/bash
##SBATCH --account=your-slurm-account
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name="sallm-mamba"
##SBATCH --mail-user=you@example.com
#SBATCH --mail-type=FAIL,END

# TODO make one script instead of seperate final scripts per architecture

set -euo pipefail

export MKL_INTERFACE_LAYER=LP64,INTEL64

CONFIG="base/mamba_125m.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/cluster_env.sh"
source "$SCRIPT_DIR/lib/auth.sh"
setup_sallm_cluster_env

export PYTHONPATH="$SCRATCH/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/hf"
load_hf_token || true

echo "--- Storage Usage ---"
df -h "$HOME" "$SCRATCH" 2>/dev/null || true
echo "-------------------------------"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate sallm-uv
set -u

export PATH="$HOME/.local/bin:$PATH"
cd "$PROJECT_ROOT"
uv sync --frozen
source .venv/bin/activate

echo "--- Checking Mamba CUDA kernels ---"
python -c "
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from causal_conv1d import causal_conv1d_fn
    print('✓ Mamba fast path available')
except ImportError:
    print('ℹ Using HF Transformers native Mamba implementation (no CUDA kernels)')
    print('  This is expected and will work correctly, just slightly slower.')
"
echo "-------------------------------"

export MAMBA_SCAN_IMPL="cuda"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Launching training with $CONFIG"
accelerate launch --mixed_precision=bf16 -m sallm.main --config-name "$CONFIG"
