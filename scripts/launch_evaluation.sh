#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

CONFIG_NAME="$1"
if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: sbatch $0 <config_name_without_yaml>"; exit 1
fi

if [[ "$CONFIG_NAME" != */* ]]; then
    CONFIG_NAME="eval/$CONFIG_NAME"
fi

CFG_NAME="${CONFIG_NAME##*/}"
JOB_NAME="eval-${CFG_NAME#mamba_}"
JOB_NAME="${JOB_NAME#llama_}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME"
  mkdir -p logs
  exec > >(tee -a "logs/${JOB_NAME}-${SLURM_JOB_ID}.out") 2>&1
fi

export HYDRA_FULL_ERROR=1

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"
# Note: Don't prepend scratch to PYTHONPATH - venv has patched transformers for xLSTM
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export TRITON_CACHE_DIR="$SCRATCH/.triton/cache"
mkdir -p "$TRITON_CACHE_DIR"

echo "--- Storage Usage ---"
df -h /home /scratch 2>/dev/null || true
echo "-------------------------------"

module load python/miniconda3-py3.12
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u
conda activate sallm-uv
set -u

export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/masters/sallm"
uv sync --frozen
source .venv/bin/activate

# Install Mamba CUDA kernels (not in lockfile, must reinstall after uv sync)
echo "--- CUDA kernel status ---"
# Test if mamba-ssm imports correctly (CUDA symbol compatibility)
if python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" 2>/dev/null; then
    echo "✓ Mamba fast path (CUDA kernels) available"
else
    echo "Mamba kernels missing or ABI mismatch, rebuilding..."
    pip uninstall -y mamba-ssm causal-conv1d 2>/dev/null || true
    pip install --no-cache-dir --no-build-isolation mamba-ssm causal-conv1d 2>&1
    if ! python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" 2>/dev/null; then
        echo "ERROR: Mamba CUDA kernels failed to load. Aborting (native impl causes OOM)."
        exit 1
    fi
    echo "✓ Mamba fast path (CUDA kernels) available"
fi
# xLSTM kernels
if ! python -c "from mlstm_kernels import mlstm" 2>/dev/null; then
    echo "Installing mlstm-kernels..."
    pip install mlstm-kernels 2>&1 | tail -5
fi
python -c "
try:
    from mlstm_kernels import mlstm
    print('✓ xLSTM fast path (Triton kernels) available')
except ImportError as e:
    print(f'ℹ Using xLSTM native implementation')
"
echo "-------------------------------"

# Set PyTorch CUDA allocation config
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

python -m sallm.main --config-name "$CONFIG_NAME"
