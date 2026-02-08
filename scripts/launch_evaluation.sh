#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

RAW_CONFIG_NAME="${1:-}"
if [[ -z "$RAW_CONFIG_NAME" ]]; then
    echo "Usage: sbatch $0 <config_name_without_yaml>"
    echo "Examples:"
    echo "  sbatch $0 run_mamba_belebele_zul"
    echo "  sbatch $0 eval/run_mamba_belebele_zul"
    exit 1
fi

CONFIG_NAME="${RAW_CONFIG_NAME%.yaml}"
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
REPO_ROOT="$HOME/masters/sallm"
cd "$REPO_ROOT"
uv sync --frozen --inexact
source .venv/bin/activate

CONFIG_PATH="$REPO_ROOT/src/conf/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: config file not found: $CONFIG_PATH"
    echo "Available eval configs:"
    if [[ -d "$REPO_ROOT/src/conf/eval" ]]; then
        for file in "$REPO_ROOT"/src/conf/eval/*.yaml; do
            [[ -e "$file" ]] || continue
            echo "  - eval/$(basename "$file" .yaml)"
        done
    else
        echo "  (missing directory: $REPO_ROOT/src/conf/eval)"
    fi
    exit 2
fi

# Install CUDA kernels based on model type
echo "--- CUDA kernel status ---"
IS_MAMBA=false
IS_XLSTM=false
[[ "$CONFIG_NAME" == *mamba* ]] && IS_MAMBA=true
[[ "$CONFIG_NAME" == *xlstm* ]] && IS_XLSTM=true

if $IS_MAMBA; then
    if python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" 2>/dev/null; then
        echo "✓ Mamba fast path (CUDA kernels) available"
    else
        echo "Mamba kernels missing or ABI mismatch, rebuilding..."
        uv pip uninstall mamba-ssm causal-conv1d 2>/dev/null || true
        uv pip install --no-build-isolation mamba-ssm causal-conv1d 2>&1
        if ! python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" 2>/dev/null; then
            echo "ERROR: Mamba CUDA kernels failed to load. Aborting (native impl causes OOM)."
            exit 1
        fi
        echo "✓ Mamba fast path (CUDA kernels) available"
    fi
fi

if $IS_XLSTM; then
    if python -c "from transformers.utils import is_xlstm_available; assert is_xlstm_available()" 2>/dev/null; then
        echo "✓ xLSTM fast path (NX-AI kernels) available"
    else
        echo "Installing xlstm package..."
        uv pip install xlstm 2>&1 | tail -5 || true
        if python -c "from transformers.utils import is_xlstm_available; assert is_xlstm_available()" 2>/dev/null; then
            echo "✓ xLSTM fast path (NX-AI kernels) available"
        else
            echo "ℹ Using xLSTM native implementation"
        fi
    fi
fi
echo "-------------------------------"

# Set PyTorch CUDA allocation config
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

python -m sallm.main --config-name "$CONFIG_NAME"
