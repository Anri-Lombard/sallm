#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=hpo-ner_all-resume
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

set -euo pipefail

SWEEP_PATH="${1:-}"
COUNT="${2:-43}"

if [[ -z "$SWEEP_PATH" ]]; then
  echo "Usage: sbatch $0 <sweep_path> [count]" >&2
  echo "Example: sbatch $0 anri-lombard/sallm-ft/z0vyuasg 43" >&2
  exit 1
fi

SWEEP_ID="${SWEEP_PATH##*/}"
mkdir -p logs
exec > >(tee -a "logs/hpo-resume-${SWEEP_ID}-${SLURM_JOB_ID}.out") 2>&1

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"
# Note: Don't set PYTHONPATH - venv has patched transformers
export TRITON_CACHE_DIR="$SCRATCH/.triton/cache"
mkdir -p "$TRITON_CACHE_DIR"
export TOKENIZERS_PARALLELISM="true"
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_TOKEN=$(cat ~/.huggingface/token)
export TORCH_DISTRIBUTED_TIMEOUT=7200
export HYDRA_FULL_ERROR=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"

module load python/miniconda3-py3.12
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sallm-uv
set -u

export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/masters/sallm"
uv sync --frozen --inexact
source .venv/bin/activate

# Install Mamba CUDA kernels (not in lockfile, must reinstall after uv sync)
# Wheels are cached so this is fast
echo "--- Mamba CUDA kernel status ---"
if ! python -c "from mamba_ssm import Mamba2" 2>/dev/null; then
    echo "Installing mamba-ssm and causal-conv1d from cached wheels..."
    uv pip install --no-build-isolation mamba-ssm causal-conv1d 2>&1 | tail -5
fi
python -c "
try:
    from mamba_ssm import Mamba2
    from causal_conv1d import causal_conv1d_fn
    print('✓ Mamba fast path (CUDA kernels) available')
except ImportError as e:
    print(f'ℹ Using HF Transformers native Mamba implementation: {e}')
"
# xLSTM kernels (NX-AI package)
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
echo "-------------------------------"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

NUM_GPUS=2
AGENTS_TO_RUN=$NUM_GPUS
BASE_PER_AGENT=$(( COUNT / AGENTS_TO_RUN ))
REMAINDER=$(( COUNT % AGENTS_TO_RUN ))

echo "Resuming sweep $SWEEP_PATH with $COUNT remaining runs across $NUM_GPUS GPUs"

PIDS=()
for IDX in $(seq 0 $((AGENTS_TO_RUN - 1))); do
  PER_AGENT=$BASE_PER_AGENT
  if [[ $IDX -lt $REMAINDER ]]; then
    PER_AGENT=$(( PER_AGENT + 1 ))
  fi
  export CUDA_VISIBLE_DEVICES="$IDX"
  echo "GPU $IDX -> wandb agent --count $PER_AGENT $SWEEP_PATH"
  wandb agent --count "$PER_AGENT" "$SWEEP_PATH" &
  PIDS+=("$!")
done

for P in "${PIDS[@]}"; do
  wait "$P" || true
done
