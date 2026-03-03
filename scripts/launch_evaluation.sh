#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

CONFIG_NAME="$1"
if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: sbatch $0 <config_name_without_yaml>"; exit 1
fi
shift || true
EXTRA_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/auth.sh"

if [[ "$CONFIG_NAME" != */* ]]; then
    CONFIG_NAME="eval/$CONFIG_NAME"
fi

CFG_NAME="${CONFIG_NAME##*/}"
JOB_NAME="eval-${CFG_NAME#llama_}"

export HYDRA_FULL_ERROR=1

export SCRATCH="${SCRATCH:-/scratch/lmbanr001}"
export JOB_LOG_DIR="$SCRATCH/masters/sallm/logs/jobs"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export TMPDIR="${SLURM_TMPDIR:-$SCRATCH/tmp}"
load_hf_token || true
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_EVALUATE_CACHE="$TMPDIR/hf/evaluate"
export HF_METRICS_CACHE="$HF_EVALUATE_CACHE/metrics"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRITON_CACHE_DIR="$SCRATCH/.triton/cache"
export WANDB_DIR="$TMPDIR/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_CONFIG_DIR="$SCRATCH/.config/wandb"
mkdir -p "$TRITON_CACHE_DIR" "$JOB_LOG_DIR"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
mkdir -p "$TMPDIR" "$HF_EVALUATE_CACHE" "$HF_METRICS_CACHE"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME"
  exec > >(tee -a "$JOB_LOG_DIR/${JOB_NAME}-${SLURM_JOB_ID}.out") 2>&1
fi

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
cd "$REPO_ROOT"
uv sync --frozen --inexact
source .venv/bin/activate

# Set PyTorch CUDA allocation config
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

python -m sallm.main --config-name "$CONFIG_NAME" "${EXTRA_ARGS[@]}"
