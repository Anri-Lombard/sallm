#!/bin/bash
##SBATCH --account=your-slurm-account
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
##SBATCH --mail-user=you@example.com
#SBATCH --mail-type=FAIL,END

CFG="$1"; [[ -z "$CFG" ]] && { echo "Usage: sbatch $0 <config_name_without_yaml>"; exit 1; }
shift || true
EXTRA_ARGS=("$@")

SUBMIT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
LIB_DIR="${SUBMIT_ROOT%/}/scripts/lib"
if [[ ! -f "$LIB_DIR/cluster_env.sh" || ! -f "$LIB_DIR/auth.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  LIB_DIR="$SCRIPT_DIR/lib"
fi
source "$LIB_DIR/cluster_env.sh"
source "$LIB_DIR/auth.sh"
setup_sallm_cluster_env

CFG_NAME="${CFG##*/}"
JOB_NAME="ft-${CFG_NAME#mamba_}"
JOB_NAME="${JOB_NAME#llama_}"

export PYTHONPATH="$SCRATCH/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="$SCRATCH/.triton/cache"
export JOB_LOG_DIR="${JOB_LOG_DIR:-$SCRATCH/sallm/logs/jobs}"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$JOB_LOG_DIR"
# export TOKENIZERS_PARALLELISM="false"
export TOKENIZERS_PARALLELISM="true"
load_hf_token || true
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/sallm/wandb}"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_CONFIG_DIR="$SCRATCH/.config/wandb"
export TORCH_DISTRIBUTED_TIMEOUT=7200
export HYDRA_FULL_ERROR=1
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
# export NCCL_BLOCKING_WAIT=1

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME"
  exec > >(tee -a "$JOB_LOG_DIR/${JOB_NAME}-${SLURM_JOB_ID}.out") 2>&1
fi

echo "--- Storage Usage ---"
df -h "$HOME" "$SCRATCH" 2>/dev/null || true
echo "-------------------------------"

echo "--- Checking GPU availability ---"
nvidia-smi
echo "-------------------------------"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate sallm-uv
set -u

export PATH="$HOME/.local/bin:$PATH"
cd "$PROJECT_ROOT"
uv sync --frozen --inexact
source .venv/bin/activate

# Install/verify Mamba CUDA kernels (not in lockfile, must reinstall after uv sync)
echo "--- Mamba CUDA kernel status ---"
if python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; from causal_conv1d import causal_conv1d_fn" 2>/dev/null; then
    echo "✓ Mamba fast path (CUDA kernels) available"
else
    echo "Mamba kernels missing or ABI mismatch, rebuilding..."
    uv pip uninstall mamba-ssm causal-conv1d 2>/dev/null || true
    uv pip install --no-build-isolation mamba-ssm causal-conv1d 2>&1
    if ! python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; from causal_conv1d import causal_conv1d_fn" 2>/dev/null; then
        echo "ERROR: Mamba CUDA kernels failed to load. Aborting (native impl is too slow)."
        exit 1
    fi
    echo "✓ Mamba fast path (CUDA kernels) available"
fi
export MAMBA_SCAN_IMPL="cuda"
echo "-------------------------------"

# Set PyTorch CUDA allocation config to reduce fragmentation (optional)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# Debug info for distributed runs
echo "LOCAL_RANK=${LOCAL_RANK:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Determine number of processes to launch based on available GPUs
NUM_PROCS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
if [[ -z "$NUM_PROCS" || "$NUM_PROCS" -le 0 ]]; then
	if command -v nvidia-smi >/dev/null 2>&1; then
		NUM_PROCS=$(nvidia-smi -L | wc -l | tr -d ' ')
	else
		NUM_PROCS=1
	fi
fi

# Use dynamic port based on job ID to avoid conflicts when multiple jobs run on same node
MASTER_PORT=$((29500 + (${SLURM_JOB_ID:-0} % 1000)))

# pass explicit accelerate options to avoid its default-warning messages
accelerate launch \
	--num_processes "$NUM_PROCS" \
	--num_machines 1 \
	--mixed_precision bf16 \
	--dynamo_backend no \
	--main_process_port "$MASTER_PORT" \
	-m sallm.main --config-name "$CFG" "${EXTRA_ARGS[@]}"
