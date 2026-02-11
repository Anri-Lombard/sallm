#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

CFG="$1"; [[ -z "$CFG" ]] && { echo "Usage: sbatch $0 <config_name_without_yaml>"; exit 1; }

CFG_NAME="${CFG##*/}"
JOB_NAME="ft-${CFG_NAME#mamba_}"
JOB_NAME="${JOB_NAME#llama_}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME"
  mkdir -p logs
  exec > >(tee -a "logs/${JOB_NAME}-${SLURM_JOB_ID}.out") 2>&1
fi

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"
export PYTHONPATH="$SCRATCH/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="$SCRATCH/.triton/cache"
mkdir -p "$TRITON_CACHE_DIR"
# export TOKENIZERS_PARALLELISM="false"
export TOKENIZERS_PARALLELISM="true"
export HF_TOKEN=$(cat "$HOME/.huggingface/token" 2>/dev/null || echo "")
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export TORCH_DISTRIBUTED_TIMEOUT=7200
export HYDRA_FULL_ERROR=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
# export NCCL_BLOCKING_WAIT=1

echo "--- Storage Usage ---"
df -h /home /scratch 2>/dev/null || true
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
cd "$HOME/masters/sallm"
uv sync --frozen --inexact
source .venv/bin/activate

# Install Mamba CUDA kernels (not in lockfile, must reinstall after uv sync)
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
	-m sallm.main --config-name "$CFG"
