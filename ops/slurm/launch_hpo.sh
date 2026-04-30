#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=FAIL,END

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env.sh"
set_sallm_cluster_env
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]]; then
  export SALLM_REPO_DIR="$SLURM_SUBMIT_DIR"
fi

SWEEP_ARG="${1:-}"
COUNT="${2:-50}"
SWEEP_DIR="src/conf/sweeps"

SWEEP_NAME="${SWEEP_ARG%.yaml}"
SWEEP_NAME="${SWEEP_NAME##*/}"
JOB_NAME="hpo-${SWEEP_NAME#mamba_}"
JOB_NAME="${JOB_NAME#llama_}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME"
  mkdir -p logs
  exec > >(tee -a "logs/${JOB_NAME}-${SLURM_JOB_ID}.out") 2>&1
fi

if [[ -z "$SWEEP_ARG" ]]; then
  echo "Usage: sbatch $0 <sweep_yaml_or_name_without_yaml> [count]" >&2
  echo "Examples: sbatch $0 llama_afrihg_xho 20" >&2
  echo "          sbatch $0 llama_t2x_xho 10" >&2
  exit 1
fi

# Note: Don't prepend scratch to PYTHONPATH - venv has patched transformers for xLSTM
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export TRITON_CACHE_DIR="$SCRATCH/.triton/cache"
mkdir -p "$TRITON_CACHE_DIR"
export TOKENIZERS_PARALLELISM="true"
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export HF_TOKEN=$(cat "${HF_TOKEN_FILE:-$SALLM_HOME_DIR/.huggingface/token}" 2>/dev/null || echo "")
export TORCH_DISTRIBUTED_TIMEOUT=7200
export HYDRA_FULL_ERROR=1

echo "--- Storage Usage ---"
df -h /home /scratch 2>/dev/null || true
echo "-------------------------------"

echo "--- Checking GPU availability ---"
nvidia-smi || true
echo "-------------------------------"

module load python/miniconda3-py3.12
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sallm-uv
set -u

export PATH="$SALLM_HOME_DIR/.local/bin:$PATH"
cd "$SALLM_REPO_DIR"
uv sync --frozen --inexact
source .venv/bin/activate

# Install CUDA kernels based on model type
echo "--- CUDA kernel status ---"
IS_MAMBA=false
IS_XLSTM=false
[[ "$SWEEP_NAME" == *mamba* ]] && IS_MAMBA=true
[[ "$SWEEP_NAME" == *xlstm* ]] && IS_XLSTM=true

if $IS_MAMBA; then
  if python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" 2>/dev/null; then
    echo "✓ Mamba fast path (CUDA kernels) available"
  else
    echo "Mamba kernels missing or ABI mismatch, rebuilding..."
    uv pip uninstall mamba-ssm causal-conv1d 2>/dev/null || true
    uv pip install --no-build-isolation mamba-ssm causal-conv1d 2>&1 || true
    if python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" 2>/dev/null; then
      echo "✓ Mamba fast path (CUDA kernels) available"
    else
      echo "ℹ Using HF Transformers native Mamba implementation"
    fi
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

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

echo "LOCAL_RANK=${LOCAL_RANK:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
if [[ -z "$NUM_GPUS" || "$NUM_GPUS" -le 0 ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
  else
    NUM_GPUS=1
  fi
fi

SWEEP_FILE=""
if [[ -f "$SWEEP_ARG" ]]; then
  SWEEP_FILE="$SWEEP_ARG"
elif [[ -f "${SWEEP_DIR}/${SWEEP_ARG}.yaml" ]]; then
  SWEEP_FILE="${SWEEP_DIR}/${SWEEP_ARG}.yaml"
elif [[ "$SWEEP_ARG" == *.yaml && -f "${SWEEP_DIR}/${SWEEP_ARG}" ]]; then
  SWEEP_FILE="${SWEEP_DIR}/${SWEEP_ARG}"
else
  echo "Sweep file not found for arg: $SWEEP_ARG" >&2
  exit 1
fi

if [[ ! -f "$SWEEP_FILE" ]]; then
  echo "Sweep file not found: $SWEEP_FILE" >&2
  exit 1
fi

if [[ -z "${TOKENIZER_PATH:-}" ]]; then
  TOKENIZER_PATH=$(python -m sallm.hpo.run --tokenizer-path)
  if [[ -n "$TOKENIZER_PATH" ]]; then
    export TOKENIZER_PATH
  fi
fi

CREATE_OUT=$(wandb sweep "$SWEEP_FILE" 2>&1)
echo "$CREATE_OUT"

SWEEP_PATH=$(printf "%s\n" "$CREATE_OUT" | sed -nE 's/^.*Run sweep agent with: wandb agent ([^[:space:]]+).*$/\1/p' | tail -n1)
if [[ -z "$SWEEP_PATH" ]]; then
  SWEEP_PATH=$(printf "%s\n" "$CREATE_OUT" | sed -nE 's@.*https?://wandb\.ai/([^/]+)/([^/]+)/sweeps/([A-Za-z0-9_-]+).*@\1/\2/\3@p' | tail -n1)
fi
if [[ -z "$SWEEP_PATH" ]]; then
  echo "Failed to parse sweep ID from wandb output" >&2
  exit 1
fi

AGENTS_TO_RUN=$NUM_GPUS
if [[ "$COUNT" -lt "$AGENTS_TO_RUN" ]]; then
  AGENTS_TO_RUN="$COUNT"
fi

BASE_PER_AGENT=$((COUNT / AGENTS_TO_RUN))
REMAINDER=$((COUNT % AGENTS_TO_RUN))

echo "Launching $AGENTS_TO_RUN agents across $NUM_GPUS GPUs (total=$COUNT runs)"

PIDS=()
for IDX in $(seq 0 $((AGENTS_TO_RUN - 1))); do
  PER_AGENT=$BASE_PER_AGENT
  if [[ $IDX -lt $REMAINDER ]]; then
    PER_AGENT=$((PER_AGENT + 1))
  fi
  if [[ $PER_AGENT -eq 0 ]]; then
    continue
  fi
  export CUDA_VISIBLE_DEVICES="$IDX"
  echo "GPU $IDX → wandb agent --count $PER_AGENT $SWEEP_PATH"
  wandb agent --count "$PER_AGENT" "$SWEEP_PATH" &
  PIDS+=("$!")
done

EXIT_CODE=0
for P in "${PIDS[@]}"; do
  if ! wait "$P"; then
    EXIT_CODE=1
  fi
done

exit "$EXIT_CODE"
