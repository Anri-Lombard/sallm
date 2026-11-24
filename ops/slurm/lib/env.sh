#!/bin/bash

set_sallm_cluster_env() {
  local user_name="${USER:-$(id -un)}"

  : "${SALLM_HOME_DIR:=${HOME}}"
  : "${SALLM_SCRATCH_DIR:=${SCRATCH:-/scratch/${user_name}}}"
  : "${SALLM_REPO_DIR:=${SALLM_HOME_DIR}/masters/sallm}"
  : "${SALLM_SLURM_USER:=${user_name}}"

  export SALLM_HOME_DIR
  export SALLM_SCRATCH_DIR
  export SALLM_REPO_DIR
  export SALLM_SLURM_USER
  export SCRATCH="${SALLM_SCRATCH_DIR}"

  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]]; then
    export SALLM_REPO_DIR="$SLURM_SUBMIT_DIR"
  fi
}

detect_gpu_count() {
  local num_gpus="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
  if [[ -z "$num_gpus" || "$num_gpus" -le 0 ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      num_gpus=$(nvidia-smi -L | wc -l | tr -d ' ')
    else
      num_gpus=1
    fi
  fi
  echo "$num_gpus"
}

setup_sallm_runtime() {
  local spec="${1:-}"
  local sync_extras="${2:-auto}"

  export UV_CACHE_DIR="$SCRATCH/.cache/uv"
  export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
  export XDG_CACHE_HOME="$SCRATCH/.cache"
  export TRITON_CACHE_DIR="$SCRATCH/.triton/cache"
  export HF_HOME="$SCRATCH/hf"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export HF_METRICS_CACHE="$HF_HOME/metrics"
  export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  export WANDB_DIR="$SCRATCH/masters/sallm/wandb"
  export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
  export WANDB_CONFIG_DIR="$SCRATCH/.config/wandb"
  export JOB_LOG_DIR="$SCRATCH/masters/sallm/logs/jobs"
  export TOKENIZERS_PARALLELISM="true"
  export TORCH_DISTRIBUTED_TIMEOUT=7200
  export HYDRA_FULL_ERROR=1
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

  mkdir -p \
    "$TRITON_CACHE_DIR" \
    "$HF_DATASETS_CACHE" \
    "$HF_METRICS_CACHE" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$WANDB_DIR" \
    "$WANDB_CACHE_DIR" \
    "$WANDB_CONFIG_DIR" \
    "$JOB_LOG_DIR"

  load_hf_token || true

  if command -v module >/dev/null 2>&1; then
    module load python/miniconda3-py3.12 || true
  fi
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base=$(conda info --base)
    # shellcheck source=/dev/null
    source "${conda_base}/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx sallm-uv; then
      set +u
      conda activate sallm-uv
      set -u
    fi
  fi

  export PATH="$SALLM_HOME_DIR/.local/bin:$PATH"
  cd "$SALLM_REPO_DIR"

  local uv_args=(--frozen --inexact)
  if [[ "$sync_extras" == "all" || "$spec" == *gdn* || "$spec" == *deltanet* || "$spec" == *hybrid* ]]; then
    uv_args+=(--extra gpu --extra pure-gdn)
  fi
  uv sync "${uv_args[@]}"
  # shellcheck source=/dev/null
  source .venv/bin/activate

  if [[ "$spec" == *mamba* ]]; then
    if ! python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; from causal_conv1d import causal_conv1d_fn" 2>/dev/null; then
      echo "Installing mamba-ssm and causal-conv1d..."
      uv pip uninstall mamba-ssm causal-conv1d 2>/dev/null || true
      uv pip install --no-build-isolation mamba-ssm causal-conv1d 2>&1 || true
    fi
    if python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; from causal_conv1d import causal_conv1d_fn" 2>/dev/null; then
      export MAMBA_SCAN_IMPL="cuda"
    fi
  fi

  if [[ "$spec" == *xlstm* ]]; then
    if ! python -c "from transformers.utils import is_xlstm_available; assert is_xlstm_available()" 2>/dev/null; then
      echo "Installing xlstm kernels..."
      uv pip install xlstm 2>&1 | tail -5 || true
    fi
  fi
}
