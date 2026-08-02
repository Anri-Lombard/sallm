#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL,END

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$SCRIPT_DIR/lib/env.sh" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}/scripts" "$HOME/masters/sallm/scripts"; do
    if [[ -f "$candidate/lib/env.sh" ]]; then
      SCRIPT_DIR="$candidate"
      break
    fi
  done
fi
if [[ ! -f "$SCRIPT_DIR/lib/env.sh" ]]; then
  echo "ERROR: Could not locate scripts/lib/env.sh." >&2
  exit 1
fi
source "$SCRIPT_DIR/lib/env.sh"
set_sallm_cluster_env
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]]; then
  export SALLM_REPO_DIR="$SLURM_SUBMIT_DIR"
fi

CONFIG_NAME="$1"
if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: sbatch $0 <config_name_without_yaml>"; exit 1
fi
shift || true
EXTRA_ARGS=("$@")

if [[ "$CONFIG_NAME" != */* ]]; then
    CONFIG_NAME="eval/$CONFIG_NAME"
fi

CONFIG_GROUP="${CONFIG_NAME%%/*}"
NORMALIZED_ARGS=()
for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
        -*) NORMALIZED_ARGS+=("$arg") ;;
        hydra.*|"$CONFIG_GROUP".*) NORMALIZED_ARGS+=("$arg") ;;
        ++*)
            arg_body="${arg#++}"
            if [[ "$arg_body" == "$CONFIG_GROUP."* ]]; then
                NORMALIZED_ARGS+=("++${arg_body}")
            else
                NORMALIZED_ARGS+=("++${CONFIG_GROUP}.${arg_body}")
            fi
            ;;
        +*)
            arg_body="${arg#+}"
            if [[ "$arg_body" == "$CONFIG_GROUP."* ]]; then
                NORMALIZED_ARGS+=("$arg")
            else
                NORMALIZED_ARGS+=("+${CONFIG_GROUP}.${arg_body}")
            fi
            ;;
        *) NORMALIZED_ARGS+=("${CONFIG_GROUP}.${arg}") ;;
    esac
done
EXTRA_ARGS=("${NORMALIZED_ARGS[@]}")

CFG_NAME="${CONFIG_NAME##*/}"
JOB_NAME="eval-${CFG_NAME#mamba_}"
JOB_NAME="${JOB_NAME#llama_}"

export HYDRA_FULL_ERROR=1

export JOB_LOG_DIR="$SCRATCH/masters/sallm/logs/jobs"
# Note: Don't prepend scratch to PYTHONPATH - venv has patched transformers for xLSTM
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_TOKEN=$(cat "${HF_TOKEN_FILE:-$SALLM_HOME_DIR/.huggingface/token}" 2>/dev/null || echo "")
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRITON_CACHE_DIR="$SCRATCH/.triton/cache"
export WANDB_DIR="$SCRATCH/masters/sallm/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_CONFIG_DIR="$SCRATCH/.config/wandb"
mkdir -p "$TRITON_CACHE_DIR" "$JOB_LOG_DIR"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME"
  exec > >(tee -a "$JOB_LOG_DIR/${JOB_NAME}-${SLURM_JOB_ID}.out") 2>&1
fi

echo "--- Storage Usage ---"
df -h /home /scratch 2>/dev/null || true
echo "-------------------------------"

if command -v module >/dev/null 2>&1; then
    module load python/miniconda3-py3.12
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx sallm-uv; then
        conda activate sallm-uv
    else
        echo "ℹ conda environment sallm-uv is unavailable; using the repository .venv."
    fi
fi

export PATH="$SALLM_HOME_DIR/.local/bin:$PATH"
cd "$SALLM_REPO_DIR"
if command -v uv >/dev/null 2>&1; then
    uv sync --frozen --inexact
fi
source .venv/bin/activate

# Install CUDA kernels based on model type
echo "--- CUDA kernel status ---"
IS_MAMBA=false
IS_XLSTM=false
[[ "$CONFIG_NAME" == *mamba* ]] && IS_MAMBA=true
[[ "$CONFIG_NAME" == *xlstm* ]] && IS_XLSTM=true

if $IS_MAMBA && [[ "${SALLM_SKIP_MAMBA_KERNEL_CHECK:-0}" != 1 ]]; then
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
    export MAMBA_SCAN_IMPL="cuda"
elif $IS_MAMBA; then
    echo "Skipping Mamba kernels for an architecture override."
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

if [[ "$CONFIG_NAME" == *gdn* || "$CONFIG_NAME" == *gated_deltanet* ]]; then
    # TileLang can be importable but unusable when its native library root is absent.
    export FLA_DISABLE_BACKEND_DISPATCH=1
    echo "GDN evaluation: disabling FLA backend dispatch; using the reference path."
fi

python -m sallm.main --config-name "$CONFIG_NAME" "${EXTRA_ARGS[@]}"
