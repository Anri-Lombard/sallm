#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=hpo-ner_all-resume
#SBATCH --mail-type=FAIL,END

set -euo pipefail

SWEEP_PATH="${1:-}"
ARCHITECTURE="${2:-}"
COUNT="${3:-43}"

if [[ -z "$SWEEP_PATH" || -z "$ARCHITECTURE" ]]; then
  echo "Usage: sbatch $0 <sweep_path> <architecture> [count]" >&2
  echo "Architectures: gated_deltanet (gdn), mamba2 (mamba), xlstm, llama" >&2
  echo "Example: sbatch $0 anri-lombard/sallm-ft/z0vyuasg gated_deltanet 43" >&2
  exit 1
fi

case "$ARCHITECTURE" in
  gated_deltanet|gdn) ARCHITECTURE="gated_deltanet" ;;
  mamba2|mamba) ARCHITECTURE="mamba2" ;;
  xlstm|llama) ;;
  *)
    echo "Unknown architecture: $ARCHITECTURE" >&2
    exit 1
    ;;
esac

if [[ ! "$COUNT" =~ ^[0-9]+$ ]] || (( 10#$COUNT <= 0 )); then
  echo "Count must be a positive integer: $COUNT" >&2
  exit 1
fi
COUNT=$((10#$COUNT))

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

SWEEP_ID="${SWEEP_PATH##*/}"
mkdir -p logs
exec > >(tee -a "logs/hpo-resume-${SWEEP_ID}-${SLURM_JOB_ID}.out") 2>&1

# Note: Don't set PYTHONPATH - venv has patched transformers
export TRITON_CACHE_DIR="$SCRATCH/.triton/cache"
mkdir -p "$TRITON_CACHE_DIR"
export TOKENIZERS_PARALLELISM="true"
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_TOKEN=$(cat "${HF_TOKEN_FILE:-$SALLM_HOME_DIR/.huggingface/token}")
export TORCH_DISTRIBUTED_TIMEOUT=7200
export HYDRA_FULL_ERROR=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"

module load python/miniconda3-py3.12
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base)
  set +u
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx sallm-uv; then
    conda activate sallm-uv
  else
    echo "Conda environment sallm-uv is unavailable; using the repository .venv."
  fi
  set -u
fi

export PATH="$SALLM_HOME_DIR/.local/bin:$PATH"
cd "$SALLM_REPO_DIR"
uv sync --frozen --inexact
source .venv/bin/activate

case "$ARCHITECTURE" in
  gated_deltanet)
    python -c "import causal_conv1d, fla; from fla.ops.gated_delta_rule import chunk_gated_delta_rule" \
      || { echo "ERROR: GatedDeltaNet fast kernels are unavailable." >&2; exit 1; }
    echo "✓ GatedDeltaNet fast path available"
    ;;
  mamba2)
    # Install/verify Mamba CUDA kernels (not in lockfile, must reinstall after uv sync)
    # Wheels are cached on cluster scratch so this should stay quick.
    echo "--- Mamba CUDA kernel status ---"
    if ! python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; from causal_conv1d import causal_conv1d_fn" 2>/dev/null; then
        echo "Installing mamba-ssm and causal-conv1d from cached wheels..."
        uv pip install --no-build-isolation mamba-ssm causal-conv1d 2>&1 | tail -5 || true
    fi
    python -c "
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from causal_conv1d import causal_conv1d_fn
    print('✓ Mamba fast path (CUDA kernels) available: selective_scan + causal_conv1d')
except ImportError as e:
    print(f'ℹ Mamba CUDA kernels unavailable: {e}')
    raise SystemExit(1)
"
    echo "-------------------------------"
    ;;
  xlstm|llama)
    echo "Skipping CUDA kernel preflight for $ARCHITECTURE; no architecture-specific kernel requirement."
    ;;
esac

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export WANDB_AGENT_MAX_INITIAL_FAILURES="${WANDB_AGENT_MAX_INITIAL_FAILURES:-100}"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-2}}"
AGENTS_TO_RUN=$NUM_GPUS
if [[ "$COUNT" -lt "$AGENTS_TO_RUN" ]]; then
  AGENTS_TO_RUN="$COUNT"
fi
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
