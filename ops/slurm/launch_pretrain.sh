#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=sallm-pretrain
#SBATCH --mail-type=FAIL,END

set -euo pipefail

CONFIG_PATH="${1:-src/conf/pretrain/mzansilm_wsd.json}"
DATA_DIR="${2:-${MP_DATA_DIR:-}}"
OUT_DIR="${3:-${MP_OUT_DIR:-}}"
if [[ "$#" -ge 3 ]]; then
  shift 3
elif [[ "$#" -gt 0 ]]; then
  shift "$#"
fi

if [[ -z "$DATA_DIR" || -z "$OUT_DIR" ]]; then
  echo "Usage: sbatch $0 <config_json> <packed_data_dir> <output_dir> [extra sallm.pretrain args...]" >&2
  echo "Example: sbatch $0 src/conf/pretrain/gdn_wsd.json \$SCRATCH/data/packed_bin \$SCRATCH/runs/gdn" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env.sh"
source "$SCRIPT_DIR/lib/auth.sh"
set_sallm_cluster_env

echo "Using pretrain config: ${CONFIG_PATH}"

export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export TRITON_CACHE_DIR="$SCRATCH/.triton/cache"
mkdir -p "$TRITON_CACHE_DIR"
load_hf_token || true
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

export MP_TOKENIZER="${MP_TOKENIZER:-$SALLM_REPO_DIR/tokenizer/sallm_bpe_tokenizer}"
export MP_VAL_PARQUET="${MP_VAL_PARQUET:-$SCRATCH/data/validation.parquet}"

echo "Setting up environment..."
module load python/miniconda3-py3.12
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx sallm-uv; then
  set +u
  conda activate sallm-uv
  set -u
else
  echo "Conda environment sallm-uv is unavailable; using the repository .venv."
fi

export PATH="$SALLM_HOME_DIR/.local/bin:$PATH"
cd "$SALLM_REPO_DIR"
uv sync --frozen --extra gpu --extra pure-gdn
source .venv/bin/activate
echo "Environment ready."

echo "Launching matched pretraining..."

NUM_PROCESSES="${SLURM_GPUS_ON_NODE:-2}"
torchrun --nproc_per_node="$NUM_PROCESSES" -m sallm.pretrain \
  --config "$CONFIG_PATH" \
  --data "$DATA_DIR" \
  --out "$OUT_DIR" \
  --stack fast \
  "$@"

echo "Pretraining finished."
