#!/bin/bash
# Run manually with sbatch; this script never submits itself.
#SBATCH --partition=a100
#SBATCH --account=nlpgroup80
#SBATCH --gres=gpu:ampere80:2
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=sallm-pure-gdn-canary
#SBATCH --mail-type=FAIL,END

set -euo pipefail

CONFIG="base/gated_deltanet_125m_pure_canary.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env.sh"
source "$SCRIPT_DIR/lib/auth.sh"
set_sallm_cluster_env

export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export TRITON_CACHE_DIR="$SCRATCH/.triton/cache"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export SALLM_ASSERT_TRAIN_BATCH_LENGTH=2048
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

load_hf_token || true
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

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
uv sync --extra pure-gdn --frozen --inexact
source .venv/bin/activate

if ! python - <<'PY'
from importlib.metadata import version

import causal_conv1d

assert version("causal-conv1d") == "1.6.2.post1"
print("causal-conv1d 1.6.2.post1 verified")
PY
then
  uv pip install --no-build-isolation "causal-conv1d==1.6.2.post1"
fi

python - <<'PY'
from importlib.metadata import version

import causal_conv1d
from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

assert version("flash-linear-attention") == "0.5.1"
assert version("causal-conv1d") == "1.6.2.post1"
print("Pure FLA GatedDeltaNet dependencies and chunk kernel import verified")
PY

RUN_ID="${SALLM_PURE_GDN_CANARY_RUN_ID:-a10080-${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="$SCRATCH/masters/sallm/checkpoints/sallm-pure-gdn-125m-canary/$RUN_ID"
LOG_DIR="$SCRATCH/masters/sallm/logs/sallm-pure-gdn-125m-canary/$RUN_ID"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/canary.out") 2>&1

echo "Running pure FLA GatedDeltaNet canary: $RUN_ID"
echo "Output: $OUTPUT_DIR"
python scripts/verify_pure_gdn_runtime.py --mode preflight --config "src/conf/$CONFIG"

if [[ -d "$OUTPUT_DIR/final_model" ]]; then
  echo "Existing final_model found; skipping completed training step."
else
  resume_args=()
  latest_checkpoint=""
  for candidate in "$OUTPUT_DIR"/checkpoint-*; do
    [[ -d "$candidate" ]] || continue
    if [[ -z "$latest_checkpoint" || "${candidate##*-}" -gt "${latest_checkpoint##*-}" ]]; then
      latest_checkpoint="$candidate"
    fi
  done
  if [[ -n "$latest_checkpoint" ]]; then
    resume_args+=("training.resume_from_checkpoint=$latest_checkpoint")
    echo "Resuming from $latest_checkpoint"
  fi

  accelerate launch --num_processes=2 --num_machines=1 --mixed_precision=bf16 \
    -m sallm.main --config-name "$CONFIG" \
    "training.output_dir=$OUTPUT_DIR" \
    "training.logging_dir=$LOG_DIR" \
    "${resume_args[@]}"
fi

python scripts/verify_pure_gdn_runtime.py \
  --mode post-checkpoint \
  --config "src/conf/$CONFIG" \
  --checkpoint "$OUTPUT_DIR/final_model"
