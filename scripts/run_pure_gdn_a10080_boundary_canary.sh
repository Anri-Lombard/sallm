#!/bin/bash
# Resume a diagnostic copy through the reproducible finite-stream boundary.
#SBATCH --partition=a100
#SBATCH --account=nlpgroup80
#SBATCH --gres=gpu:ampere80:2
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pure-gdn-boundary-canary
#SBATCH --mail-type=FAIL,END

set -euo pipefail

CONFIG="base/gated_deltanet_125m_pure.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$SCRIPT_DIR/lib/env.sh" || ! -f "$SCRIPT_DIR/lib/auth.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$HOME/masters/sallm}/scripts"
fi
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
export WANDB_MODE=disabled

load_hf_token || true
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

module load python/miniconda3-py3.12
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base)
  set +u
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx sallm-uv; then
    conda activate sallm-uv
  fi
  set -u
fi

export PATH="$SALLM_HOME_DIR/.local/bin:$PATH"
cd "$SALLM_REPO_DIR"
uv sync --extra pure-gdn --frozen --inexact
source .venv/bin/activate

SOURCE_CHECKPOINT="$SCRATCH/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/checkpoint-29000"
OUTPUT_DIR="$SCRATCH/masters/sallm/checkpoints/sallm-pure-gdn-125m-diagnostics/stream-boundary-20260804"
LOG_DIR="$SCRATCH/masters/sallm/logs/sallm-pure-gdn-125m-diagnostics/stream-boundary-20260804"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/pretrain.out") 2>&1

echo "Pure GDN finite-stream boundary canary"
echo "Read-only source checkpoint: $SOURCE_CHECKPOINT"
echo "Diagnostic output: $OUTPUT_DIR"
test -f "$SOURCE_CHECKPOINT/trainer_state.json"
python scripts/verify_pure_gdn_runtime.py --mode preflight --config "src/conf/$CONFIG"

accelerate launch --num_processes=2 --num_machines=1 --mixed_precision=bf16 \
  -m sallm.main --config-name "$CONFIG" \
  "base.training.output_dir=$OUTPUT_DIR" \
  "base.training.logging_dir=$LOG_DIR" \
  "base.training.resume_from_checkpoint=$SOURCE_CHECKPOINT" \
  "base.training.max_steps=30010" \
  "base.training.eval_strategy=no" \
  "base.training.save_strategy=no" \
  "base.training.report_to=none"

python scripts/verify_pure_gdn_runtime.py \
  --mode post-checkpoint \
  --config "src/conf/$CONFIG" \
  --checkpoint "$OUTPUT_DIR/final_model"
