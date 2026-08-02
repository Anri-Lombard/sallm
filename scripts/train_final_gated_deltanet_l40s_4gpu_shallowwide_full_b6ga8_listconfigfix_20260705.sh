#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --account=l40sfree
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name="gdn33w-b6-l40s-r2"
#SBATCH --mail-type=FAIL,END

set -euo pipefail

export MKL_INTERFACE_LAYER=LP64,INTEL64

CONFIG="base/gated_deltanet_125m_shallowwide_full_4x40_pack_nogc_b6ga8_20260628.yaml"
SCRIPT_DIR="$PWD/scripts"
source "$SCRIPT_DIR/lib/env.sh"
source "$SCRIPT_DIR/lib/auth.sh"
set_sallm_cluster_env

export PYTHONPATH="$SCRATCH/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/hf"
load_hf_token || true
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="$SCRATCH/.triton"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.cache/torchinductor"
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300
export CUDA_CACHE_PATH="$SCRATCH/.cache/nv"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"

export PATH="$SALLM_HOME_DIR/.local/bin:$PATH"
cd "$SALLM_REPO_DIR"
mkdir -p /scratch/lmbanr001/masters/sallm_wandb
uv sync --frozen --inexact
source .venv/bin/activate

echo "--- Checking GatedDeltaNet support ---"
python - <<'PY'
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

print("Qwen3Next support available")
try:
    import causal_conv1d  # noqa: F401
    import fla  # noqa: F401
except ImportError:
    print("FLA fast path unavailable; Transformers will use torch fallback")
else:
    print("FLA fast path available")
PY
python - <<'PY'
import causal_conv1d
import fla

print("Required fast kernels import OK")
PY
echo "-------------------------------"

RUN_ID="sallm-gated-deltanet-125m-shallowwide-full-4x40-pack-nogc-b6ga8-listconfigfix-20260705"
echo "Launching full shallow/wide GatedDeltaNet run with $CONFIG as $RUN_ID"
accelerate launch --num_processes=4 --num_machines=1 --mixed_precision=bf16 -m sallm.main \
  --config-name "$CONFIG" \
  wandb.name="125m-shallowwide-4x40-pack-nogc-b6ga8-full-listconfigfix-20260705" \
  training.output_dir="\${oc.env:SCRATCH}/masters/sallm/checkpoints/$RUN_ID" \
  training.logging_dir="\${oc.env:SCRATCH}/masters/sallm/logs/$RUN_ID"
