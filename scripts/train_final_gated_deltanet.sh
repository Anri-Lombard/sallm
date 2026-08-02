#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --account=l40sfree
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name="sallm-gdn-hybrid"
#SBATCH --mail-type=FAIL,END

set -euo pipefail

export MKL_INTERFACE_LAYER=LP64,INTEL64

CONFIG="base/gated_deltanet_125m_shallowwide_full_4x40_pack_nogc_b6ga8_listconfigfix_dtypefix_20260707.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env.sh"
source "$SCRIPT_DIR/lib/auth.sh"
set_sallm_cluster_env

export PYTHONPATH="$SCRATCH/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/hf"
load_hf_token || true
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate sallm-uv
set -u

export PATH="$SALLM_HOME_DIR/.local/bin:$PATH"
cd "$SALLM_REPO_DIR"
uv sync --frozen
source .venv/bin/activate

echo "--- Checking GDN–Attention Hybrid (Qwen3Next implementation) support ---"
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
echo "-------------------------------"

echo "Launching GDN–Attention Hybrid (Qwen3Next implementation) training with $CONFIG"
accelerate launch --mixed_precision=bf16 -m sallm.main --config-name "$CONFIG"
