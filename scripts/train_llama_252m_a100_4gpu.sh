#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere:4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=sallm-llama-252m
#SBATCH --mail-type=FAIL,END

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env.sh"
source "$SCRIPT_DIR/lib/auth.sh"
set_sallm_cluster_env
require_hf_token
export HF_HOME="$SCRATCH/hf"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

bash "$HOME/masters/sallm/scripts/train_final_model.sh" base/llama_252m "$@"

source "$HOME/masters/sallm/.venv/bin/activate"
python - <<'PY'
from huggingface_hub import HfApi

repo_id = "anrilombard/sallm-llama-252m"
api = HfApi()
api.create_repo(repo_id, private=True, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    folder_path="/scratch/lmbanr001/masters/sallm/checkpoints/sallm-llama-252m/final_model",
)
PY
