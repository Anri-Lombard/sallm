#!/bin/bash
##SBATCH --account=your-slurm-account
#SBATCH --partition=l40s
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --job-name="sallm-resume"
##SBATCH --mail-user=you@example.com
#SBATCH --mail-type=FAIL,END

set -euo pipefail

CONFIG_NAME="$1"
WANDB_RUN_ID="$2"
RESUME_CHECKPOINT="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/cluster_env.sh"
source "$SCRIPT_DIR/lib/auth.sh"
setup_sallm_cluster_env

if [[ -z "$CONFIG_NAME" || -z "$WANDB_RUN_ID" ]]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <config_name> <wandb_run_id> [resume_checkpoint_path]"
    exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="resume-${WANDB_RUN_ID}"
  mkdir -p logs
  exec > >(tee -a "logs/resume-${WANDB_RUN_ID}-${SLURM_JOB_ID}.out") 2>&1
fi

export PYTHONPATH="$SCRATCH/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
load_hf_token || true

echo "Setting up environment for resumed run ${WANDB_RUN_ID}..."
module load python/miniconda3-py3.12
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate sallm-uv
set -u

export PATH="$HOME/.local/bin:$PATH"
cd "$PROJECT_ROOT"
uv sync --frozen
source .venv/bin/activate

export HYDRA_FULL_ERROR=1

echo "Launching resumed training run..."
cmd=(
  accelerate launch
  --num_processes 4
  --num_machines 1
  --mixed_precision bf16
  --dynamo_backend no
  -m sallm.main
  --config-name "$CONFIG_NAME"
  "wandb.id=$WANDB_RUN_ID"
)

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  cmd+=("training.resume_from_checkpoint=$RESUME_CHECKPOINT")
fi

"${cmd[@]}"

echo "Resumed training run finished."
