#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --job-name="sallm-resume"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

# set -e

CONFIG_PATH="$1"
WANDB_RUN_ID="$2"

if [ -z "$CONFIG_PATH" ] || [ -z "$WANDB_RUN_ID" ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <config_path> <wandb_run_id>"
    exit 1
fi

#SBATCH --job-name="sallm-resume-${WANDB_RUN_ID}"

echo "Setting up environment for resumed run ${WANDB_RUN_ID}..."
module load python/miniconda3-py3.12
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate sallm-ner

export HYDRA_FULL_ERROR=1

echo "Launching resumed training run..."
accelerate launch --num_processes 4 --num_machines 1 --mixed_precision bf16 --dynamo_backend no src/main/sallm/main.py \
    --config_path "$CONFIG_PATH" \
    --wandb_run_id "$WANDB_RUN_ID"

echo "Resumed training run finished."
