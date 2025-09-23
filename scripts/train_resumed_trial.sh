#!/bin/bash
#SBATCH --account=a100
#SBATCH --partition=a100
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --job-name="sallm-resume"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

# ----------------------------------------------------------------------------
# Resume a previously-started training run using a WandB run id
#
# Purpose
#   Resume training from an existing WandB run by passing the run id and the
#   configuration path. The script sets up the conda environment and launches
#   the training entrypoint with the provided run id to allow continuation.
#
# Usage
#   sbatch train_resumed_trial.sh <config_path> <wandb_run_id>
#
# Notes
#   - The script updates the Slurm job name to include the run id for easier
#     tracking.
# ----------------------------------------------------------------------------

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
