#!/bin/bash
#SBATCH --account l40sfree
#SBATCH --partition=l40s
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --job-name="sallm-hpo-agent"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

set -e

SWEEP_CONFIG_PATH="${1}"

if [ -z "$SWEEP_CONFIG_PATH" ]; then
    echo "Error: Path to sweep configuration YAML not provided."
    echo "Usage: sbatch $0 <path/to/your/sweep.yaml>"
    exit 1
fi

if [ ! -f "$SWEEP_CONFIG_PATH" ]; then
    echo "Error: Sweep configuration file not found at: $SWEEP_CONFIG_PATH"
    exit 1
fi

echo "Setting up environment..."
module load python/miniconda3-py3.12
conda activate sallm-ner
echo "Environment ready."

SWEEP_ID_FILE="$(dirname "$SWEEP_CONFIG_PATH")/.$(basename "$SWEEP_CONFIG_PATH" .yaml).id"

if [ ! -f "$SWEEP_ID_FILE" ]; then
    echo "No sweep ID file found. Creating a new sweep..."

    WANDB_OUTPUT=$(wandb sweep "$SWEEP_CONFIG_PATH" 2>&1)
    echo "$WANDB_OUTPUT"
    SWEEP_ID=$(echo "$WANDB_OUTPUT" | grep "Creating sweep with ID:" | sed 's/.*: //')

    if [ -z "$SWEEP_ID" ]; then
        echo "Error: Failed to create sweep or parse ID. Aborting."
        exit 1
    fi

    echo "$SWEEP_ID" > "$SWEEP_ID_FILE"
    echo "New sweep created with ID: $SWEEP_ID. Saved to $SWEEP_ID_FILE"
else
    SWEEP_ID=$(cat "$SWEEP_ID_FILE")
    echo "Found existing sweep ID: $SWEEP_ID from $SWEEP_ID_FILE"
fi

echo "Checking for crashed trials to resume..."
python scripts/resume_crashed_trials.py --sweep_id "$SWEEP_ID" --config_path "$SWEEP_CONFIG_PATH"
RECOVERY_STATUS=$?

if [ $RECOVERY_STATUS -eq 1 ]; then
    echo "No crashed trials found. Starting a new agent to generate trials..."
    export WANDB_SWEEP_ID=$SWEEP_ID
    accelerate launch $(which wandb) agent "$WANDB_SWEEP_ID"
else
    echo "Recovery job was submitted by the helper script. This agent will now exit."
fi

echo "HPO launcher script finished."