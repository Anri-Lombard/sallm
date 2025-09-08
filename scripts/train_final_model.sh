#!/bin/bash
#SBATCH --account=a100
#SBATCH --partition=a100
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name="sallm-llama-mobilellama"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

set -e

CONFIG_PATH="configs/base/llama_125m.yaml"

echo "Using configuration: ${CONFIG_PATH}"

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"

echo "Setting up environment..."
module load python/miniconda3-py3.12
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate sallm-ner
echo "Environment ready."

echo "Launching final training run..."

accelerate launch --num_processes 4 --num_machines 1 --mixed_precision bf16 --dynamo_backend no src/main/sallm/main.py \
    --config_path "$CONFIG_PATH"

echo "Final training run finished."