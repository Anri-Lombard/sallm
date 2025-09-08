#!/bin/bash
#SBATCH --account=a100
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --job-name="sallm-pipeline"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

CONFIG_NAME="$1"
if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: sbatch $0 <config_name_without_yaml>"; exit 1
fi

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sallm-ner

python -m sallm.main --config-name "$CONFIG_NAME"
