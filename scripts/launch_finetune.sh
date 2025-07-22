#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name="sallm-ft"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

CFG="$1"; [[ -z "$CFG" ]] && { echo "Usage: sbatch $0 <config_name_without_yaml>"; exit 1; }

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"
export TOKENIZERS_PARALLELISM="false"

echo "--- Checking GPU availability ---"
nvidia-smi
echo "-------------------------------"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sallm-ner

accelerate launch --num_processes 2 -m sallm.main --config-name "$CFG"
