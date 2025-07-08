#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name="sallm-ft"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

CFG="$1"; [[ -z "$CFG" ]] && { echo "Usage: sbatch $0 <config_name_without_yaml>"; exit 1; }

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"


echo "--- Checking GPU availability ---"
nvidia-smi
echo "-------------------------------"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sallm-ner

accelerate launch -m sallm.main --config-name "$CFG"
