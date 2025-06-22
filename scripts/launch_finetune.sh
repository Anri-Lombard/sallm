#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --job-name="sallm-ft"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

CFG="$1"; [[ -z "$CFG" ]] && { echo "Usage: sbatch $0 <config.yaml>"; exit 1; }

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sallm-ner

accelerate launch -m sallm.main --config_path "$CFG"
