#!/bin/bash
#SBATCH --account l40sfree
#SBATCH --partition=l40s
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name="sallm-eval"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

CONFIG_PATH="$1"
if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: sbatch $0 <configs/eval/models/...yaml>"; exit 1
fi

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"

module load python/miniconda3-py3.12
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate sallm-ner

python -m sallm.main --config_path "$CONFIG_PATH"
