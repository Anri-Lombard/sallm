#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name="sallm-eval"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

CONFIG_NAME="$1"
if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: sbatch $0 <config_name_without_yaml>"; exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONFIG_DIR="$REPO_ROOT/src/conf/eval"
CONFIG_NAME="${CONFIG_NAME%.yaml}"

export HYDRA_FULL_ERROR=1

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"

module load python/miniconda3-py3.12
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate sallm-ner

python -m sallm.main --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME"
