#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=2
#SBATCH --time=48:00:00
#SBATCH --job-name="sallm-mamba-final"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

# TODO make one script instead of seperate final scripts per architecture

set -euo pipefail

export MKL_INTERFACE_LAYER=LP64,INTEL64

CONFIG="configs/base/mamba_125m.yaml"

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sallm-ner

export MAMBA_SCAN_IMPL="cuda"
export TORCHDYNAMO_DISABLE="1"

echo "Launching training with $CONFIG"
accelerate launch --mixed_precision=bf16 -m sallm.main --config_path "$CONFIG"
