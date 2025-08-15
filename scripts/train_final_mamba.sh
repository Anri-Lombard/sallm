#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=1
#SBATCH --job-name="sallm-mamba"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

# TODO make one script instead of seperate final scripts per architecture

set -euo pipefail

export MKL_INTERFACE_LAYER=LP64,INTEL64

CONFIG="base/mamba_125m.yaml"

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate sallm-ner
set -u

export MAMBA_SCAN_IMPL="cuda"
export TORCHDYNAMO_DISABLE="1"

echo "Launching training with $CONFIG"
accelerate launch --mixed_precision=bf16 -m sallm.main --config-name "$CONFIG"
