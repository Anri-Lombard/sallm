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

# ----------------------------------------------------------------------------
# Launch training for Mamba models (final evaluation/experiments)
#
# Purpose
#   Execute a training job using a Mamba model configuration. The script sets
#   up environment variables and `accelerate launch` flags optimized for the
#   Mamba model family.
#
# Notes
#   - The script uses bf16 mixed precision and sets MAMBA_SCAN_IMPL.
#   - Ensure the `sallm-ner` conda environment exists on the compute node.
# ----------------------------------------------------------------------------

# Consider merging final scripts into a single configurable script

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
