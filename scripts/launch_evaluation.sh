#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name="sallm-eval"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

set -euo pipefail

TASK="${TASK:?TASK env var not set}"
ARCH="${ARCH:?ARCH env var not set}"
LANG="${LANG:?LANG env var not set}"

CONFIG_NAME="eval/${TASK}"

export HYDRA_FULL_ERROR=1

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"

module load python/miniconda3-py3.12
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Define ADDR2LINE to prevent unbound variable errors during activation
export ADDR2LINE="${ADDR2LINE:-$(command -v addr2line 2>/dev/null || echo addr2line)}"

set +u
conda activate sallm-ner
set -u

python -m sallm.main --config-name "$CONFIG_NAME" architecture="$ARCH" language="$LANG" "$@"
