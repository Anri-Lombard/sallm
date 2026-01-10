#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

CONFIG_NAME="$1"
if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: sbatch $0 <config_name_without_yaml>"; exit 1
fi

if [[ "$CONFIG_NAME" != */* ]]; then
    CONFIG_NAME="eval/$CONFIG_NAME"
fi

CFG_NAME="${CONFIG_NAME##*/}"
JOB_NAME="eval-${CFG_NAME#mamba_}"
JOB_NAME="${JOB_NAME#llama_}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME"
  mkdir -p logs
  exec > >(tee -a "logs/${JOB_NAME}-${SLURM_JOB_ID}.out") 2>&1
fi

export HYDRA_FULL_ERROR=1

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"

module load python/miniconda3-py3.12
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u
conda activate sallm-uv
set -u

export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen

python -m sallm.main --config-name "$CONFIG_NAME"
