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

set -euo pipefail

TASK="${TASK:?TASK env var not set}"
ARCH="${ARCH:?ARCH env var not set}"
LANG="${LANG:?LANG env var not set}"

CFG="finetune/${TASK}"

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"
# export TOKENIZERS_PARALLELISM="false"
export TOKENIZERS_PARALLELISM="true"
export HF_HOME="$SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export TORCH_DISTRIBUTED_TIMEOUT=7200
# export NCCL_BLOCKING_WAIT=1

echo "--- Checking GPU availability ---"
nvidia-smi
echo "-------------------------------"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sallm-ner

accelerate launch --num_processes 2 -m sallm.main --config-name "$CFG" architecture="$ARCH" language="$LANG" "$@"
