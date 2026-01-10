#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name=hpo-ner_all-resume
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

set -euo pipefail

SWEEP_PATH="${1:-anri-lombard/sallm-ft/5k15lf2s}"
COUNT="${2:-25}"

mkdir -p logs
exec > >(tee -a "logs/hpo-ner_all-resume-${SLURM_JOB_ID}.out") 2>&1

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"
export TOKENIZERS_PARALLELISM="true"
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_DISTRIBUTED_TIMEOUT=7200
export HYDRA_FULL_ERROR=1

module load python/miniconda3-py3.12
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sallm-uv
set -u

export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

NUM_GPUS=2
AGENTS_TO_RUN=$NUM_GPUS
BASE_PER_AGENT=$(( COUNT / AGENTS_TO_RUN ))
REMAINDER=$(( COUNT % AGENTS_TO_RUN ))

echo "Resuming sweep $SWEEP_PATH with $COUNT remaining runs across $NUM_GPUS GPUs"

PIDS=()
for IDX in $(seq 0 $((AGENTS_TO_RUN - 1))); do
  PER_AGENT=$BASE_PER_AGENT
  if [[ $IDX -lt $REMAINDER ]]; then
    PER_AGENT=$(( PER_AGENT + 1 ))
  fi
  export CUDA_VISIBLE_DEVICES="$IDX"
  echo "GPU $IDX -> wandb agent --count $PER_AGENT $SWEEP_PATH"
  wandb agent --count "$PER_AGENT" "$SWEEP_PATH" &
  PIDS+=("$!")
done

for P in "${PIDS[@]}"; do
  wait "$P" || true
done
