#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name="sallm-ft"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

# ----------------------------------------------------------------------------
# Launch fine-tuning on a SLURM cluster (accelerate + Hydra)
#
# Purpose
#   Submit a Slurm job that activates the project conda environment and runs
#   the fine-tuning entry point with `accelerate launch`. The script attempts
#   to detect GPU resources exposed by SLURM and configures a few runtime
#   environment variables to reduce fragmentation and improve stability.
#
# Usage
#   sbatch launch_finetune.sh <config_name_without_yaml>
#
# Example
#   sbatch launch_finetune.sh base/mamba_125m
#
# Notes
#   - Requires a conda environment named `sallm-ner` to be available on the
#     compute node. Adjust the activation command if your environment differs.
#   - The script sets `PYTORCH_CUDA_ALLOC_CONF` to reduce CUDA fragmentation.
#   - Environment initialization (module load / conda) is tuned for the
#     institutional cluster used by the project; adapt as needed for other
#     clusters or local runs.
# ----------------------------------------------------------------------------

CFG="$1"; [[ -z "$CFG" ]] && { echo "Usage: sbatch $0 <config_name_without_yaml>"; exit 1; }

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"
export TOKENIZERS_PARALLELISM="true"
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export TORCH_DISTRIBUTED_TIMEOUT=7200
export HYDRA_FULL_ERROR=1

echo "--- Checking GPU availability ---"
nvidia-smi
echo "-------------------------------"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sallm-ner

# PyTorch CUDA allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# Debug info for distributed runs
echo "LOCAL_RANK=${LOCAL_RANK:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Compute number of processes from SLURM or nvidia-smi
NUM_PROCS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
if [[ -z "$NUM_PROCS" || "$NUM_PROCS" -le 0 ]]; then
	if command -v nvidia-smi >/dev/null 2>&1; then
		NUM_PROCS=$(nvidia-smi -L | wc -l | tr -d ' ')
	else
		NUM_PROCS=1
	fi
fi

# Explicit accelerate options
accelerate launch \
	--num_processes "$NUM_PROCS" \
	--num_machines 1 \
	--mixed_precision bf16 \
	--dynamo_backend no \
	-m sallm.main --config-name "$CFG"
