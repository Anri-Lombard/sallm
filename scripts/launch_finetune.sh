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

CFG="$1"; [[ -z "$CFG" ]] && { echo "Usage: sbatch $0 <config_name_without_yaml>"; exit 1; }

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"
# export TOKENIZERS_PARALLELISM="false"
export TOKENIZERS_PARALLELISM="true"
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export TORCH_DISTRIBUTED_TIMEOUT=7200
export HYDRA_FULL_ERROR=1
# export NCCL_BLOCKING_WAIT=1

echo "--- Checking GPU availability ---"
nvidia-smi
echo "-------------------------------"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sallm-ner

# Set PyTorch CUDA allocation config to reduce fragmentation (optional)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# Debug info for distributed runs
echo "LOCAL_RANK=${LOCAL_RANK:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Determine number of processes to launch based on available GPUs
NUM_PROCS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
if [[ -z "$NUM_PROCS" || "$NUM_PROCS" -le 0 ]]; then
	if command -v nvidia-smi >/dev/null 2>&1; then
		NUM_PROCS=$(nvidia-smi -L | wc -l | tr -d ' ')
	else
		NUM_PROCS=1
	fi
fi

# pass explicit accelerate options to avoid its default-warning messages
accelerate launch \
	--num_processes "$NUM_PROCS" \
	--num_machines 1 \
	--mixed_precision bf16 \
	--dynamo_backend no \
	-m sallm.main --config-name "$CFG"
