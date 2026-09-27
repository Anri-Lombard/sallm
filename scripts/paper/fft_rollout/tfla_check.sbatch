#!/bin/bash
#SBATCH --job-name=fft-tfla-check
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926/speedcheck_20260927/%x-%j.out
# padded/raw TFLA checks on one L40S in the sealed xLSTM runtime (~10 min): kernel level + model-level parity.
set -euo pipefail
R=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926
OUT=$R/speedcheck_20260927
mkdir -p $OUT
export PYTHONPATH=$R/pydeps_tfla PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TRITON_CACHE_DIR=$OUT/triton HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
PY=$R/../standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
$PY $R/code/fft_rollout/tfla_check.py kernel $OUT/kernel_l40s_$SLURM_JOB_ID.json
$PY $R/code/fft_rollout/tfla_check.py parity $R/runs/xlstm/base $OUT/parity_l40s_$SLURM_JOB_ID.json
