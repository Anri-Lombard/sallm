#!/bin/bash
#SBATCH --job-name=fft-effbench
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926/logs/%x-%j.out
# Efficiency benchmark: the four architectures one after another on the SAME L40S (runners/efficiency_bench.py).
# Run it only when no rollout lane is using that node's GPU (see RUNBOOK.md "Efficiency benchmark").
# env: [DTYPES="bfloat16 float32"]  [ARCHS="mzansilm mamba2 xlstm gdn"]
set -euo pipefail
R=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926
S=/scratch/lmbanr001/masters/sallm
DTYPES=${DTYPES:-"bfloat16 float32"}
ARCHS=${ARCHS:-"mzansilm mamba2 xlstm gdn"}
(cd "$R/code" && sha256sum -c CODE.sha256 --quiet)
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used --format=csv,noheader
echo "compute apps on this GPU at start:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q L40S || { echo "not an L40S" >&2; exit 3; }
mkdir -p "$R/efficiency_bench"
for dtype in $DTYPES; do for arch in $ARCHS; do
  out=$R/efficiency_bench/${arch}_${dtype}.json
  [[ -e "$out" ]] && { echo "skip $arch $dtype (exists)"; continue; }
  py=/home/lmbanr001/masters/sallm/.venv/bin/python; path=$R/code/sallm/src/main
  [[ $arch == xlstm ]] && py=$S/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
  [[ $arch == mamba2 ]] && { path=$R/code/fft_rollout/hooks:$path; export SALLM_FLA_MAMBA2=1; } || unset SALLM_FLA_MAMBA2
  # same environment as rollout.py's Rollout.env (eval): kernels, offline HF, per-architecture Triton cache
  PYTHONPATH=$path PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 \
  FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda SALLM_SKIP_MAMBA_KERNEL_CHECK=1 TOKENIZERS_PARALLELISM=false \
  HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True \
  TRITON_CACHE_DIR=$R/triton_cache/$arch \
  "$py" "$R/code/fft_rollout/runners/efficiency_bench.py" --arch "$arch" --checkpoint "$R/runs/$arch/base" --dtype "$dtype" --output "$out" \
    || echo "FAILED $arch $dtype" >&2
done; done
echo "end=$(date -Is)"
ls -l "$R/efficiency_bench"
