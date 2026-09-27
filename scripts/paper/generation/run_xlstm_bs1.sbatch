#!/bin/bash
#SBATCH --job-name=gen-xlstm-bs1
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/generation_protocol_v4_xlstm_bs1_20260925/slurm/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/generation_protocol_v4_xlstm_bs1_20260925/slurm/%x-%j.err
# usage: sbatch run_xlstm_bs1.sbatch <unit index> [<unit index> ...]   (indices into units_xlstm_bs1.json)
set -uo pipefail
here=/scratch/lmbanr001/masters/sallm_snapshots/generation-protocol-v4-xlstm-bs1-20260925
gen=/scratch/lmbanr001/masters/sallm_snapshots/generation-protocol-v3-greedy-20260924
source=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
out=/scratch/lmbanr001/masters/sallm/results/generation_protocol_v4_xlstm_bs1_20260925
python=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
export HF_HOME=/scratch/lmbanr001/hf-cache HF_DATASETS_CACHE=/scratch/lmbanr001/hf-cache/datasets HF_HUB_CACHE=/scratch/lmbanr001/hf-cache/hub
export HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1
export FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda
export SALLM_AFRIHG_CACHE_DIR=$gen/data/afrihg_cache SALLM_T2X_CACHE_DIR=$gen/data/t2x_cache
export PYTHONPATH="$source/src/main:$bundle"
cd "$gen"
nvidia-smi -L
for index in "$@"; do
  unit="$(jq -r ".[$index].unit_id" "$here/units_xlstm_bs1.json")"
  start=$(date +%s)
  echo "START $unit $(date -Is) job=$SLURM_JOB_ID" >> "$out/status.txt"
  if "$python" "$here/run_generation_direct_bs1.py" --spec "$here/units_xlstm_bs1.json" --index "$index" \
      --source "$source" --output "$out/$unit/greedy" --split test --system-prompt drop --decoding greedy \
      > "$out/logs/$unit.log" 2>&1; then
    echo "DONE $unit secs=$(( $(date +%s) - start )) $(date -Is)" >> "$out/status.txt"
  else
    echo "FAIL $unit secs=$(( $(date +%s) - start )) $(date -Is)" >> "$out/status.txt"
  fi
done
