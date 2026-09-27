#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --qos=l40sfree
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
# Runs several small trainings concurrently on one GPU (one Slurm slot). LABELS="a b c"
set -uo pipefail
root=/scratch/lmbanr001/masters/sallm/results/monomulti_retrain_20260924
pids=()
for label in ${LABELS:?}; do
  LABEL="$label" bash "$root/jobs/train_hex.sbatch" > "$root/logs/$label-${SLURM_JOB_ID}.out" 2>&1 &
  pids+=($!)
  sleep 60
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
nvidia-smi
exit $rc
