#!/bin/bash
#SBATCH --job-name=mm-rescore
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/monomulti_rescore_20260924/slurm/%x-%A_%a.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/monomulti_rescore_20260924/slurm/%x-%A_%a.err
set -euo pipefail
R=/scratch/lmbanr001/masters/sallm/results/monomulti_rescore_20260924
line=$(awk -F'\t' -v i="$SLURM_ARRAY_TASK_ID" '$1==i' "$JOBS")
[[ -n "$line" ]]
IFS=$'\t' read -r _ UNIT ARCH TASK RUNNER POSLANG CHECKPOINT ADAPTER PROTOCOL OUTPUT <<< "$line"
echo "UNIT $UNIT"
export ARCH TASK RUNNER POSLANG CHECKPOINT ADAPTER PROTOCOL OUTPUT EXPECT_GPU="${EXPECT_GPU:-L40S}"
bash $R/scripts/run_unit.sbatch
