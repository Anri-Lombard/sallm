#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --qos=l40sfree
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=rsel-smokefeb
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925/slurm/%x-%j.out
# stage_val smoke for a Feb-era Mono Intent label (upstream dev val source), timed, then a resume rerun.
set -uo pipefail
root=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925
export RESELECT_TRAIN_ROOT=$root/tmp/smoke/train RESELECT_VAL_ROOT=$root/tmp/smoke/val
nvidia-smi --query-gpu=name --format=csv,noheader
for i in 1 2; do t0=$(date +%s); bash $root/jobs/stage_val.sh mamba2_intent_mono_sot; echo "=== END rc=$? $(( $(date +%s) - t0 ))s"; done
echo SMOKE_FEB_DONE
