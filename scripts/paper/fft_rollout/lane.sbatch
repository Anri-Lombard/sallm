#!/bin/bash
#SBATCH --job-name=fft-lane
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926/logs/%x-%j.out
# One worker lane (one L40S) of an architecture's rollout; launch.sh submits MAX_LANES of them. See RUNBOOK.md.
# env: OUT=<run dir>  LANE=<index>  [LANE_HOURS=47.5]
set -euo pipefail
: "${OUT:?OUT is required}" "${LANE:?LANE is required}"
R=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926
(cd "$R/code" && sha256sum -c CODE.sha256 --quiet)
echo "lane=$LANE job=$SLURM_JOB_ID node=$(hostname) out=$OUT start=$(date -Is)"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used --format=csv,noheader
df -BG /dev/shm | tail -1
/scratch/slurm/bin/purequota 2>/dev/null | grep -E '^/scratch' || true
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/home/lmbanr001/masters/sallm/.venv/bin/python "$R/code/fft_rollout/rollout.py" lane "$OUT"
echo "lane=$LANE end=$(date -Is)"
