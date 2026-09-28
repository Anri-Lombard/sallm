#!/bin/bash
#SBATCH --job-name=mp-full
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --exclude=srvrocgpu009
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/logs/%x-%j.out
# Full matched run: ARCH LR MB [extra]. WSD, 1 epoch of matched_pretrain_v1 (E = 2,839,511,040), fast stack,
# global batch 192x2048 (GPU count from --gres; only micro-batch/accumulation change). Auto-resumes from resume_latest.
set -eo pipefail
source /scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/code/hex_env.sh
export HF_HUB_OFFLINE=1 MP_VAL_PARQUET=$R/data/mpv1/validation.parquet
ARCH=$1; LR=$2; MB=$3; shift 3
NG=$(nvidia-smi -L | wc -l)
OUT=$R/runs/full_${ARCH}_wsd_lr${LR}
mkdir -p $OUT
echo "job=$SLURM_JOB_ID node=$(hostname) arch=$ARCH lr=$LR gpus=$NG mb=$MB extra=$* start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv
cd $R/code && sha256sum -c CODE.sha256
RES=(); [ -f $OUT/resume_latest/state.pt ] && RES=(--resume $OUT/resume_latest) && echo "resuming from $OUT/resume_latest"
nvidia-smi --query-gpu=timestamp,index,uuid,power.draw,utilization.gpu,memory.used,clocks.sm,temperature.gpu --format=csv,noheader -lms 1000 >> $OUT/power_$SLURM_JOB_ID.csv &
SMI=$!
$PY/torchrun --standalone --nproc_per_node=$NG pretrain.py --config ../configs/${ARCH}_wsd.json --data $R/data/mpv1 \
  --out $OUT --stack fast --micro-batch $MB --peak-lr $LR --eval-doc-start eos "${RES[@]}" "$@" 2>&1 \
  | grep -v -E "Warning|warn\(|torch._dynamo|To log all|To diagnose|last reason|function: " || echo "TRAIN FAILED"
kill $SMI || true
echo "end=$(date -Is)"
