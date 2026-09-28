#!/bin/bash
#SBATCH --job-name=mp-lrprobe
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:40:00
#SBATCH --nodes=1
#SBATCH --exclude=srvrocgpu009
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/logs/%x-%j.out
# LR probe (pre-registered in configs/README.md): ARCH LR MB [extra]
set -eo pipefail
source /scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/code/hex_env.sh
export HF_HUB_OFFLINE=1 MP_VAL_PARQUET=$R/data/mpv1/validation.parquet
ARCH=$1; LR=$2; MB=$3; shift 3
NG=$(nvidia-smi -L | wc -l)
OUT=$R/runs/lrprobe_v1_${ARCH}_lr${LR}
mkdir -p $OUT
echo "job=$SLURM_JOB_ID node=$(hostname) arch=$ARCH lr=$LR gpus=$NG mb=$MB extra=$* start=$(date -Is)"
cd $R/code && sha256sum -c CODE.sha256
nvidia-smi --query-gpu=timestamp,index,uuid,power.draw,utilization.gpu,memory.used,clocks.sm,temperature.gpu --format=csv,noheader -lms 1000 > $OUT/power.csv &
SMI=$!
$PY/torchrun --standalone --nproc_per_node=$NG pretrain.py --config ../configs/${ARCH}_wsd.json --data $R/data/mpv1 \
  --out $OUT --stack fast --micro-batch $MB --total-tokens 300e6 --warmup-tokens 113639424 --eval-doc-start eos --peak-lr $LR \
  --no-step0-eval --weights-final-only --resume-every 0 "$@" 2>&1 | grep -v -E "Warning|warn\(" || echo "TRAIN FAILED"
kill $SMI || true
echo "end=$(date -Is)"
