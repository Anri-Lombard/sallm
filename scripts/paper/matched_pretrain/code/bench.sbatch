#!/bin/bash
#SBATCH --job-name=mp-bench
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=srvrocgpu009
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/logs/%x-%j.out
# Throughput benchmark: ARCH STACK [MB] [EXTRA...]; GPUs from --gres at submit time.
set -eo pipefail
source /scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/code/hex_env.sh
export HF_HUB_OFFLINE=1
ARCH=$1; STACK=$2; MB=${3:-8}; shift 3 || true
NG=$(nvidia-smi -L | wc -l)
OUT=$R/runs/bench_${ARCH}_${STACK}_g${NG}_mb${MB}_$SLURM_JOB_ID
mkdir -p $OUT
echo "job=$SLURM_JOB_ID node=$(hostname) arch=$ARCH stack=$STACK gpus=$NG mb=$MB extra=$* start=$(date -Is)"
nvidia-smi; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv
cd $R/code && sha256sum -c CODE.sha256
nvidia-smi --query-gpu=timestamp,index,uuid,power.draw,utilization.gpu,memory.used,clocks.sm,temperature.gpu --format=csv,noheader -lms 1000 > $OUT/power.csv &
SMI=$!
T0=$(date +%s)
$PY/torchrun --standalone --nproc_per_node=$NG pretrain.py --config ../configs/${ARCH}_cosine.json --data $R/data/bench3 \
  --out $OUT --stack $STACK --micro-batch $MB --max-steps ${STEPS:-300} --max-minutes ${MINUTES:-15} --eval-at ${EVAL_AT:-60} --bench-save "$@" \
  2>&1 | grep -v -E "Warning|warn\(" || echo "TRAIN FAILED"
kill $SMI || true
echo "torchrun_wall_s=$(( $(date +%s) - T0 )) end=$(date -Is)"
du -sh $OUT/weights $OUT/resume_latest 2>/dev/null || true
