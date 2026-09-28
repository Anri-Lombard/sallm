#!/bin/bash
#SBATCH --job-name=mp-resumetest
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --exclude=srvrocgpu009
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/logs/%x-%j.out
# GDN fast, 2 GPUs: train -> SIGKILL after step 90 -> resume from resume_latest (step 80) -> WSD stable ckpt at decay
# start (step 122) -> decay branch from it to step 153. Budget 60M tokens (153 steps), warmup 20, resumable every 40.
set -eo pipefail
source /scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/code/hex_env.sh
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled
OUT=$R/runs/resumetest_gdn_$SLURM_JOB_ID
cd $R/code && sha256sum -c CODE.sha256
COMMON=(--config ../configs/gdn_wsd.json --data $R/data/bench3 --stack fast --micro-batch 24 --total-tokens 60e6
        --warmup-tokens 7864320 --resume-every 40 --no-step0-eval --weights-final-only)
run() { $PY/torchrun --standalone --nproc_per_node=2 pretrain.py "${COMMON[@]}" "$@" 2>&1 | grep -v -E "Warning|warn\(" ; }
echo "== phase 1 $(date -Is)"
setsid $PY/torchrun --standalone --nproc_per_node=2 pretrain.py "${COMMON[@]}" --out $OUT/main --eval-at 99999 > $OUT.phase1.log 2>&1 &
P=$!
until [ "$(wc -l < $OUT/main/train_log.jsonl 2>/dev/null || echo 0)" -ge 90 ]; do sleep 2; done
kill -9 -- -$P || true; sleep 5; pkill -9 -f "out $OUT/main" || true; sleep 5
cp $OUT/main/train_log.jsonl $OUT/phase1_train_log.jsonl
python3 -c "import json;print('phase1 killed at step', json.loads(open('$OUT/phase1_train_log.jsonl').readlines()[-1])['step'])"
python3 -c "import torch" 2>/dev/null || true
ls $OUT/main; echo "resume_latest step: $($PY/python -c "import torch;print(torch.load('$OUT/main/resume_latest/state.pt',map_location='cpu',weights_only=True)['step'])")"
echo "== phase 2 (resume -> stable ckpt) $(date -Is)"
run --out $OUT/main --resume $OUT/main/resume_latest --stop-before-decay --eval-at 99999
echo "== phase 3 (decay branch) $(date -Is)"
run --out $OUT/branch --resume $OUT/main/stable_step000122 --eval-at 153
$PY/python check_resume.py $OUT
echo "end=$(date -Is)"
