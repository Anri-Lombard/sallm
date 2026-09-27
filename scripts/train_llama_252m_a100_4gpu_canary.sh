#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere:3
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=llama252-canary
#SBATCH --mail-type=FAIL,END

set -euo pipefail

export WANDB_MODE=offline

exec bash "$HOME/masters/sallm/ops/slurm/train_final_model.sh" \
  base/llama_252m \
  base.training.max_steps=300 \
  base.training.output_dir='${oc.env:SCRATCH}/masters/sallm/checkpoints/sallm-llama-252m-canary' \
  base.training.logging_dir='${oc.env:SCRATCH}/masters/sallm/logs/sallm-llama-252m-canary' \
  base.training.save_steps=250 \
  base.training.eval_strategy=no
