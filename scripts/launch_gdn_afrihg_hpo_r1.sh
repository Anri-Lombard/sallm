#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5%2
#SBATCH --job-name=gdn-afrihg-hpo
#SBATCH --mail-type=FAIL,END

set -euo pipefail

export SCRATCH="${SCRATCH:-/scratch/${USER}}"

trial="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
learning_rates=(3.0e-05 8.0e-05 2.0e-04 3.0e-05 8.0e-05 2.0e-04)
label_smoothing=(0.0 0.0 0.0 0.05 0.05 0.05)

export SALLM_JOB_NAME="gdn-hg-hpo-${trial}"

exec bash "${SALLM_REPO_DIR:-$HOME/masters/sallm}/scripts/launch_finetune.sh" \
  finetune/gdn_afrihg_all_hpo_r1 \
  "training.learning_rate=${learning_rates[$trial]}" \
  "training.label_smoothing_factor=${label_smoothing[$trial]}" \
  "training.output_dir=${SCRATCH}/masters/sallm/checkpoints/gdn_afrihg_hpo_r1/trial_${trial}" \
  "training.logging_dir=${SCRATCH}/masters/sallm/logs/gdn_afrihg_hpo_r1/trial_${trial}" \
  "training.run_name=gdn-125m-afrihg-hpo-r1-trial-${trial}" \
  "wandb.name=gdn-125m-afrihg-hpo-r1-trial-${trial}"
