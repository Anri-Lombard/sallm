#!/bin/bash
#SBATCH --account=nlpgroup80
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup80
#SBATCH --gres=gpu:ampere80:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/home/lmbanr001/masters/sallm
#SBATCH --job-name=hpo-gdn-general-a0-resume-corrected
#SBATCH --output=/scratch/lmbanr001/masters/sallm/logs/jobs/hpo-gdn-general-a0-resume-corrected-%j.out
#SBATCH --mail-type=FAIL,END

set -euo pipefail

snapshot="$HOME/masters/sallm_snapshots/uniform-adapter-hpo-general-a0-resume-correction-20260828-45fec06b"
runtime="$HOME/masters/sallm"
output="/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/general/stage_a/a0/seed_42"
checkpoint="$output/checkpoint-10912"

[[ "$(sha256sum "$snapshot/scripts/run_pure_gdn_validation_trial.sh" | cut -d' ' -f1)" == "45fec06bf2ea71d4f69a43a686c88b9b8d895750d85b797ed104ac59271cceab" ]]
[[ ! -e "$output/final_adapter/adapter_model.safetensors" ]]

export SALLM_REPO_DIR="$snapshot"
export SALLM_RUNTIME_REPO="$runtime"
export SALLM_HPO_REGISTRY="$snapshot/src/conf/hpo/pure_gdn_enhanced_v1.json"
export SALLM_HPO_RESUME_FROM_CHECKPOINT="$checkpoint"

exec bash "$snapshot/scripts/run_validation_hpo_trial.sh" pure_gdn general stage_a a0 42
