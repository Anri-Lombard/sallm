#!/bin/bash
#SBATCH --job-name=mp-causality
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --exclude=srvrocgpu009
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/logs/%x-%j.out
# No-future-leak check (causality.py): OUT_JSON arch:weights_dir[:kernel] ...
set -eo pipefail
source /scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/code/hex_env.sh
export HF_HUB_OFFLINE=1 MP_VAL_PARQUET=$R/data/mpv1/validation.parquet
cd $R/code && sha256sum -c CODE.sha256
$PY/python causality.py "$@" 2>&1 | grep -v -E "Warning|warn\("
