#!/bin/bash
#SBATCH --job-name=mp-convert-mpv1
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --exclude=srvrocgpu009
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/logs/%x-%j.out
set -eo pipefail
source /scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/code/hex_env.sh
cd $R/code && $PY/python convert_mpv1.py /scratch/lmbanr001/masters/sallm/data/matched_pretrain_v1 $R/data/mpv1
echo "end=$(date -Is)"
