#!/bin/bash
#SBATCH --job-name=mp-pack5
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=srvrocgpu009
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/logs/%x-%j.out
# CPU packing of the LR-probe data (5 shards, same seed as bench3/real run). The GPU is idle; HEX head node must not do this.
set -eo pipefail
source /scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/code/hex_env.sh
cd $R/code
[ -f $R/data/probe5/train.bin ] || $PY/python pack.py --repo uctnlp/mzansi-text-tokenized --rev 7dab1ebf1e1f00ea967a3845aea0acde6cfc0792 --files 5 --out $R/data/probe5
echo "end=$(date -Is)"
