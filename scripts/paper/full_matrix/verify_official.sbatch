#!/bin/bash
#SBATCH --job-name=fm-verify-v1
#SBATCH --account=nlpgroup
#SBATCH --partition=ada
#SBATCH --qos=nlpgroup
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%j.err
set -euo pipefail
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
result=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1
python=/home/lmbanr001/masters/sallm/.venv/bin/python
(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
"$python" "$bundle/verify_official.py" \
  --inventory "$bundle/frozen-inventory" \
  --release "$result/release/READY_FOR_OFFICIAL.json" \
  --official-root "$result/official" \
  --output "$result/verified/OFFICIAL_VERIFIED.json"
