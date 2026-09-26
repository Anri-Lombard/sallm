#!/bin/bash
#SBATCH --job-name=fm-release-v1
#SBATCH --account=nlpgroup
#SBATCH --partition=ada
#SBATCH --qos=nlpgroup
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%j.err
set -euo pipefail
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
result=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1
sequence_selection=/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/SELECTION.json
python=/home/lmbanr001/masters/sallm/.venv/bin/python
(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
"$python" "$bundle/release_official.py" \
  --inventory "$bundle/frozen-inventory" \
  --bindings "$result/bindings/BINDINGS.json" \
  --selection-root "$result/selection" \
  --sequence-selection "$sequence_selection" \
  --output "$result/release"
