#!/bin/bash
#SBATCH --job-name=fm-select-v1
#SBATCH --account=nlpgroup
#SBATCH --partition=ada
#SBATCH --qos=nlpgroup
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%j.err
set -euo pipefail
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
result=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1
sequence_selection=/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/SELECTION.json
python=/home/lmbanr001/masters/sallm/.venv/bin/python
(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
"$python" "$bundle/select_bindings.py" \
  --inventory "$bundle/frozen-inventory" \
  --validation-root "$result/validation" \
  --sequence-selection "$sequence_selection" \
  --bindings "$result/bindings/BINDINGS.json" \
  --protocol-template "$bundle/general_sequence_protocol_template.json" \
  --output "$result/selection"
