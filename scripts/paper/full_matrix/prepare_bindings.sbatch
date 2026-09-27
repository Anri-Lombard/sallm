#!/bin/bash
#SBATCH --job-name=fm-bindings-v1
#SBATCH --account=nlpgroup
#SBATCH --partition=ada
#SBATCH --qos=nlpgroup
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%j.err
set -euo pipefail
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
result=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1
source=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
python=/home/lmbanr001/masters/sallm/.venv/bin/python
mkdir -p "$result/slurm"
(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
(cd "$source" && sha256sum -c SNAPSHOT_MANIFEST.sha256 --quiet)
export HF_HOME=/scratch/lmbanr001/hf-cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$source/src/main:$bundle"
"$python" "$bundle/preflight_protocol.py" \
  --inventory "$bundle/frozen-inventory" \
  --source "$source" \
  --output "$result/PROTOCOL_PREFLIGHT_PASSED.json"
"$python" "$bundle/prepare_bindings.py" \
  --inventory "$bundle/frozen-inventory" \
  --protocol-template "$bundle/general_sequence_protocol_template.json" \
  --output "$result/bindings"
