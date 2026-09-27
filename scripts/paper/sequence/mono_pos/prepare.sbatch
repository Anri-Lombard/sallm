#!/bin/bash
#SBATCH --job-name=gdn-mono-pos-prepare
#SBATCH --account=maths
#SBATCH --partition=ada
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%j.err
set -euo pipefail
snapshot=/scratch/lmbanr001/masters/sallm_snapshots/gdn-mono-pos-official-20260920-v2
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
source=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
(cd "$snapshot" && sha256sum -c ARTIFACTS.sha256 --quiet)
(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
export PYTHONPATH="$source/src/main:$bundle"
export PYTHONDONTWRITEBYTECODE=1
/home/lmbanr001/masters/sallm/.venv/bin/python "$snapshot/run_gdn_mono_pos.py" --language xho --self-check
/home/lmbanr001/masters/sallm/.venv/bin/python "$snapshot/prepare.py"
