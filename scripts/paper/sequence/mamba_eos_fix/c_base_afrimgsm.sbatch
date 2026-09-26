#!/bin/bash
#SBATCH --job-name=eosfix-base-mgsm
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/logs/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/logs/%x-%j.err
set -euo pipefail
source /scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/sb/common.sh
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
source=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
export PYTHONPATH="$source/src/main:$bundle"
out=$R/base_afrimgsm/prompt/7.json
mkdir -p "$(dirname $out)"
"$PY" "$bundle/run_lm_eval_unit.py" --mode official --index 7 \
  --inventory $R/protocols/u007_inventory --bindings $R/protocols/u007_bindings/BINDINGS.json --source "$source" \
  --release /scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/base-mamba-xlstm-20260924/release/READY_FOR_OFFICIAL.json \
  --output "$out"
date -Is
