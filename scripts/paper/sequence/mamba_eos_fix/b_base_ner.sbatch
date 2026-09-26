#!/bin/bash
#SBATCH --job-name=eosfix-base-ner
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
result=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1
source=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
export PYTHONPATH="$source/src/main:$bundle"
protocol=$R/protocols/u005-base-mamba2-ner-eosfix.json
out=$R/base_ner/sequence/5.json
mkdir -p "$(dirname $out)"
"$PY" "$bundle/run_general_sequence_eval_20260914.py" --task ner --phase test --architecture mamba2 \
  --checkpoint "$FIX" --protocol "$protocol" \
  --selection /scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection/SELECTION.json \
  --release "$result/partial-ready-1r1/release/NER_TEST_ACCESS_RELEASED.json" --output "$out"
date -Is
