#!/bin/bash
#SBATCH --job-name=fm-validate-v1
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --gres=gpu:ampere:1
#SBATCH --array=0-24%3
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%A_%a.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%A_%a.err
set -euo pipefail
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
result=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1
source=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
default_python=/home/lmbanr001/masters/sallm/.venv/bin/python
xlstm_python=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
index="${SLURM_ARRAY_TASK_ID:?}"
architecture="$(jq -r ".[${index}].architecture" "$bundle/frozen-inventory/validation_arms.json")"
python="$default_python"
[[ "$architecture" != xlstm ]] || python="$xlstm_python"
[[ -x "$python" ]]
(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
[[ -s "$result/bindings/BINDINGS_RESOLVED.json" ]]
export HF_HOME=/scratch/lmbanr001/hf-cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export FLA_DISABLE_BACKEND_DISPATCH=1
export MAMBA_SCAN_IMPL=cuda
export PYTHONPATH="$source/src/main:$bundle"
mkdir -p "$result/validation"
"$python" "$bundle/run_lm_eval_unit.py" \
  --mode validation --index "$index" \
  --inventory "$bundle/frozen-inventory" \
  --bindings "$result/bindings/BINDINGS.json" \
  --source "$source" \
  --output "$result/validation/${index}.json"
