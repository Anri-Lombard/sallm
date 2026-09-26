#!/bin/bash
#SBATCH --job-name=fm-official-v1
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --gres=gpu:ampere:1
#SBATCH --array=0-96%3
#SBATCH --time=24:00:00
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
sequence_selection=/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/SELECTION.json
default_python=/home/lmbanr001/masters/sallm/.venv/bin/python
xlstm_python=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
index="${SLURM_ARRAY_TASK_ID:?}"
unit_id="$(jq -r ".[${index}].unit_id" "$bundle/frozen-inventory/official_units.json")"
architecture="$(jq -r ".[${index}].architecture" "$bundle/frozen-inventory/official_units.json")"
group="$(jq -r ".[${index}].task_group" "$bundle/frozen-inventory/official_units.json")"
python="$default_python"
[[ "$architecture" != xlstm ]] || python="$xlstm_python"
[[ -x "$python" ]]
(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
release="$result/release/READY_FOR_OFFICIAL.json"
[[ -s "$release" ]]
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
if [[ "$group" == ner || "$group" == pos ]]; then
  mkdir -p "$result/official/sequence"
  protocol="$result/selection/sequence_protocols/${unit_id}.json"
  checkpoint="$(jq -r ".models.${architecture}.base_path" "$protocol")"
  adapter="$(jq -r ".models.${architecture}.adapter_path // empty" "$protocol")"
  args=(
    --task "$group" --phase test --architecture "$architecture"
    --checkpoint "$checkpoint" --protocol "$protocol"
    --selection "$sequence_selection"
    --release "$result/release/$(printf '%s' "$group" | tr '[:lower:]' '[:upper:]')_TEST_ACCESS_RELEASED.json"
    --output "$result/official/sequence/${index}.json"
  )
  [[ -z "$adapter" ]] || args+=(--adapter "$adapter")
  "$python" "$bundle/run_general_sequence_eval_20260914.py" "${args[@]}"
elif [[ "$group" == generation ]]; then
  mkdir -p "$result/official/generation"
  "$python" "$bundle/run_generation_unit.py" \
    --index "$index" --inventory "$bundle/frozen-inventory" \
    --bindings "$result/bindings/BINDINGS.json" --source "$source" \
    --release "$release" --output "$result/official/generation/${index}"
else
  mkdir -p "$result/official/prompt"
  "$python" "$bundle/run_lm_eval_unit.py" \
    --mode official --index "$index" --inventory "$bundle/frozen-inventory" \
    --bindings "$result/bindings/BINDINGS.json" --source "$source" \
    --release "$release" --output "$result/official/prompt/${index}.json"
fi
