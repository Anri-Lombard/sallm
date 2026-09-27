#!/bin/bash
#SBATCH --job-name=gen-protocol-v3-greedy
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere:1
#SBATCH --array=0-11
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/generation_protocol_v3_greedy_20260924/slurm/%x-%A_%a.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/generation_protocol_v3_greedy_20260924/slurm/%x-%A_%a.err
set -euo pipefail
here=/scratch/lmbanr001/masters/sallm_snapshots/generation-protocol-v3-greedy-20260924
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
source=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
root=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/generation-remaining-20260920-v1
out=/scratch/lmbanr001/masters/sallm/results/generation_protocol_v3_greedy_20260924
IFS=$'\t' read -r kind index < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$here/joblist_v3.tsv")
if [[ "$kind" == inv ]]; then
  unit="$(jq -r --argjson i "$index" '.[$i].unit_id' "$root/subset/official_units.json")"
  architecture="$(jq -r --argjson i "$index" '.[$i].architecture' "$root/subset/official_units.json")"
else
  unit="$(jq -r --argjson i "$index" '.[$i].unit_id' "$here/extra_units.json")"
  architecture="$(jq -r --argjson i "$index" '.[$i].architecture' "$here/extra_units.json")"
fi
python=/home/lmbanr001/masters/sallm/.venv/bin/python
[[ "$architecture" != xlstm ]] || python=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
export HF_HOME=/scratch/lmbanr001/hf-cache HF_DATASETS_CACHE=/scratch/lmbanr001/hf-cache/datasets HF_HUB_CACHE=/scratch/lmbanr001/hf-cache/hub
export HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1
export FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda
export PYTHONPATH="$source/src/main:$bundle"
cd "$here"
if [[ "$kind" == inv ]]; then
  "$python" "$here/run_generation_protocol.py" --index "$index" --inventory "$root/subset" \
    --bindings "$root/selection/BINDINGS.json" --source "$source" \
    --release "$root/release/READY_FOR_OFFICIAL.json" \
    --output "$out/$unit/greedy" --split test --system-prompt drop --decoding greedy
else
  "$python" "$here/run_generation_direct.py" --spec "$here/extra_units.json" --index "$index" \
    --source "$source" --output "$out/$unit/greedy" --split test --system-prompt drop --decoding greedy
fi
