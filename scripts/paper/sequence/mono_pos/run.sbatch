#!/bin/bash
#SBATCH --job-name=gdn-mono-pos-official
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=27-29%2
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%A_%a.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/slurm/%x-%A_%a.err
set -euo pipefail
snapshot=/scratch/lmbanr001/masters/sallm_snapshots/gdn-mono-pos-official-20260920-v2
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
source=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
root=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1
prepared="$root/gdn-mono-pos-official-20260920-v2"
selection=/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection/SELECTION.json
index="${SLURM_ARRAY_TASK_ID:?}"
case "$index" in
  27) language=tsn ;;
  28) language=xho ;;
  29) language=zul ;;
  *) exit 64 ;;
esac
(cd "$snapshot" && sha256sum -c ARTIFACTS.sha256 --quiet)
(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
jq -e --argjson index "$index" '.status == "PREPARED" and .test_accessed == false and (.indices | index($index) != null)' "$prepared/PREPARED.json" >/dev/null
protocol="$prepared/protocols/${index}.json"
release="$prepared/release/POS_TEST_ACCESS_RELEASED.json"
output="$root/official/sequence/${index}.json"
[[ -s "$protocol" && -s "$release" && ! -e "$output" ]]
checkpoint="$(jq -r '.models.gdn.base_path' "$protocol")"
adapter="$(jq -r '.models.gdn.adapter_path' "$protocol")"
export HF_HOME=/scratch/lmbanr001/hf-cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export PYTHONDONTWRITEBYTECODE=1
export FLA_DISABLE_BACKEND_DISPATCH=1
export PYTHONPATH="$source/src/main:$bundle"
/home/lmbanr001/masters/sallm/.venv/bin/python "$snapshot/run_gdn_mono_pos.py" \
  --task pos --phase test --architecture gdn --language "$language" \
  --checkpoint "$checkpoint" --adapter "$adapter" --protocol "$protocol" \
  --selection "$selection" --release "$release" --output "$output"
