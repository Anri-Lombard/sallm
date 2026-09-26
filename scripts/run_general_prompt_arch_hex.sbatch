#!/usr/bin/env bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --nodelist=srvrocgpu010
#SBATCH --gres=gpu:ampere:1
#SBATCH --array=0-7%2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,END

set -euo pipefail
/scratch/slurm/bin/purequota

phase="${1:?expected validation or test}"
[[ "$phase" == validation || "$phase" == test ]] || exit 2

snapshot=/scratch/lmbanr001/masters/sallm_snapshots/general-prompt-correction-20260914-v1
asset_root=/scratch/lmbanr001/masters/sallm/assets/general_prompt_correction_20260914_v1
result_root=/scratch/lmbanr001/masters/sallm/results/retained_standardized_20260913_v1/general_prompt_corrected_20260914_v1
selection_root="$result_root/selection"
python_bin=/home/lmbanr001/masters/sallm/.venv/bin/python

architectures=(mzansilm mzansilm mamba2 mamba2 xlstm xlstm gdn gdn)
units=(lm_eval intent lm_eval intent lm_eval intent lm_eval intent)
architecture="${architectures[$SLURM_ARRAY_TASK_ID]}"
unit="${units[$SLURM_ARRAY_TASK_ID]}"
checkpoint="$asset_root/$architecture/base"
adapter="$asset_root/$architecture/adapter"

[[ -x "$python_bin" && -d "$checkpoint" && -d "$adapter" ]]
(cd "$snapshot" && sha256sum --check --strict SNAPSHOT_EXECUTION_MANIFEST.json.sha256)
"$python_bin" "$snapshot/scripts/create_execution_manifest.py" \
  --verify "$snapshot/SNAPSHOT_EXECUTION_MANIFEST.json" \
  --expected-repo-root "$snapshot"
sha256sum --check --strict "$snapshot/.audit/general_prompt_correction_model_files_20260914.sha256"

if [[ "$phase" == test ]]; then
  [[ -f "$selection_root/READY_FOR_TEST.json" ]]
  selection_sha="$(sha256sum "$selection_root/SELECTION.json" | awk '{print $1}')"
  [[ "$selection_sha" == "$(jq -r .selection_sha256 "$selection_root/READY_FOR_TEST.json")" ]]
fi

output="$result_root/raw/$phase/$unit/$architecture.json"
[[ ! -e "$output" && ! -e "$output.tmp" ]]
mkdir -p "$(dirname "$output")"

export PYTHONPATH="$snapshot/src/main:$snapshot/scripts"
export HF_HOME=/scratch/lmbanr001/hf
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export TMPDIR="${SLURM_TMPDIR:-/scratch/lmbanr001/tmp/general-prompt-$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

extra_model_args=()
if [[ "$architecture" == xlstm ]]; then
  extra_model_args=(--merge-lora --tie-word-embeddings false)
fi

printf '%s\tGENERAL_PROMPT_START\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$phase" "$architecture" "$unit"
if [[ "$unit" == lm_eval ]]; then
  command=(
    "$python_bin" "$snapshot/scripts/run_general_prompt_lm_eval.py"
    --phase "$phase"
    --architecture "$architecture"
    --checkpoint "$checkpoint"
    --adapter "$adapter"
    --output "$output"
    "${extra_model_args[@]}"
  )
  if [[ "$phase" == test ]]; then
    command+=(--selection "$selection_root/SELECTION.json")
  fi
  "${command[@]}"
else
  command=(
    "$python_bin" "$snapshot/scripts/run_injongointent_mean_eval.py"
    --config "$snapshot/src/conf/finetune/llama_injongointent_all.yaml"
    --checkpoint "$checkpoint"
    --adapter "$adapter"
    --architecture "$architecture"
    --output "$output"
    --split "$phase"
    --max-length 1024
  )
  if [[ "$phase" == test ]]; then
    command+=(--template-selection "$selection_root/intent_selection.json")
  fi
  "${command[@]}"
fi
printf '%s\tGENERAL_PROMPT_COMPLETE\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$phase" "$architecture" "$unit" "$(sha256sum "$output" | awk '{print $1}')"
