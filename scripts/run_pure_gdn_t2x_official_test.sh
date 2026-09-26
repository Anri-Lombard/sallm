#!/bin/bash
set -euo pipefail

/scratch/slurm/bin/purequota

snapshot="${SALLM_OFFICIAL_SNAPSHOT:?missing immutable snapshot}"
source_manifest="${SALLM_OFFICIAL_SOURCE_MANIFEST:?missing source manifest}"
base_manifest="${SALLM_OFFICIAL_BASE_MANIFEST:?missing base manifest}"
adapter_manifest="${SALLM_OFFICIAL_ADAPTER_MANIFEST:?missing adapter manifest}"
base="${SALLM_OFFICIAL_BASE:?missing base checkpoint}"
adapter="${SALLM_OFFICIAL_ADAPTER:?missing frozen adapter}"
output="${SALLM_OFFICIAL_OUTPUT:?missing official output root}"
protocol_root="${SALLM_OFFICIAL_PROTOCOL_ROOT:?missing protocol root}"
python_bin="${SALLM_RUNTIME_PYTHON:-$HOME/masters/sallm/.venv/bin/python}"
recovery_id="${SALLM_OFFICIAL_RECOVERY_ID:-}"
runtime_cache="${SALLM_OFFICIAL_RUNTIME_CACHE:-}"

expected_output=/scratch/lmbanr001/masters/sallm/results/official_test/familywise_20260901/t2x_xho
expected_protocol=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_20260901/t2x_xho
if [[ -n "$recovery_id" ]]; then
  [[ "$recovery_id" == pure-gdn-heldout-recovery-20260903-v2 ]]
  expected_output=/scratch/lmbanr001/masters/sallm/results/official_test/familywise_recovery_20260903_v2/t2x
  expected_protocol=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_recovery_20260903_v2/t2x
  [[ "$runtime_cache" == /scratch/lmbanr001/masters/sallm/data/official_recovery_runtime_cache/20260903_v2/t2x_* ]]
  [[ -d "$runtime_cache/hf" ]]
fi

[[ "$snapshot" == "$HOME"/masters/sallm_snapshots/* ]]
[[ "$base" == /scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model ]]
[[ "$adapter" == /scratch/lmbanr001/masters/sallm/checkpoints/frozen_family_winners/t2x_b7_seed42_20260813/final_adapter ]]
[[ "$output" == "$expected_output" ]]
[[ "$protocol_root" == "$expected_protocol" ]]
[[ -d "$base" && -d "$adapter" && ! -e "$output" ]]

export PYTHONPATH="$snapshot/src/main"
if [[ -n "$recovery_id" ]]; then
  export HF_HOME="$runtime_cache/hf"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HF_MODULES_CACHE="$HF_HOME/modules"
  export HF_EVALUATE_CACHE="$HF_HOME/evaluate"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export FLA_DISABLE_BACKEND_DISPATCH=1
export SALLM_SKIP_MAMBA_KERNEL_CHECK=1

"$python_bin" "$snapshot/scripts/create_execution_manifest.py" \
  --verify "$source_manifest" \
  --verify-runtime \
  --expected-repo-root "$snapshot"
"$python_bin" "$snapshot/scripts/create_execution_manifest.py" \
  --verify "$base_manifest" \
  --verify-runtime \
  --expected-repo-root "$snapshot" \
  --expected-artifact-root "$base"
"$python_bin" "$snapshot/scripts/create_execution_manifest.py" \
  --verify "$adapter_manifest" \
  --verify-runtime \
  --expected-repo-root "$snapshot" \
  --expected-artifact-root "$adapter"

args=(
  --config-name eval/run_mamba_t2x_xho
  "eval.eval_model.checkpoint=$base"
  '++eval.eval_model.tie_word_embeddings=true'
  "++eval.eval_model.peft_adapter=$adapter"
  'eval.eval_model.merge_lora=false'
  "eval.evaluation.output_dir=$output"
  'eval.wandb.name=official-pure-gdn-t2x-familywise-20260901'
)

mkdir -p "$protocol_root"
"$python_bin" -m sallm.main "${args[@]}" --cfg job --resolve \
  > "$protocol_root/resolved_config.yaml"
shasum -a 256 "$protocol_root/resolved_config.yaml" \
  > "$protocol_root/resolved_config.yaml.sha256"

if [[ "${SALLM_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  exit 0
fi

"$python_bin" -m sallm.main "${args[@]}"
"$python_bin" "$snapshot/scripts/verify_official_generation_eval.py" \
  --root "$output" \
  --expected-task t2x_xho \
  --expected-language xho \
  --expected-rows 378 \
  --required-metric eval/t2x_xho/all_chrf \
  --output "$output/structural_verification.json"
shasum -a 256 "$output/structural_verification.json" \
  > "$output/structural_verification.json.sha256"
chmod 0444 \
  "$protocol_root/resolved_config.yaml" \
  "$protocol_root/resolved_config.yaml.sha256" \
  "$output/structural_verification.json" \
  "$output/structural_verification.json.sha256"
