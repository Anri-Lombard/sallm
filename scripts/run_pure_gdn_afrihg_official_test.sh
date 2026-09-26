#!/bin/bash
set -euo pipefail

/scratch/slurm/bin/purequota

snapshot="${SALLM_OFFICIAL_SNAPSHOT:?missing immutable snapshot}"
source_manifest="${SALLM_OFFICIAL_SOURCE_MANIFEST:?missing source manifest}"
base_manifest="${SALLM_OFFICIAL_BASE_MANIFEST:?missing base manifest}"
dataset_manifest="${SALLM_OFFICIAL_DATASET_MANIFEST:?missing dataset manifest}"
base="${SALLM_OFFICIAL_BASE:?missing base checkpoint}"
dataset_root="${SALLM_OFFICIAL_DATASET_ROOT:?missing dataset root}"
multi_adapter="${SALLM_OFFICIAL_MULTI_ADAPTER:?missing frozen Multi adapter}"
multi_manifest="${SALLM_OFFICIAL_MULTI_ADAPTER_MANIFEST:?missing Multi manifest}"
mono_xho="${SALLM_OFFICIAL_MONO_XHO:?missing frozen Xhosa Mono adapter}"
mono_xho_manifest="${SALLM_OFFICIAL_MONO_XHO_MANIFEST:?missing Xhosa Mono manifest}"
mono_zul="${SALLM_OFFICIAL_MONO_ZUL:?missing frozen Zulu Mono adapter}"
mono_zul_manifest="${SALLM_OFFICIAL_MONO_ZUL_MANIFEST:?missing Zulu Mono manifest}"
result_root="${SALLM_OFFICIAL_RESULT_ROOT:?missing official result root}"
protocol_root="${SALLM_OFFICIAL_PROTOCOL_ROOT:?missing protocol root}"
python_bin="${SALLM_RUNTIME_PYTHON:-$HOME/masters/sallm/.venv/bin/python}"
preflight_only="${SALLM_PREFLIGHT_ONLY:-0}"
recovery_id="${SALLM_OFFICIAL_RECOVERY_ID:-}"
runtime_cache="${SALLM_OFFICIAL_RUNTIME_CACHE:-}"
continue_after_verifier_fix="${SALLM_OFFICIAL_CONTINUE_AFTER_VERIFIER_FIX:-0}"
correction_root="${SALLM_OFFICIAL_CORRECTION_ROOT:-}"
generation_verifier="${SALLM_OFFICIAL_GENERATION_VERIFIER:-$snapshot/scripts/verify_official_generation_eval.py}"

expected_base=/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model
expected_dataset=/home/lmbanr001/masters/sallm/data/afrihg_cache
expected_multi=/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/afrihg/stage_b/b7/seed_42/final_adapter
expected_mono_root=/scratch/lmbanr001/masters/sallm/checkpoints/pure_gdn_mono_familywise_v1/afrihg
expected_result=/scratch/lmbanr001/masters/sallm/results/official_test/familywise_20260901/afrihg
expected_protocol=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_20260901/afrihg_cache_pinned_v1
if [[ -n "$recovery_id" ]]; then
  [[ "$recovery_id" == pure-gdn-heldout-recovery-20260903-v2 ]]
  expected_result=/scratch/lmbanr001/masters/sallm/results/official_test/familywise_recovery_20260903_v2/afrihg
  expected_protocol=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_recovery_20260903_v2/afrihg
  [[ "$runtime_cache" == /scratch/lmbanr001/masters/sallm/data/official_recovery_runtime_cache/20260903_v2/afrihg_* ]]
  [[ -d "$runtime_cache/hf" ]]
fi
if [[ "$continue_after_verifier_fix" == 1 ]]; then
  [[ "$recovery_id" == pure-gdn-heldout-recovery-20260903-v2 ]]
  [[ "$correction_root" == /scratch/lmbanr001/masters/sallm/overlays/pure-gdn-afrihg-recovery-verifier-correction-20260903-* ]]
  [[ "$(readlink -f "$0")" == "$correction_root/run_pure_gdn_afrihg_official_test.sh" ]]
  [[ "$generation_verifier" == "$correction_root/verify_official_generation_eval.py" ]]
  (
    cd "$correction_root"
    shasum -a 256 -c overlay_sha256.txt
  )
else
  [[ "$continue_after_verifier_fix" == 0 ]]
  [[ -z "$correction_root" ]]
fi

[[ "$snapshot" == "$HOME"/masters/sallm_snapshots/* ]]
[[ "$base" == "$expected_base" ]]
[[ "$dataset_root" == "$expected_dataset" ]]
[[ "$multi_adapter" == "$expected_multi" ]]
[[ "$mono_xho" == "$expected_mono_root/xho/seed_42/final_adapter" ]]
[[ "$mono_zul" == "$expected_mono_root/zul/seed_42/final_adapter" ]]
[[ "$result_root" == "$expected_result" ]]
[[ "$protocol_root" == "$expected_protocol" ]]
[[ -d "$base" && -d "$dataset_root" && -d "$multi_adapter" ]]
[[ -d "$mono_xho" && -d "$mono_zul" ]]

labels=(multi_xho multi_zul mono_xho mono_zul)
configs=(
  eval/run_llama_afrihg_xho
  eval/run_llama_afrihg_zul
  eval/run_llama_afrihg_xho
  eval/run_llama_afrihg_zul
)
tasks=(afrihg_xho afrihg_zul afrihg_xho afrihg_zul)
languages=(xho zul xho zul)
expected_rows=(1305 1776 1305 1776)
adapters=("$multi_adapter" "$multi_adapter" "$mono_xho" "$mono_zul")
adapter_manifests=(
  "$multi_manifest"
  "$multi_manifest"
  "$mono_xho_manifest"
  "$mono_zul_manifest"
)

for label in "${labels[@]}"; do
  if [[ "$continue_after_verifier_fix" == 1 && "$label" == multi_xho ]]; then
    [[ -d "$result_root/$label" ]]
    [[ -f "$result_root/$label/afrihg_xho/examples.jsonl" ]]
    [[ ! -e "$result_root/$label/structural_verification.json" ]]
  else
    [[ ! -e "$result_root/$label" ]]
  fi
done

export PYTHONPATH="$snapshot/src/main"
if [[ -n "$recovery_id" ]]; then
  export HF_HOME="$runtime_cache/hf"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HF_MODULES_CACHE="$HF_HOME/modules"
  export HF_EVALUATE_CACHE="$HF_HOME/evaluate"
fi
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export FLA_DISABLE_BACKEND_DISPATCH=1
export SALLM_SKIP_MAMBA_KERNEL_CHECK=1
export SALLM_AFRIHG_CACHE_ONLY=1
export SALLM_AFRIHG_CACHE_DIR="$dataset_root"

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
  --verify "$dataset_manifest" \
  --expected-repo-root "$snapshot" \
  --expected-artifact-root "$dataset_root"

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  adapter="${adapters[$index]}"
  output="$result_root/$label"
  args=(
    --config-name "${configs[$index]}"
    "eval.eval_model.checkpoint=$base"
    '++eval.eval_model.tie_word_embeddings=true'
    "++eval.eval_model.peft_adapter=$adapter"
    'eval.eval_model.merge_lora=false'
    "eval.evaluation.output_dir=$output"
    "eval.wandb.name=official-pure-gdn-afrihg-${label}-familywise-20260901"
  )

  "$python_bin" "$snapshot/scripts/create_execution_manifest.py" \
    --verify "${adapter_manifests[$index]}" \
    --verify-runtime \
    --expected-repo-root "$snapshot" \
    --expected-artifact-root "$adapter"
  if [[ "$preflight_only" == 1 ]]; then
    mkdir -p "$protocol_root"
    "$python_bin" -m sallm.main "${args[@]}" --cfg job --resolve \
      > "$protocol_root/${label}.resolved_config.yaml"
    shasum -a 256 "$protocol_root/${label}.resolved_config.yaml" \
      > "$protocol_root/${label}.resolved_config.yaml.sha256"
  else
    (
      cd "$protocol_root"
      shasum -a 256 -c "${label}.resolved_config.yaml.sha256"
    )
  fi
done

[[ "$preflight_only" == 1 ]] && exit 0

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  output="$result_root/$label"
  args=(
    --config-name "${configs[$index]}"
    "eval.eval_model.checkpoint=$base"
    '++eval.eval_model.tie_word_embeddings=true'
    "++eval.eval_model.peft_adapter=${adapters[$index]}"
    'eval.eval_model.merge_lora=false'
    "eval.evaluation.output_dir=$output"
    "eval.wandb.name=official-pure-gdn-afrihg-${label}-familywise-20260901"
  )

  if [[ "$continue_after_verifier_fix" != 1 || "$label" != multi_xho ]]; then
    "$python_bin" -m sallm.main "${args[@]}"
  fi
  "$python_bin" "$generation_verifier" \
    --root "$output" \
    --expected-task "${tasks[$index]}" \
    --expected-language "${languages[$index]}" \
    --expected-rows "${expected_rows[$index]}" \
    --required-metric "eval/${tasks[$index]}/all_chrf" \
    --output "$output/structural_verification.json"
  shasum -a 256 "$output/structural_verification.json" \
    > "$output/structural_verification.json.sha256"
  chmod 0444 \
    "$protocol_root/${label}.resolved_config.yaml" \
    "$protocol_root/${label}.resolved_config.yaml.sha256" \
    "$output/structural_verification.json" \
    "$output/structural_verification.json.sha256"
done
