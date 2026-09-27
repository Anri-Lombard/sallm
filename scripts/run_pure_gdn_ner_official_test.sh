#!/bin/bash
set -euo pipefail

/scratch/slurm/bin/purequota

snapshot="${SALLM_OFFICIAL_SNAPSHOT:?missing immutable snapshot}"
source_manifest="${SALLM_OFFICIAL_SOURCE_MANIFEST:?missing source manifest}"
base_manifest="${SALLM_OFFICIAL_BASE_MANIFEST:?missing base manifest}"
base="${SALLM_OFFICIAL_BASE:?missing base checkpoint}"
multi_adapter="${SALLM_OFFICIAL_MULTI_ADAPTER:?missing frozen Multi adapter}"
multi_manifest="${SALLM_OFFICIAL_MULTI_ADAPTER_MANIFEST:?missing Multi adapter manifest}"
mono_tsn="${SALLM_OFFICIAL_MONO_TSN:?missing frozen Tswana Mono adapter}"
mono_tsn_manifest="${SALLM_OFFICIAL_MONO_TSN_MANIFEST:?missing Tswana Mono manifest}"
mono_xho="${SALLM_OFFICIAL_MONO_XHO:?missing frozen Xhosa Mono adapter}"
mono_xho_manifest="${SALLM_OFFICIAL_MONO_XHO_MANIFEST:?missing Xhosa Mono manifest}"
mono_zul="${SALLM_OFFICIAL_MONO_ZUL:?missing frozen Zulu Mono adapter}"
mono_zul_manifest="${SALLM_OFFICIAL_MONO_ZUL_MANIFEST:?missing Zulu Mono manifest}"
result_root="${SALLM_OFFICIAL_RESULT_ROOT:?missing official result root}"
protocol_root="${SALLM_OFFICIAL_PROTOCOL_ROOT:?missing protocol root}"
python_bin="${SALLM_RUNTIME_PYTHON:-$HOME/masters/sallm/.venv/bin/python}"
recovery_id="${SALLM_OFFICIAL_RECOVERY_ID:-}"
runtime_cache="${SALLM_OFFICIAL_RUNTIME_CACHE:-}"

expected_base=/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model
expected_multi=/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/ner/stage_b/b7/seed_42/final_adapter
expected_mono_root=/scratch/lmbanr001/masters/sallm/checkpoints/pure_gdn_mono_familywise_v1/ner
expected_result=/scratch/lmbanr001/masters/sallm/results/official_test/familywise_20260901/ner
expected_protocol=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_20260901/ner_kernel_runtime_correction_v3
if [[ -n "$recovery_id" ]]; then
  [[ "$recovery_id" == pure-gdn-heldout-recovery-20260903-v2 ]]
  expected_result=/scratch/lmbanr001/masters/sallm/results/official_test/familywise_recovery_20260903_v2/ner
  expected_protocol=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_recovery_20260903_v2/ner
  [[ "$runtime_cache" == /scratch/lmbanr001/masters/sallm/data/official_recovery_runtime_cache/20260903_v2/ner_* ]]
  [[ -d "$runtime_cache/hf" ]]
fi

[[ "$snapshot" == "$HOME"/masters/sallm_snapshots/* ]]
[[ "$base" == "$expected_base" ]]
[[ "$multi_adapter" == "$expected_multi" ]]
[[ "$mono_tsn" == "$expected_mono_root/tsn/seed_42_cacheid_correction_20260901/final_adapter" ]]
[[ "$mono_xho" == "$expected_mono_root/xho/seed_42_cacheid_correction_20260901/final_adapter" ]]
[[ "$mono_zul" == "$expected_mono_root/zul/seed_42_cacheid_correction_20260901/final_adapter" ]]
[[ "$result_root" == "$expected_result" ]]
[[ "$protocol_root" == "$expected_protocol" ]]
[[ -d "$base" && -d "$multi_adapter" && -d "$mono_tsn" && -d "$mono_xho" && -d "$mono_zul" ]]

labels=(multi_tsn multi_xho multi_zul mono_tsn mono_xho mono_zul)
configs=(
  eval/run_mamba_masakhaner_tsn
  eval/run_mamba_masakhaner_xho
  eval/run_mamba_masakhaner_zul
  eval/run_mamba_masakhaner_tsn
  eval/run_mamba_masakhaner_xho
  eval/run_mamba_masakhaner_zul
)
tasks=(masakhaner_tsn masakhaner_xho masakhaner_zul masakhaner_tsn masakhaner_xho masakhaner_zul)
task_prefixes=(sallm_masakhaner_tn sallm_masakhaner_xh sallm_masakhaner_zu sallm_masakhaner_tn sallm_masakhaner_xh sallm_masakhaner_zu)
expected_rows=(4980 5000 5000 4980 5000 5000)
adapters=("$multi_adapter" "$multi_adapter" "$multi_adapter" "$mono_tsn" "$mono_xho" "$mono_zul")
adapter_manifests=("$multi_manifest" "$multi_manifest" "$multi_manifest" "$mono_tsn_manifest" "$mono_xho_manifest" "$mono_zul_manifest")

for label in "${labels[@]}"; do
  [[ ! -e "$result_root/$label" ]]
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

"$python_bin" "$snapshot/scripts/create_execution_manifest.py" \
  --verify "$source_manifest" \
  --verify-runtime \
  --expected-repo-root "$snapshot"
"$python_bin" "$snapshot/scripts/create_execution_manifest.py" \
  --verify "$base_manifest" \
  --verify-runtime \
  --expected-repo-root "$snapshot" \
  --expected-artifact-root "$base"

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  adapter="${adapters[$index]}"
  adapter_manifest="${adapter_manifests[$index]}"
  output="$result_root/$label"
  args=(
    --config-name "${configs[$index]}"
    "eval.eval_model.checkpoint=$base"
    '++eval.eval_model.tie_word_embeddings=true'
    "++eval.eval_model.peft_adapter=$adapter"
    'eval.eval_model.merge_lora=false'
    "eval.evaluation.output_dir=$output"
    "eval.wandb.name=official-pure-gdn-ner-${label}-familywise-20260901"
  )

  "$python_bin" "$snapshot/scripts/create_execution_manifest.py" \
    --verify "$adapter_manifest" \
    --verify-runtime \
    --expected-repo-root "$snapshot" \
    --expected-artifact-root "$adapter"
  mkdir -p "$protocol_root"
  "$python_bin" -m sallm.main "${args[@]}" --cfg job --resolve \
    > "$protocol_root/${label}.resolved_config.yaml"
  shasum -a 256 "$protocol_root/${label}.resolved_config.yaml" \
    > "$protocol_root/${label}.resolved_config.yaml.sha256"
done

if [[ "${SALLM_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  exit 0
fi

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  task="${tasks[$index]}"
  output="$result_root/$label"
  args=(
    --config-name "${configs[$index]}"
    "eval.eval_model.checkpoint=$base"
    '++eval.eval_model.tie_word_embeddings=true'
    "++eval.eval_model.peft_adapter=${adapters[$index]}"
    'eval.eval_model.merge_lora=false'
    "eval.evaluation.output_dir=$output"
    "eval.wandb.name=official-pure-gdn-ner-${label}-familywise-20260901"
  )

  "$python_bin" -m sallm.main "${args[@]}"
  "$python_bin" "$snapshot/scripts/verify_official_task_pack_eval.py" \
    --root "$output" \
    --expected-pack "$task" \
    --expected-task-prefix "${task_prefixes[$index]}" \
    --expected-rows "${expected_rows[$index]}" \
    --output "$output/structural_verification.json"
  shasum -a 256 "$output/structural_verification.json" \
    > "$output/structural_verification.json.sha256"
  chmod 0444 \
    "$protocol_root/${label}.resolved_config.yaml" \
    "$protocol_root/${label}.resolved_config.yaml.sha256" \
    "$output/structural_verification.json" \
    "$output/structural_verification.json.sha256"
done
