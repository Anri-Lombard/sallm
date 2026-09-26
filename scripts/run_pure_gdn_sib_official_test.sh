#!/bin/bash
set -euo pipefail

/scratch/slurm/bin/purequota

snapshot="${SALLM_OFFICIAL_SNAPSHOT:?missing immutable snapshot}"
source_manifest="${SALLM_OFFICIAL_SOURCE_MANIFEST:?missing source manifest}"
base_manifest="${SALLM_OFFICIAL_BASE_MANIFEST:?missing base manifest}"
dataset_manifest="${SALLM_OFFICIAL_DATASET_MANIFEST:?missing dataset manifest}"
base="${SALLM_OFFICIAL_BASE:?missing base checkpoint}"
dataset_cache="${SALLM_OFFICIAL_DATASET_CACHE:?missing frozen dataset cache}"
multi_adapter="${SALLM_OFFICIAL_MULTI_ADAPTER:?missing frozen Multi adapter}"
multi_manifest="${SALLM_OFFICIAL_MULTI_ADAPTER_MANIFEST:?missing Multi manifest}"
mono_root="${SALLM_OFFICIAL_MONO_ROOT:?missing frozen Mono root}"
result_root="${SALLM_OFFICIAL_RESULT_ROOT:?missing official result root}"
protocol_root="${SALLM_OFFICIAL_PROTOCOL_ROOT:?missing protocol root}"
python_bin="${SALLM_RUNTIME_PYTHON:-$HOME/masters/sallm/.venv/bin/python}"
preflight_only="${SALLM_PREFLIGHT_ONLY:-0}"

expected_base=/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model
expected_cache=/scratch/lmbanr001/masters/sallm/data/official_cache/sib_v1
expected_multi=/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/sib/stage_a/a0/seed_42/final_adapter
expected_mono_root=/scratch/lmbanr001/masters/sallm/checkpoints/pure_gdn_mono_familywise_v1/sib
expected_result=/scratch/lmbanr001/masters/sallm/results/official_test/familywise_20260901/sib
expected_protocol=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_20260901/sib_v3
cache_tree_manifest="$dataset_cache/cache_tree_sha256.txt"
verifier="$snapshot/scripts/verify_official_task_pack_eval.py"

[[ "$snapshot" == "$HOME"/masters/sallm_snapshots/* ]]
[[ "$base" == "$expected_base" ]]
[[ "$dataset_cache" == "$expected_cache" ]]
[[ "$multi_adapter" == "$expected_multi" ]]
[[ "$mono_root" == "$expected_mono_root" ]]
[[ "$result_root" == "$expected_result" ]]
[[ "$protocol_root" == "$expected_protocol" ]]
[[ -d "$base" && -d "$dataset_cache" && -d "$multi_adapter" ]]
[[ -f "$cache_tree_manifest" && -f "$verifier" ]]
(
  cd "$dataset_cache"
  shasum -a 256 -c "$(basename "$cache_tree_manifest")"
)

languages=(afr eng nso sot xho zul)
labels=()
configs=()
tasks=()
adapters=()
adapter_manifests=()
for mode in multi mono; do
  for language in "${languages[@]}"; do
    labels+=("${mode}_${language}")
    configs+=("eval/run_llama_sib_${language}")
    tasks+=("sib_${language}")
    if [[ "$mode" == multi ]]; then
      adapters+=("$multi_adapter")
      adapter_manifests+=("$multi_manifest")
    else
      adapter="$mono_root/$language/seed_42/final_adapter"
      manifest_variable="SALLM_OFFICIAL_MONO_${language^^}_MANIFEST"
      manifest="${!manifest_variable:?missing $manifest_variable}"
      [[ -d "$adapter" ]]
      adapters+=("$adapter")
      adapter_manifests+=("$manifest")
    fi
  done
done

[[ "${#labels[@]}" == 12 ]]
for label in "${labels[@]}"; do
  [[ ! -e "$result_root/$label" ]]
done

export PYTHONPATH="$snapshot/src/main"
export HF_HOME="$dataset_cache/hf"
export HF_DATASETS_CACHE="$dataset_cache/hf/datasets"
export HF_HUB_CACHE="$dataset_cache/hf/hub"
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
"$python_bin" "$snapshot/scripts/create_execution_manifest.py" \
  --verify "$dataset_manifest" \
  --verify-runtime \
  --expected-repo-root "$snapshot" \
  --expected-artifact-root "$dataset_cache"

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
    "eval.wandb.name=official-pure-gdn-sib-${label}-familywise-20260901"
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

for label in "${labels[@]}"; do
  [[ ! -w "$protocol_root/${label}.resolved_config.yaml" ]]
  [[ ! -w "$protocol_root/${label}.resolved_config.yaml.sha256" ]]
done

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
    "eval.wandb.name=official-pure-gdn-sib-${label}-familywise-20260901"
  )

  "$python_bin" -m sallm.main "${args[@]}"
  "$python_bin" "$verifier" \
    --root "$output" \
    --expected-pack "${tasks[$index]}" \
    --expected-task-prefix "${tasks[$index]}" \
    --expected-prompt-count 5 \
    --expected-task-suffix '' \
    --required-metric 'f1,none' \
    --expected-rows 1020 \
    --output "$output/structural_verification.json"
  shasum -a 256 "$output/structural_verification.json" \
    > "$output/structural_verification.json.sha256"
  chmod 0444 \
    "$protocol_root/${label}.resolved_config.yaml" \
    "$protocol_root/${label}.resolved_config.yaml.sha256" \
    "$output/structural_verification.json" \
    "$output/structural_verification.json.sha256"
done
