#!/bin/bash
set -euo pipefail

/scratch/slurm/bin/purequota

family="${SALLM_OFFICIAL_FAMILY:?missing official family}"
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
recovery_id="${SALLM_OFFICIAL_RECOVERY_ID:-}"
runtime_cache="${SALLM_OFFICIAL_RUNTIME_CACHE:-}"

expected_base=/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model
case "$family" in
  news)
    expected_cache=/scratch/lmbanr001/masters/sallm/data/official_cache/news_v1
    expected_multi=/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/news/stage_a/a0/seed_42_env_correction_20260901/final_adapter
    expected_mono_root=/scratch/lmbanr001/masters/sallm/checkpoints/pure_gdn_mono_familywise_v1/news
    expected_result=/scratch/lmbanr001/masters/sallm/results/official_test/familywise_20260901/news
    expected_protocol=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_20260901/news_v2
    languages=(eng xho)
    declare -A expected_rows=([eng]=4740 [xho]=1485)
    ;;
  intent)
    expected_cache=/scratch/lmbanr001/masters/sallm/data/official_cache/intent_v1
    expected_multi=/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/intent/stage_a/a1/seed_42/final_adapter
    expected_mono_root=/scratch/lmbanr001/masters/sallm/checkpoints/pure_gdn_mono_familywise_v1/intent
    expected_result=/scratch/lmbanr001/masters/sallm/results/official_test/familywise_20260901/intent
    expected_protocol=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_20260901/intent_v2
    languages=(eng sot xho zul)
    declare -A expected_rows=([eng]=3110 [sot]=3200 [xho]=3200 [zul]=3200)
    ;;
  *)
    echo "unsupported official family: $family" >&2
    exit 2
    ;;
esac

if [[ -n "$recovery_id" ]]; then
  [[ "$family" == news ]]
  [[ "$recovery_id" == pure-gdn-heldout-recovery-20260903-v2 ]]
  expected_cache=/scratch/lmbanr001/masters/sallm/data/official_recovery_cache/20260903_v2
  expected_result=/scratch/lmbanr001/masters/sallm/results/official_test/familywise_recovery_20260903_v2/news
  expected_protocol=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_recovery_20260903_v2/news
  [[ "$runtime_cache" == /scratch/lmbanr001/masters/sallm/data/official_recovery_runtime_cache/20260903_v2/news_* ]]
  [[ -d "$runtime_cache/hf" ]]
fi

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

labels=()
adapters=()
adapter_manifests=()
for mode in multi mono; do
  for language in "${languages[@]}"; do
    labels+=("${mode}_${language}")
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

[[ "${#labels[@]}" == "$((${#languages[@]} * 2))" ]]
for label in "${labels[@]}"; do
  [[ ! -e "$result_root/$label" ]]
done

export PYTHONPATH="$snapshot/src/main"
if [[ -n "$recovery_id" ]]; then
  export HF_HOME="$runtime_cache/hf"
else
  export HF_HOME="$dataset_cache/hf"
fi
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_EVALUATE_CACHE="$HF_HOME/evaluate"
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

build_args() {
  local label="$1" language="$2" adapter="$3" task_pack
  if [[ "$family" == news ]]; then
    config="eval/run_llama_masakhanews_${language}"
    task_pack="masakhanews_${language}_test_chat"
  else
    config="eval/run_llama_injongointent_${language}"
    task_pack="injongointent_${language}"
  fi
  args=(
    --config-name "$config"
    "eval.eval_model.checkpoint=$base"
    '++eval.eval_model.tie_word_embeddings=true'
    "++eval.eval_model.peft_adapter=$adapter"
    '++eval.eval_model.merge_lora=false'
    "eval.evaluation.task_packs=[$task_pack]"
    "eval.evaluation.output_dir=$result_root/$label"
    "eval.wandb.name=official-pure-gdn-${family}-${label}-familywise-20260901"
  )
}

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  language="${label#*_}"
  adapter="${adapters[$index]}"
  build_args "$label" "$language" "$adapter"
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
  language="${label#*_}"
  build_args "$label" "$language" "${adapters[$index]}"
  "$python_bin" -m sallm.main "${args[@]}"
  if [[ "$family" == news ]]; then
    expected_pack="masakhanews_${language}_test_chat"
    task_prefix="sallm_masakhanews_${language}_test"
  else
    expected_pack="injongointent_${language}"
    task_prefix="$expected_pack"
  fi
  output="$result_root/$label"
  "$python_bin" "$verifier" \
    --root "$output" \
    --expected-pack "$expected_pack" \
    --expected-task-prefix "$task_prefix" \
    --expected-prompt-count 5 \
    --expected-task-suffix '' \
    --required-metric 'f1,none' \
    --expected-rows "${expected_rows[$language]}" \
    --output "$output/structural_verification.json"
  shasum -a 256 "$output/structural_verification.json" \
    > "$output/structural_verification.json.sha256"
  chmod 0444 \
    "$protocol_root/${label}.resolved_config.yaml" \
    "$protocol_root/${label}.resolved_config.yaml.sha256" \
    "$output/structural_verification.json" \
    "$output/structural_verification.json.sha256"
done
