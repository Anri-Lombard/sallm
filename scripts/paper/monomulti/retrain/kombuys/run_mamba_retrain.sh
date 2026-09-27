#!/usr/bin/env bash
# Mamba retrain on Kombuys, adapted from full_matrix_targeted_recovery_20260922_kombuys_v15/audit/run_v15_kombuys_recovery_arm_20260922_v2.sh
set -euo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1

label="${1:?usage: $0 LABEL GPU}"
gpu="${2:?usage: $0 LABEL GPU}"
snapshot=/scratch/alombard/sallm_snapshots/full-matrix-targeted-recovery-20260919-v15
manifest_sha=928711a05cadc3860ef57df737a02057fc17d013b9758e44c22f7f1bba487fad
base=/scratch/alombard/sallm/assets/mamba-news-common-validation-20260914-v2/base
base_weight_sha=7eaa6b8a1c24ce1a0638c17efb84cdaa44eab5ec167f76881742e2f89414abde
runtime=/scratch/alombard/sallm/.venv/bin/python
root=/scratch/alombard/sallm/results/monomulti_retrain_20260924
runner="$snapshot/.audit/run_train_validation_only_20260914.py"
canary_marker=/scratch/alombard/sallm/results/full_matrix_targeted_recovery_20260922_kombuys_v15/canary/CANARY_PASSED.json

[[ "$(sha256sum "$snapshot/SNAPSHOT_MANIFEST.sha256" | cut -d' ' -f1)" == "$manifest_sha" ]]
(cd "$snapshot" && sha256sum -c SNAPSHOT_MANIFEST.sha256 --quiet)
[[ "$(sha256sum "$base/pytorch_model.bin" | cut -d' ' -f1)" == "$base_weight_sha" ]]
jq -e --arg manifest "$manifest_sha" '.status == "CANARY_PASSED" and .snapshot_manifest_sha256 == $manifest and .mamba_validation_label_microbatch == 17 and .heldout_access == false' "$canary_marker" >/dev/null

bs=2
keep_all=(
  finetune.training.save_strategy=epoch
  ++finetune.training.save_total_limit=null
  ++finetune.training.save_only_model=true
)
case "$label" in
  mamba2_sib_mono_*)
    lang="${label##*_}"; config="mamba_sib_$lang"
    declare -A ga_map=([afr]=64 [eng]=128 [nso]=32 [sot]=32 [xho]=32 [zul]=16)
    ga="${ga_map[$lang]}"
    extra=("${keep_all[@]}" ++finetune.training.load_best_model_at_end=false finetune.training.early_stopping_patience=null)
    ;;
  mamba2_sib_multi)
    config=mamba_sib_all; ga=32
    extra=("${keep_all[@]}" ++finetune.training.load_best_model_at_end=false finetune.training.early_stopping_patience=null)
    ;;
  mamba2_pos_multi)
    config=mamba_pos_all; ga=32
    extra=("${keep_all[@]}")
    ;;
  mamba2_ner_mono_tsn|mamba2_ner_mono_xho|mamba2_ner_mono_zul)
    # effective batch 64 (Mamba Multi NER setting) as 8 x 8 instead of 2 x 32 for throughput
    config="mamba_ner_${label##*_}"; ga=8; bs=8
    extra=("${keep_all[@]}")
    ;;
  *) echo "ERROR: unknown label $label" >&2; exit 4 ;;
esac

output="$root/train/$label"
[[ ! -e "$output" ]]
mkdir -p "$output"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export HF_HOME=/scratch/alombard/sallm/hf
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=disabled
export WANDB_SILENT=true
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export SALLM_T2X_TRAIN_VALIDATION_ONLY=1
export SALLM_T2X_CACHE_DIR="$root/assets/t2x_train_validation_only"
export SALLM_MAMBA_VALIDATION_LABEL_MICROBATCH=17
export PYTHONPATH="$snapshot/src/main"
export TRITON_CACHE_DIR="$output/triton-cache"

args=(
  --config-name "finetune/$config"
  "finetune.model.init_checkpoint=$base"
  "finetune.tokenizer.path=$base"
  "finetune.training.output_dir=$output"
  "finetune.training.logging_dir=$output/logs"
  "finetune.training.run_name=$label-monomulti-retrain-20260924"
  ++finetune.training.seed=42
  finetune.training.report_to=none
  finetune.hub.enabled=false
  finetune.hub.push_adapter=false
  finetune.hub.push_merged=false
  "hydra.run.dir=$output/hydra"
  "finetune.peft.kwargs.target_modules=[in_proj,x_proj]"
  "finetune.training.per_device_train_batch_size=$bs"
  finetune.training.per_device_eval_batch_size=1
  "finetune.training.gradient_accumulation_steps=$ga"
  "${extra[@]}"
)

"$runtime" "$runner" "${args[@]}" --cfg job --resolve > "$output/resolved_config.stdout"
sed -n '/^finetune:/,$p' "$output/resolved_config.stdout" > "$output/resolved_config.yaml"
[[ -s "$output/resolved_config.yaml" ]]
! grep -Eiq '(^|[[:space:]])(test|test_split)[[:space:]]*:' "$output/resolved_config.yaml"
printf '%s\n' "${args[@]}" > "$output/overrides.txt"
{
  echo "label=$label"; echo "snapshot=$snapshot"; echo "snapshot_manifest_sha256=$manifest_sha"
  echo "base=$base"; echo "base_weight_sha256=$base_weight_sha"; echo "gpu_index=$gpu"
  nvidia-smi -i "$gpu" --query-gpu=name,uuid,memory.total --format=csv,noheader
  echo "start=$(date -Is)"
} > "$output/run_info.txt"
"$runtime" "$runner" "${args[@]}"
echo "end=$(date -Is)" >> "$output/run_info.txt"
[[ -s "$output/final_adapter/adapter_config.json" ]]
(cd "$output/final_adapter" && find . -type f ! -path './.cache/*' | sort | xargs sha256sum) > "$output/final_adapter.sha256"
echo "final_adapter_tree_sha256=$(sha256sum "$output/final_adapter.sha256" | cut -d' ' -f1)" >> "$output/run_info.txt"
echo DONE
