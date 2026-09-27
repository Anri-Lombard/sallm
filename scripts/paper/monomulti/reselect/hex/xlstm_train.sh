#!/bin/bash
# Retrain an xLSTM Mono/Multi News or Intent adapter with its original July recipe
# (xlstm_news_recovery_20260728/*_rolefix_val_r2, intent_recovery_20260729/xlstm_mono_*_r4_r2,
# intent_recovery_20260729/xlstm_clean_r4), changing only checkpoint handling: save every epoch,
# no early stopping, no keep-best. Evidence: $root/tmp/xlstm_recipe/.
# Usage: xlstm_train.sh LABEL            (train, run on a GPU node by run_unit.sh)
#        xlstm_train.sh LABEL --resolve-only   (CPU: write resolved config to $root/tmp/xlstm_recipe/dryrun/LABEL)
set -euo pipefail
umask 022
label="${1:?label}"
mode="${2:-train}"
root=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925
code="$root/jobs/xlstm_src_20260729"
code_manifest_sha=1ed0dadffb88b2e97c3d7e03ffac7f42a27a325dd699b89a0f69773a620f919c
wrapper="$root/jobs/run_xlstm_july_train.py"
wrapper_sha=9bba936ec31644ab7635a7e86ca818b6f458d4beced1e910f3736f872bc628ff
venv=/scratch/lmbanr001/masters/sallm_venv
venv_pkgset_sha=5f51b7436a56838208d25a04a778737416cceb06c38987a306e3c422a5f110cf
overlay=/scratch/lmbanr001/masters/sallm_recovery/xlstm_sib_mono_reconstruction_20260914_v2/environment-overlay-datasets-4.4.2
overlay_tree_sha=ed6b5b5010ea637ea42d82af955c8c6eb364aadf08a0f591e8460b5e4f25de01
base_id=anrilombard/sallm-xlstm-125m-native-3epoch-20260531
base_snap=/scratch/lmbanr001/hf/hub/models--anrilombard--sallm-xlstm-125m-native-3epoch-20260531/snapshots/ba2ff845335c8cbf750f8f6f3ebc09468008fef9
base_weight_sha=4e3e37db9a43089e1683a56c4940bc05e9e446ae7da82b5dc5d8621beda09674
base_config_sha=8f837e6d7efa905d958cf11308187fdc07aec7a218caa5390a5b1fe5089663c9
tokenizer=/home/lmbanr001/masters/sallm/tokenizer/sallm_bpe_tokenizer
intent_snap=/scratch/lmbanr001/hf/hub/datasets--masakhane--InjongoIntent/snapshots/fe4be3882a1614161dfe231ec793197bb74f4b44
ckpt=/scratch/lmbanr001/masters/sallm/checkpoints

case "$mode" in
  train) out="$root/train/$label" ;;
  --resolve-only) out="$root/tmp/xlstm_recipe/dryrun/$label" ;;
  *) echo "ERROR: unknown mode $mode" >&2; exit 4 ;;
esac

# Only checkpoint handling differs from the original runs.
select_args=(
  finetune.training.eval_strategy=epoch
  finetune.training.save_strategy=epoch
  ++finetune.training.load_best_model_at_end=false
  ++finetune.training.metric_for_best_model=eval_classification/all_f1
  ++finetune.training.greater_is_better=true
  ++finetune.training.save_total_limit=null
  ++finetune.training.save_only_model=true
  ++finetune.training.early_stopping_patience=null
)
hub_args=(finetune.hub.enabled=false finetune.hub.push_adapter=false finetune.hub.push_merged=false)

case "$label" in
  xlstm_news_mono_eng|xlstm_news_mono_xho|xlstm_news_multi)
    case "$label" in
      xlstm_news_mono_eng) config=xlstm_news_eng; orig="$ckpt/xlstm_news_recovery_20260728/eng_rolefix_val_r2"; orig_job=1127017 ;;
      xlstm_news_mono_xho) config=xlstm_news_xho; orig="$ckpt/xlstm_news_recovery_20260728/xho_rolefix_val_r2"; orig_job=1127018 ;;
      xlstm_news_multi) config=xlstm_news_all; orig="$ckpt/xlstm_news_recovery_20260728/all_rolefix_val_r2"; orig_job=1127019 ;;
    esac
    recipe_args=()
    ;;
  xlstm_intent_mono_eng|xlstm_intent_mono_sot|xlstm_intent_mono_xho|xlstm_intent_mono_zul)
    lang="${label##*_}"
    config=xlstm_injongointent_all
    orig="$ckpt/intent_recovery_20260729/xlstm_mono_${lang}_r4_r2"
    case "$lang" in eng) orig_job=1137486 ;; sot) orig_job=1137492 ;; xho) orig_job=1137488 ;; zul) orig_job=1137490 ;; esac
    recipe_args=(
      finetune.peft.kwargs.r=4
      finetune.peft.kwargs.lora_alpha=8
      finetune.peft.kwargs.lora_dropout=0.05
      "finetune.peft.kwargs.target_modules=[q,k,v,out_proj]"
      "finetune.dataset.subset=$lang"
      "finetune.dataset.languages=[$lang]"
      "finetune.training.run_name=xlstm-intent-mono-$lang-r4-r2"
      finetune.training.learning_rate=0.00022056
      finetune.training.per_device_train_batch_size=8
      finetune.training.per_device_eval_batch_size=4
      finetune.training.gradient_accumulation_steps=4
      finetune.training.num_train_epochs=8
      ++finetune.training.pad_to_multiple_of=64
      "finetune.wandb.name=xlstm-intent-mono-$lang-r4-r2"
    )
    ;;
  xlstm_intent_multi)
    config=xlstm_injongointent_all
    orig="$ckpt/intent_recovery_20260729/xlstm_clean_r4"
    orig_job=1130583
    recipe_args=(
      "finetune.peft.kwargs.target_modules=[q,k,v,out_proj]"
      finetune.training.learning_rate=2.2056e-4
      finetune.training.lr_scheduler_type=constant_with_warmup
      finetune.training.per_device_train_batch_size=8
      finetune.training.gradient_accumulation_steps=4
      finetune.training.num_train_epochs=8
      ++finetune.training.pad_to_multiple_of=64
      finetune.dataset.max_seq_length=1024
      ++finetune.training.early_stopping_threshold=0.001
      finetune.wandb.project=sallm-ft
    )
    ;;
  *) echo "ERROR: unknown label $label" >&2; exit 4 ;;
esac

if [[ "$mode" == train ]]; then
  [[ ! -e "$out/final_adapter" ]] || { echo "already trained: $out"; exit 0; }
  [[ ! -e "$out" ]] || { echo "ERROR: partial output exists: $out" >&2; exit 5; }
else
  rm -rf "$out"
fi

args=(
  "finetune.model.init_checkpoint=$base_id"
  "finetune.training.output_dir=$out"
  "finetune.training.logging_dir=$out/logs"
  "${recipe_args[@]}"
  "${select_args[@]}"
  finetune.training.report_to=none
  "${hub_args[@]}"
  "hydra.run.dir=$out/hydra"
)

[[ "$(sha256sum "$code/SOURCE_MANIFEST.sha256" | cut -d' ' -f1)" == "$code_manifest_sha" ]]
(cd "$code" && sha256sum -c SOURCE_MANIFEST.sha256 --quiet)
[[ "$(sha256sum "$wrapper" | cut -d' ' -f1)" == "$wrapper_sha" ]]
[[ "$(cd "$venv/lib/python3.12/site-packages" && ls -d *.dist-info | LC_ALL=C sort | sha256sum | cut -d' ' -f1)" == "$venv_pkgset_sha" ]]
[[ "$(cd "$overlay" && find . -type f ! -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)" == "$overlay_tree_sha" ]]
[[ "$(sha256sum "$venv/lib/python3.12/site-packages/transformers/models/xlstm/modeling_xlstm.py" | cut -d' ' -f1)" == c8361b620a21994602f930e6d62a4389ab23e6beeba956cb724ee717d3596ae3 ]]
[[ "$(sha256sum "$base_snap/pytorch_model.bin" | cut -d' ' -f1)" == "$base_weight_sha" ]]
[[ "$(sha256sum "$base_snap/config.json" | cut -d' ' -f1)" == "$base_config_sha" ]]
(cd "$tokenizer" && sha256sum -c --quiet <<'EOF'
446895905ea9b20c746317eefd0c6a3b097bcbbef71e8e44b0bf9772d664782a  tokenizer.json
169bedadc3f18d3d1bad46dd10450d811f7a14dc50bb752f0c30dc899b5f0b3a  tokenizer_config.json
fdd1575e8bd811652f3708f5d70495fd581ca5b59111e6edaa903e43783ce944  special_tokens_map.json
EOF
)
(cd "$intent_snap" && sha256sum -c --quiet <<'EOF'
88b7121da2cc5cfdcbee354f651d78a803e0e1cb16d998107621e7f00dbaf334  eng/test.jsonl
156347e0097c08408fbc8ba4582fd383c3255072754ed91c3f703da72a4474f3  eng/train.jsonl
40219dc71bf2cf13517fbf34cfb99eb41afa2fe358066c1f27e58c3a0aeb4432  sot/test.jsonl
3104b770c019142fd655ca3e7b43224d09c790bcd6ce8583fd04573f57d1dd1e  sot/train.jsonl
402fb63050c91fc8328136158249bef24442bee49c69bc51c2849746034d0493  xho/test.jsonl
e23bc692d472f9344aaf93c73db8ecc4c8f3723e82efa98be1b99f10c525d66d  xho/train.jsonl
8e14a16543f7a336dae4061dbd7b0302e02106746bf1fb47a6d68e9d8e77ef59  zul/test.jsonl
2f5dd46df757217fb706f0e7cb2231e9a211d120e227f431905e70f77a8e5d40  zul/train.jsonl
EOF
)

export HF_HOME=/scratch/lmbanr001/hf HF_DATASETS_CACHE=/scratch/lmbanr001/hf/datasets HF_METRICS_CACHE=/scratch/lmbanr001/hf/metrics
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export SALLM_DISABLE_TASK_METRICS=0
unset SALLM_GENERAL_SELECTION_PROTOCOL SALLM_EXECUTION_MANIFEST
export SALLM_INJONGOINTENT_BASE_URL="file://$intent_snap"
export SALLM_JOB_NAME="$label-reselect" TOKENIZERS_PARALLELISM=true HYDRA_FULL_ERROR=1 TORCH_DISTRIBUTED_TIMEOUT=7200
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export PYTHONPATH="$code/src/main:$overlay"
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="$out/triton-cache"
mkdir -p "$out"
cd "$code"

"$venv/bin/python" -c 'import sallm,datasets;print("sallm from",sallm.__file__);print("datasets",datasets.__version__,datasets.__file__)' | tee "$out/import_check.txt"
grep -q "sallm from $code/src/main/sallm/__init__.py" "$out/import_check.txt"
grep -q "datasets 4.4.2 $overlay/datasets/__init__.py" "$out/import_check.txt"

"$venv/bin/python" "$wrapper" --config-name "finetune/$config" "${args[@]}" --cfg job --resolve > "$out/resolved_config.stdout"
sed -n '/^finetune:/,$p' "$out/resolved_config.stdout" > "$out/resolved_config.yaml"
[[ -s "$out/resolved_config.yaml" ]]
printf '%s\n' "${args[@]}" > "$out/overrides.txt"
{
  echo "label=$label"; echo "config=finetune/$config"
  echo "code=$code (copy of /home/lmbanr001/masters/sallm/src as of 2026-09-25, mtimes preserved, evaluation/lm_eval_runner.py replaced by the import-compatible git 5d450ab version; SOURCE_MANIFEST sha256 $code_manifest_sha)"
  echo "launcher=accelerate launch --num_processes 1 --mixed_precision bf16 --dynamo_backend no (as scripts/launch_finetune.sh in July) + $wrapper (sha256 $wrapper_sha)"
  echo "venv=$venv (July venv; torch 2.9.1, transformers 4.57.3 vanilla modeling_xlstm, trl 0.26.2, peft 0.18.1, accelerate 1.12.0, xlstm 2.0.5, mlstm_kernels 2.0.2; package-set sha256 $venv_pkgset_sha)"
  echo "datasets_overlay=$overlay (datasets 4.4.2, the July version)"
  echo "base=$base_id resolved offline from $base_snap; pytorch_model.bin sha256 $base_weight_sha; config.json sha256 $base_config_sha"
  echo "tokenizer=$tokenizer (tokenizer.json sha256 446895905ea9b20c746317eefd0c6a3b097bcbbef71e8e44b0bf9772d664782a)"
  echo "original_run_dir=$orig"; echo "original_slurm_job=$orig_job"
  echo "changes_vs_original=load_best_model_at_end true->false; save_total_limit 1->null; early_stopping_patience 3->null (news, intent multi) or absent->null (intent mono); save_only_model true (was true for news, false for intent); report_to none (intent mono logged to wandb); hub.push_adapter false (hub was already disabled)"
  echo "unavoidable_deviations=HF offline: base model, tokenizer, MasakhaNews and InjongoIntent read from local caches (InjongoIntent JSONL via file:// from hub snapshot fe4be388 instead of https resolve/main); per-run TRITON_CACHE_DIR; GPU may be shared and may differ from the original A100 (news: A100-80GB, intent: A100-PCIE-40GB); source files edited after the July runs (training/factory.py, training/trainer.py, models/factory.py, models/registry.py, models/optional.py, evaluation/generation_metrics.py, evaluation/harness.py, fine_tune/run.py for news and intent multi) differ only in GDN-only, IterableDataset-only, save-guard, typing or INSTRUCTION-truncation code, see tmp/xlstm_recipe/NOTES.txt; evaluation/lm_eval_runner.py is imported but not called during fine-tuning"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"; echo "node=$(hostname)"; echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-}"
} > "$out/run_info.txt"

if [[ "$mode" != train ]]; then
  echo "RESOLVED $label -> $out/resolved_config.yaml"
  exit 0
fi

nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader >> "$out/run_info.txt"
echo "start=$(date -Is)" >> "$out/run_info.txt"
"$venv/bin/accelerate" launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  --main_process_port "$((31000 + RANDOM % 2000))" \
  "$wrapper" --config-name "finetune/$config" "${args[@]}"
echo "end=$(date -Is)" >> "$out/run_info.txt"
[[ -s "$out/final_adapter/adapter_config.json" ]]
(cd "$out/final_adapter" && find . -type f | LC_ALL=C sort | xargs sha256sum) > "$out/final_adapter.sha256"
echo "TRAIN_DONE $label"
