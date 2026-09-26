#!/bin/bash
# Retrain a GDN Mono/Multi task adapter with its original recipe (run_pure_gdn_validation_trial.sh /
# run_validation_hpo_trial.sh in snapshot pure-gdn-t2x-official-test-20260901-v2-daf6d8cf), changing only
# checkpoint handling: save every epoch, no early stopping, no keep-best.
set -euo pipefail
umask 022
label="${1:?label}"
root=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925
snap=/home/lmbanr001/masters/sallm_snapshots/pure-gdn-t2x-official-test-20260901-v2-daf6d8cf
venv=/home/lmbanr001/masters/sallm/.venv
model=/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model
out="$root/train/$label"
[[ ! -e "$out/final_adapter" ]] || { echo "already trained: $out"; exit 0; }
[[ ! -e "$out" ]] || { echo "ERROR: partial output exists: $out" >&2; exit 5; }

IFS=_ read -r _ family regime lang <<< "$label"
case "$family" in
  news) metric=eval_classification/all_macro_f1; maxlen=1024; lr=3e-05 ;;
  sib) metric=eval_classification/all_macro_f1; maxlen=1024; lr=3e-05 ;;
  intent) metric=eval_classification/all_macro_f1; maxlen=2048; lr=8e-05 ;;
  *) echo "bad family" >&2; exit 4 ;;
esac
if [[ "$regime" == mono ]]; then
  case "$family" in news) config="llama_news_$lang";; sib) config="llama_sib_$lang";; intent) config="llama_injongointent_$lang";; esac
  [[ "$family" == intent ]] && lr=0.00008 || lr=0.00003
  run_id="gdn-mono-$family-$lang-s42-reselect"
else
  case "$family" in news) config=gdn_pure_news_all_hpo_r1;; sib) config=llama_sib_all;; intent) config=llama_injongointent_all;; esac
  run_id="hpo-pure_gdn-$family-multi-reselect"
fi

[[ "$(sha256sum "$snap/deployment_manifest.json" | cut -d' ' -f1)" == f5ec949feef494075e67fe94e55942e9e1a72d943dd24230921b800f937d7c7a ]]
[[ "$(sha256sum "$model/pytorch_model.bin" | cut -d' ' -f1)" == 37bcc3d080dafcc5d8812821b969a46ef3208b84341bc32ae2c54dd94a9d8108 ]]

export SCRATCH=/scratch/lmbanr001 PURE_GDN_MODEL="$model"
export HF_HOME=/scratch/lmbanr001/hf HF_DATASETS_CACHE=/scratch/lmbanr001/hf/datasets HF_METRICS_CACHE=/scratch/lmbanr001/hf/metrics
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export FLA_DISABLE_BACKEND_DISPATCH=1 SALLM_SKIP_MAMBA_KERNEL_CHECK=1 SALLM_DISABLE_TASK_METRICS=0
unset SALLM_GENERAL_SELECTION_PROTOCOL SALLM_EXECUTION_MANIFEST
export SALLM_JOB_NAME="$run_id" TOKENIZERS_PARALLELISM=true HYDRA_FULL_ERROR=1 TORCH_DISTRIBUTED_TIMEOUT=7200
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export PYTHONPATH="$snap/src/main:$SCRATCH/.local/lib/python3.12/site-packages"
export TRITON_CACHE_DIR="$root/triton/$label"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$out" "$TRITON_CACHE_DIR"

args=(
  finetune.model.architecture=gated_deltanet
  "finetune.model.init_checkpoint=$model"
  "finetune.tokenizer.path=$model"
  finetune.peft.method=lora finetune.peft.kwargs.r=16 finetune.peft.kwargs.lora_alpha=32 finetune.peft.kwargs.lora_dropout=0.05
  "++finetune.peft.kwargs.target_modules=[q_proj,k_proj,v_proj,a_proj,b_proj,g_proj,o_proj,gate_proj,up_proj,down_proj]"
  "finetune.dataset.max_seq_length=$maxlen"
  "finetune.training.output_dir=$out"
  "finetune.training.logging_dir=$out/logs"
  "finetune.training.run_name=$run_id"
  "finetune.training.learning_rate=$lr"
  ++finetune.training.weight_decay=0.01 ++finetune.training.adam_beta1=0.9 ++finetune.training.adam_beta2=0.95
  ++finetune.training.max_grad_norm=1.0 ++finetune.training.lr_scheduler_type=cosine ++finetune.training.warmup_ratio=0.03
  finetune.training.per_device_train_batch_size=4 finetune.training.per_device_eval_batch_size=4
  ++finetune.training.gradient_accumulation_steps=2
  finetune.training.num_train_epochs=10
  finetune.training.bf16=true ++finetune.training.gradient_checkpointing=false
  ++finetune.training.label_smoothing_factor=0.05
  finetune.training.eval_strategy=epoch finetune.training.save_strategy=epoch
  ++finetune.training.save_total_limit=null
  ++finetune.training.save_only_model=true
  ++finetune.training.load_best_model_at_end=false
  "++finetune.training.metric_for_best_model=$metric" ++finetune.training.greater_is_better=true
  ++finetune.training.early_stopping_patience=null
  ++finetune.training.seed=42 ++finetune.training.data_seed=42
  finetune.hub.enabled=false finetune.hub.push_adapter=false finetune.hub.push_merged=false
  "finetune.wandb.name=$run_id"
  ++finetune.training.report_to=none
  "hydra.run.dir=$out/hydra"
)
cd /home/lmbanr001/masters/sallm
"$venv/bin/python" -c 'import sallm,sys;print("sallm from",sallm.__file__)' | tee "$out/import_check.txt"
grep -q "$snap/src/main" "$out/import_check.txt"
"$venv/bin/python" -m sallm.main --config-name "finetune/$config" "${args[@]}" --cfg job --resolve > "$out/resolved_config.yaml"
printf '%s\n' "${args[@]}" > "$out/overrides.txt"
{ echo "label=$label"; echo "config=finetune/$config"; echo "snapshot=$snap"; echo "venv=$venv"; echo "base=$model"
  echo "changes_vs_original=save_total_limit 1->null; load_best_model_at_end true->false; early_stopping_patience 2->null; save_only_model true (no optimizer state in checkpoints; training unaffected); report_to none; offline HF"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"; echo "node=$(hostname)"; echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1; echo "start=$(date -Is)"; } > "$out/run_info.txt"
"$venv/bin/accelerate" launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  --main_process_port "$((29500 + RANDOM % 900))" -m sallm.main --config-name "finetune/$config" "${args[@]}"
echo "end=$(date -Is)" >> "$out/run_info.txt"
[[ -s "$out/final_adapter/adapter_config.json" ]]
echo "TRAIN_DONE $label"
