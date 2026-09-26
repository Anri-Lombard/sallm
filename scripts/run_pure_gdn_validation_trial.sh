#!/bin/bash
# Validation-only adapter task-family trial. Defaults preserve the pure-GDN grid.
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/home/lmbanr001/masters/sallm
#SBATCH --mail-type=FAIL,END

set -euo pipefail

family="${1:?Usage: $0 <ner|pos|sib|intent|t2x|afrihg|general> <0|1|2>}"
trial="${2:?Usage: $0 <ner|pos|sib|intent|t2x|afrihg|general> <0|1|2>}"
learning_rates=(3.0e-05 8.0e-05 1.5e-04)
attempt_tag="${SALLM_ATTEMPT_TAG:-}"

if [[ ! "$trial" =~ ^[0-2]$ ]]; then
  echo "ERROR: trial must be 0, 1, or 2." >&2
  exit 1
fi
if [[ ! "$attempt_tag" =~ ^[-_a-zA-Z0-9]*$ ]]; then
  echo "ERROR: SALLM_ATTEMPT_TAG contains unsupported characters." >&2
  exit 1
fi

eval_template_args=()
disable_task_metrics=0
case "$family" in
  news)
    config=gdn_pure_news_all_hpo_r1
    metric=eval_classification/all_macro_f1
    greater=true
    epochs=10
    max_length=1024
    gradient_accumulation=2
    ;;
  ner)
    config=llama_ner_all
    metric=eval_all_f1
    greater=true
    epochs=15
    max_length=2048
    gradient_accumulation=2
    eval_template_args+=(++dataset.eval_template_choice=ALL)
    ;;
  pos)
    config=gdn_pure_pos_all_hpo_r2
    metric=eval_all_token_accuracy
    greater=true
    epochs=15
    max_length=2048
    gradient_accumulation=2
    eval_template_args+=(++dataset.eval_template_choice=ALL)
    ;;
  sib)
    config=llama_sib_all
    metric=eval_classification/all_macro_f1
    greater=true
    epochs=10
    max_length=1024
    gradient_accumulation=2
    ;;
  intent)
    config=llama_injongointent_all
    metric=eval_classification/all_macro_f1
    greater=true
    epochs=10
    max_length=2048
    gradient_accumulation=2
    ;;
  t2x)
    config=llama_t2x_xho
    metric=eval_all_chrf
    greater=true
    epochs=4
    max_length=1024
    gradient_accumulation=2
    ;;
  afrihg)
    config=llama_afrihg_all
    metric=eval_all_chrf
    greater=true
    epochs=5
    max_length=1024
    gradient_accumulation=2
    ;;
  general)
    config=llama_sa_general_tokenbalanced_r1
    metric=eval_loss
    greater=false
    epochs=5
    max_length=2048
    gradient_accumulation=4
    disable_task_metrics=1
    ;;
  *)
    echo "ERROR: unsupported family '$family'." >&2
    exit 1
    ;;
esac

repo="${SALLM_REPO_DIR:-$HOME/masters/sallm}"
scratch="${SCRATCH:-/scratch/lmbanr001}"
config="${SALLM_HPO_CONFIG:-$config}"
model_architecture="${SALLM_HPO_MODEL_ARCHITECTURE:-gated_deltanet}"
model="${SALLM_HPO_MODEL:-${PURE_GDN_MODEL:-$scratch/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model}}"
tokenizer="${SALLM_HPO_TOKENIZER:-$model}"
lr="${SALLM_HPO_LEARNING_RATE:-${learning_rates[$trial]}}"
lora_rank="${SALLM_HPO_LORA_RANK:-16}"
lora_alpha="${SALLM_HPO_LORA_ALPHA:-32}"
lora_dropout="${SALLM_HPO_LORA_DROPOUT:-0.05}"
warmup_ratio="${SALLM_HPO_WARMUP_RATIO:-0.03}"
seed="${SALLM_HPO_SEED:-42}"
data_seed="${SALLM_HPO_DATA_SEED:-$seed}"
train_batch_size="${SALLM_HPO_TRAIN_BATCH_SIZE:-4}"
eval_batch_size="${SALLM_HPO_EVAL_BATCH_SIZE:-4}"
gradient_accumulation="${SALLM_HPO_GRADIENT_ACCUMULATION_STEPS:-$gradient_accumulation}"
max_length="${SALLM_HPO_MAX_LENGTH:-$max_length}"
gradient_checkpointing="${SALLM_HPO_GRADIENT_CHECKPOINTING:-false}"
model_id="${SALLM_HPO_MODEL_ID:-pure_gdn}"
stage="${SALLM_HPO_STAGE:-stage_a}"
candidate="${SALLM_HPO_CANDIDATE:-a${trial}}"
run_id="${SALLM_HPO_RUN_ID:-pure-gdn-${family}-validation-lr${trial}-r2${attempt_tag}}"
output_dir="${SALLM_HPO_OUTPUT_DIR:-$scratch/masters/sallm/checkpoints/pure_gdn_adapter_hpo_r2/$family/lr_${trial}${attempt_tag}}"
logging_dir="${SALLM_HPO_LOGGING_DIR:-$scratch/masters/sallm/logs/pure_gdn_adapter_hpo_r2/$family/lr_${trial}${attempt_tag}}"
resume_checkpoint="${SALLM_HPO_RESUME_FROM_CHECKPOINT:-}"

for label in "$model_id" "$stage" "$candidate" "$run_id"; do
  if [[ ! "$label" =~ ^[-_.a-zA-Z0-9]+$ ]]; then
    echo "ERROR: unsafe HPO label '$label'." >&2
    exit 1
  fi
done
for number in "$lr" "$lora_dropout" "$warmup_ratio"; do
  if [[ ! "$number" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "ERROR: invalid numeric HPO value '$number'." >&2
    exit 1
  fi
done
for integer in "$lora_rank" "$lora_alpha" "$seed" "$data_seed" "$train_batch_size" "$eval_batch_size" "$gradient_accumulation" "$max_length"; do
  if [[ ! "$integer" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid integer HPO value '$integer'." >&2
    exit 1
  fi
done
if [[ "$gradient_checkpointing" != true && "$gradient_checkpointing" != false ]]; then
  echo "ERROR: gradient checkpointing must be true or false." >&2
  exit 1
fi
if (( lora_alpha != 2 * lora_rank )); then
  echo "ERROR: LoRA alpha must equal twice the rank." >&2
  exit 1
fi

resume_args=()
if [[ -n "$resume_checkpoint" ]]; then
  resume_checkpoint="$(realpath "$resume_checkpoint")"
  output_dir="$(realpath "$output_dir")"
  if [[ "$(dirname "$resume_checkpoint")" != "$output_dir" || ! "$(basename "$resume_checkpoint")" =~ ^checkpoint-[0-9]+$ ]]; then
    echo "ERROR: resume checkpoint must be a numbered checkpoint inside the trial output directory." >&2
    exit 1
  fi
  for required in optimizer.pt scheduler.pt rng_state.pth trainer_state.json; do
    [[ -f "$resume_checkpoint/$required" ]] || {
      echo "ERROR: incomplete resume checkpoint; missing $required." >&2
      exit 1
    }
  done
  if [[ ! -f "$resume_checkpoint/adapter_model.bin" && ! -f "$resume_checkpoint/adapter_model.safetensors" ]]; then
    echo "ERROR: incomplete resume checkpoint; missing adapter weights." >&2
    exit 1
  fi
  resume_args+=("+training.resume_from_checkpoint=$resume_checkpoint")
fi

target_module_args=()
target_modules="${SALLM_HPO_TARGET_MODULES:-}"
if [[ -z "$target_modules" && "$model_architecture" == gated_deltanet ]]; then
  target_modules='[q_proj,k_proj,v_proj,a_proj,b_proj,g_proj,o_proj,gate_proj,up_proj,down_proj]'
fi

if [[ -n "$target_modules" ]]; then
  target_module_args+=("++peft.kwargs.target_modules=$target_modules")
fi

if [[ -f "$output_dir/final_adapter/adapter_model.safetensors" ]]; then
  echo "Validation trial already complete: $output_dir/final_adapter"
  exit 0
fi

export FLA_DISABLE_BACKEND_DISPATCH=1
export SALLM_SKIP_MAMBA_KERNEL_CHECK=1
export SALLM_DISABLE_TASK_METRICS="$disable_task_metrics"
export SALLM_JOB_NAME="$run_id"
unset SALLM_GENERAL_SELECTION_PROTOCOL
if [[ "$family" == general ]]; then
  export SALLM_GENERAL_SELECTION_PROTOCOL=equal_family_assistant_token_nll_v1
fi

if [[ "${SALLM_DRY_RUN:-0}" == 1 ]]; then
  printf 'model=%s family=%s stage=%s candidate=%s seed=%s lr=%s rank=%s alpha=%s dropout=%s warmup=%s config=%s architecture=%s model_path=%s tokenizer=%s targets=%s metric=%s greater=%s epochs=%s train_batch=%s eval_batch=%s accumulation=%s effective_batch=%s max_length=%s gradient_checkpointing=%s output=%s task_metrics=%s resume=%s\n' \
    "$model_id" "$family" "$stage" "$candidate" "$seed" "$lr" \
    "$lora_rank" "$lora_alpha" "$lora_dropout" "$warmup_ratio" \
    "$config" "$model_architecture" "$model" "$tokenizer" "$target_modules" \
    "$metric" "$greater" "$epochs" "$train_batch_size" "$eval_batch_size" \
    "$gradient_accumulation" "$((train_batch_size * gradient_accumulation))" \
    "$max_length" "$gradient_checkpointing" "$output_dir" \
    "$((1 - disable_task_metrics))" "$resume_checkpoint"
  exit 0
fi

manifest_suffix=""
[[ -n "$resume_checkpoint" ]] && manifest_suffix=".resume-${SLURM_JOB_ID:-manual}"
manifest_path="$output_dir/execution_manifest${manifest_suffix}.json"
mkdir -p "$output_dir"
manifest_python="${SALLM_RUNTIME_REPO:-$repo}/.venv/bin/python"
if [[ ! -x "$manifest_python" ]]; then
  manifest_python=python3
fi
artifact_args=()
[[ -d "$model" ]] && artifact_args+=(--artifact-root "$model")
"$manifest_python" "$repo/scripts/create_execution_manifest.py" \
  --repo-root "$repo" \
  --output "$manifest_path" \
  --entrypoint "${SALLM_HPO_ENTRYPOINT:-scripts/run_pure_gdn_validation_trial.sh}" \
  --command "model=$model_id family=$family stage=$stage candidate=$candidate seed=$seed lr=$lr rank=$lora_rank alpha=$lora_alpha dropout=$lora_dropout warmup=$warmup_ratio run_id=$run_id resume=$resume_checkpoint" \
  "${artifact_args[@]}"
export SALLM_EXECUTION_MANIFEST="$manifest_path"

if [[ -n "${SALLM_HPO_REGISTRY:-}" ]]; then
  "$manifest_python" "$repo/scripts/hpo_protocol.py" write-trial \
    --registry "$SALLM_HPO_REGISTRY" \
    --candidate "$candidate" \
    --seed "$seed" \
    --model "$model_id" \
    --family "$family" \
    --stage "$stage" \
    --manifest "$manifest_path" \
    --output "$output_dir/hpo_trial${manifest_suffix}.json"
fi

exec bash "$repo/scripts/launch_finetune.sh" "finetune/$config" \
  "model.architecture=$model_architecture" \
  "model.init_checkpoint=$model" \
  "tokenizer.path=$tokenizer" \
  peft.method=lora \
  "peft.kwargs.r=$lora_rank" \
  "peft.kwargs.lora_alpha=$lora_alpha" \
  "peft.kwargs.lora_dropout=$lora_dropout" \
  "${target_module_args[@]}" \
  "dataset.max_seq_length=$max_length" \
  "training.output_dir=$output_dir" \
  "training.logging_dir=$logging_dir" \
  "training.run_name=$run_id" \
  "training.learning_rate=$lr" \
  ++training.weight_decay=0.01 \
  ++training.adam_beta1=0.9 \
  ++training.adam_beta2=0.95 \
  ++training.max_grad_norm=1.0 \
  ++training.lr_scheduler_type=cosine \
  "++training.warmup_ratio=$warmup_ratio" \
  "training.per_device_train_batch_size=$train_batch_size" \
  "training.per_device_eval_batch_size=$eval_batch_size" \
  "++training.gradient_accumulation_steps=$gradient_accumulation" \
  "training.num_train_epochs=$epochs" \
  training.bf16=true \
  "++training.gradient_checkpointing=$gradient_checkpointing" \
  ++training.label_smoothing_factor=0.05 \
  training.eval_strategy=epoch \
  training.save_strategy=epoch \
  ++training.save_total_limit=1 \
  ++training.load_best_model_at_end=true \
  "++training.metric_for_best_model=$metric" \
  "++training.greater_is_better=$greater" \
  ++training.early_stopping_patience=2 \
  ++training.early_stopping_threshold=0.001 \
  "++training.seed=$seed" \
  "++training.data_seed=$data_seed" \
  "${resume_args[@]}" \
  hub.enabled=false \
  hub.push_adapter=false \
  hub.push_merged=false \
  "wandb.name=$run_id" \
  "${eval_template_args[@]}"
