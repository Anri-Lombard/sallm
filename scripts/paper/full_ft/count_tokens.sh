#!/bin/bash
# Token accounting for runs that started before train_fft.py recorded it: build the exact training datasets
# (FFT_COUNT_ONLY=1 stops before training) and write runs/<arch>/tokens.json. Run inside an existing job (srun --overlap).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT=/scratch/lmbanr001/masters/sallm/results/equal_recipe_fullft_t2x_pilot_20260925
S=$ROOT/scripts
snapshot=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-targeted-recovery-20260916-v9
main_py=/home/lmbanr001/masters/sallm/.venv/bin/python
sealed_py=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
tmp=/dev/shm/fft_count_$$; mkdir -p $tmp; trap 'rm -rf $tmp' EXIT
for spec in "mzansilm llama /scratch/lmbanr001/masters/sallm/checkpoints/sallm-llama-125m/final_model $main_py" \
            "mamba2 mamba2 $ROOT/bases/mamba2_eosfix $main_py" \
            "xlstm xlstm /scratch/lmbanr001/masters/sallm_snapshots/full-matrix-retained-bindings-20260916-v1/bases/xlstm $sealed_py" \
            "gdn gated_deltanet /scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model $main_py"; do
  read -r arch key base py <<< "$spec"
  [[ -s "$ROOT/runs/$arch/tokens.json" ]] && continue
  env WANDB_MODE=disabled HYDRA_FULL_ERROR=1 SALLM_DISABLE_TASK_METRICS=1 SALLM_T2X_TRAIN_VALIDATION_ONLY=1 \
    SALLM_T2X_CACHE_DIR=$ROOT/assets/t2x_train_validation_only PYTHONPATH=$snapshot/src/main HF_HOME=/scratch/lmbanr001/hf \
    FLA_DISABLE_BACKEND_DISPATCH=1 SALLM_SKIP_MAMBA_KERNEL_CHECK=1 TOKENIZERS_PARALLELISM=false \
    FFT_COUNT_ONLY=1 FFT_RUN_INFO=$tmp/$arch.json \
    "$py" "$S/train_fft.py" --config-name finetune/llama_t2x_xho "finetune.model.architecture=$key" \
      "finetune.model.init_checkpoint=$base" "finetune.tokenizer.path=$base" finetune.peft.method=none \
      "finetune.training.output_dir=$tmp/out_$arch" "finetune.training.logging_dir=$tmp/logs_$arch" "hydra.run.dir=$tmp/hydra_$arch" \
      finetune.training.report_to=none finetune.hub.enabled=false finetune.dataset.max_seq_length=1024 \
      finetune.training.metric_for_best_model=null finetune.training.greater_is_better=null \
      finetune.training.per_device_train_batch_size=16 finetune.training.bf16=true > $tmp/$arch.log 2>&1 || { tail -20 $tmp/$arch.log; exit 1; }
  cp $tmp/$arch.json "$ROOT/runs/$arch/tokens.json"
  echo "$arch $(jq -c '{train_tokens_per_epoch,train_loss_tokens_per_epoch,train_max_tokens,val_tokens_per_epoch,val_max_tokens}' "$ROOT/runs/$arch/tokens.json")"
done
