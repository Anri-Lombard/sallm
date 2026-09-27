#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --qos=nlpgroup
#SBATCH --gres=gpu:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1

label="${LABEL:?LABEL is required}"
root=/scratch/lmbanr001/masters/sallm/results/monomulti_retrain_20260924
venv=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv
runtime_manifest=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/runtime_manifest.json
[[ "$(sha256sum "$runtime_manifest" | cut -d' ' -f1)" == 9f30eeb73d415bd8be3535b0f74689c5d53d04f622e0e3ddea04a8c5a2df24ab ]]
python="$venv/bin/python"
accelerate="$venv/bin/accelerate"
tokenizer=/home/lmbanr001/masters/sallm/tokenizer/sallm_bpe_tokenizer

output="$root/train/$label"
[[ ! -e "$output" ]] || { echo "ERROR: output exists: $output" >&2; exit 5; }
mkdir -p "$output"

common=(
  "finetune.training.output_dir=$output"
  "finetune.training.logging_dir=$output/logs"
  "finetune.training.run_name=$label-monomulti-retrain-20260924"
  ++finetune.training.seed=42
  finetune.training.report_to=none
  finetune.hub.enabled=false
  finetune.hub.push_adapter=false
  finetune.hub.push_merged=false
  "hydra.run.dir=$output/hydra"
)

case "$label" in
  xlstm_ner_multi|xlstm_sib_multi)
    # v15 snapshot: the embedding-fix snapshot passes the MasakhaNER revision as the config name.
    # trainable_token_indices=null reproduces its embedding fix (no trainable-token wrapper when embeddings are a LoRA target).
    snapshot=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-targeted-recovery-20260919-v15
    manifest_sha=928711a05cadc3860ef57df737a02057fc17d013b9758e44c22f7f1bba487fad
    base=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-retained-bindings-20260916-v1/bases/xlstm
    base_weight=pytorch_model.bin
    base_weight_sha=4e3e37db9a43089e1683a56c4940bc05e9e446ae7da82b5dc5d8621beda09674
    launcher=accelerate
    if [[ "$label" == xlstm_ner_multi ]]; then
      config=xlstm_ner_all
      select=(
        finetune.training.save_strategy=epoch
        ++finetune.training.save_only_model=true
        ++finetune.training.load_best_model_at_end=true
        ++finetune.training.metric_for_best_model=eval_all_f1
        ++finetune.training.greater_is_better=true
      )
    else
      config=xlstm_sib_all
      select=(
        finetune.training.save_strategy=epoch
        ++finetune.training.save_only_model=true
        ++finetune.training.load_best_model_at_end=false
      )
    fi
    args=(
      --config-name "finetune/$config"
      "finetune.model.init_checkpoint=$base"
      "finetune.tokenizer.path=$tokenizer"
      "${common[@]}"
      ++finetune.training.data_seed=42
      ++finetune.training.pad_to_multiple_of=64
      ++finetune.peft.kwargs.modules_to_save=null
      ++finetune.peft.kwargs.trainable_token_indices=null
      "${select[@]}"
    )
    ;;
  mzansilm_pos_mono_tsn)
    snapshot=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-targeted-recovery-20260916-v9
    manifest_sha=10b40f2bd5e93f757d7bff07a93ee9c2e724838a33e72224a65d8b6056afb4c2
    base=/scratch/lmbanr001/masters/sallm/checkpoints/sallm-llama-125m/final_model
    base_weight=pytorch_model.bin
    base_weight_sha=7388b67c8fe73a1bb8f80a97b8b756de4a110e914ac8e1dfcaf0d06697a61273
    launcher=runner
    args=(
      --config-name finetune/llama_pos_tsn
      "finetune.model.init_checkpoint=$base"
      "finetune.tokenizer.path=$base"
      "${common[@]}"
      "++finetune.peft.kwargs.target_modules=[q_proj,v_proj]"
    )
    ;;
  mamba2_ner_mono_tsn|mamba2_ner_mono_xho|mamba2_ner_mono_zul|mamba2_pos_multi)
    snapshot=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-targeted-recovery-20260919-v15
    manifest_sha=928711a05cadc3860ef57df737a02057fc17d013b9758e44c22f7f1bba487fad
    base=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-retained-bindings-20260916-v1/bases/mamba2
    base_weight=pytorch_model.bin
    base_weight_sha=7eaa6b8a1c24ce1a0638c17efb84cdaa44eab5ec167f76881742e2f89414abde
    launcher=runner
    if [[ "$label" == mamba2_pos_multi ]]; then
      config=mamba_pos_all; bs=2; ga=32
    else
      # effective batch 64 (Mamba Multi NER setting) as 8 x 8 instead of 2 x 32 for throughput
      config="mamba_ner_${label##*_}"; bs=8; ga=8
    fi
    args=(
      --config-name "finetune/$config"
      "finetune.model.init_checkpoint=$base"
      "finetune.tokenizer.path=$base"
      "${common[@]}"
      "finetune.peft.kwargs.target_modules=[in_proj,x_proj]"
      "finetune.training.per_device_train_batch_size=$bs"
      finetune.training.per_device_eval_batch_size=1
      "finetune.training.gradient_accumulation_steps=$ga"
      finetune.training.save_strategy=epoch
      ++finetune.training.save_total_limit=null
      ++finetune.training.save_only_model=true
    )
    export SALLM_MAMBA_VALIDATION_LABEL_MICROBATCH=17
    ;;
  *) echo "ERROR: unknown label $label" >&2; exit 4 ;;
esac

[[ "$(sha256sum "$snapshot/SNAPSHOT_MANIFEST.sha256" | cut -d' ' -f1)" == "$manifest_sha" ]]
(cd "$snapshot" && sha256sum -c SNAPSHOT_MANIFEST.sha256 --quiet)
[[ "$(sha256sum "$base/$base_weight" | cut -d' ' -f1)" == "$base_weight_sha" ]]
sha256sum "$tokenizer/tokenizer.json" | grep -q '^446895905ea9b20c746317eefd0c6a3b097bcbbef71e8e44b0bf9772d664782a '

export HF_HOME=/scratch/lmbanr001/hf
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=disabled
export WANDB_SILENT=true
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export SALLM_T2X_TRAIN_VALIDATION_ONLY=1
export SALLM_T2X_CACHE_DIR="$root/assets/t2x_train_validation_only"
export PYTHONPATH="$snapshot/src/main"
export TRITON_CACHE_DIR="$output/triton-cache"

runner="$snapshot/.audit/run_train_validation_only_20260914.py"
if [[ "$launcher" == accelerate ]]; then
  "$python" -m sallm.main "${args[@]}" --cfg job --resolve > "$output/resolved_config.stdout"
else
  "$python" "$runner" "${args[@]}" --cfg job --resolve > "$output/resolved_config.stdout"
fi
sed -n '/^finetune:/,$p' "$output/resolved_config.stdout" > "$output/resolved_config.yaml"
[[ -s "$output/resolved_config.yaml" ]]
if grep -Eiq '(^|[[:space:]])(test|test_split)[[:space:]]*:' "$output/resolved_config.yaml"; then
  echo "ERROR: held-out field in resolved config" >&2; exit 6
fi
printf '%s\n' "${args[@]}" > "$output/overrides.txt"
{
  echo "label=$label"; echo "snapshot=$snapshot"; echo "snapshot_manifest_sha256=$manifest_sha"
  echo "base=$base"; echo "base_weight_sha256=$base_weight_sha"; echo "launcher=$launcher"
  echo "xlstm_wrapper_sha256=$(sha256sum "$root/jobs/run_xlstm_train.py" | cut -d' ' -f1)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"; echo "node=$(hostname)"
  nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
  echo "start=$(date -Is)"
} > "$output/run_info.txt"

if [[ "$launcher" == accelerate ]]; then
  "$accelerate" launch --num_processes 1 --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no --main_process_port "$((30300 + RANDOM % 500))" \
    "$root/jobs/run_xlstm_train.py" "${args[@]}"
else
  "$python" "$runner" "${args[@]}"
fi
echo "end=$(date -Is)" >> "$output/run_info.txt"
[[ -s "$output/final_adapter/adapter_config.json" ]]
(cd "$output/final_adapter" && find . -type f ! -path './.cache/*' | sort | xargs sha256sum) > "$output/final_adapter.sha256"
echo "final_adapter_tree_sha256=$(sha256sum "$output/final_adapter.sha256" | cut -d' ' -f1)" >> "$output/run_info.txt"
echo DONE
