# Shared by mamba2_train.sh and mzansilm_train.sh (sourced, not executed).
# The caller sets: label code venv launcher config args[] env_extra[] base base_file base_sha
#                  tok tok_sha original changes deviations recipe_note
# DRY_RUN=1 resolves the config into $root/tmp/mm_recipe/dryrun/<label> and exits.
# DRY_RUN=1 DATA_CHECK=1 also builds model, datasets and trainer on CPU and exits before training.

root=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925
jobs="$root/jobs"
entry="$jobs/sallm_entry.py"
live_venv=/home/lmbanr001/masters/sallm/.venv
std_venv=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv
v15=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-targeted-recovery-20260919-v15
v15_sha=928711a05cadc3860ef57df737a02057fc17d013b9758e44c22f7f1bba487fad
code_jul="$jobs/code/sallm-live-20260925"
code_feb="$jobs/code/sallm-cf0ed06"
hf_mamba_blob=/scratch/lmbanr001/hf/hub/models--anrilombard--sallm-mamba-125m/snapshots/0c57d7bdcd47209894bdc5098e62658c4f05fa59
injongo_snapshot=/scratch/lmbanr001/hf/hub/datasets--masakhane--InjongoIntent/snapshots/fe4be3882a1614161dfe231ec793197bb74f4b44
sallm_tok=/home/lmbanr001/masters/sallm/tokenizer/sallm_bpe_tokenizer
sallm_tok_sha=446895905ea9b20c746317eefd0c6a3b097bcbbef71e8e44b0bf9772d664782a
mamba_sha=7eaa6b8a1c24ce1a0638c17efb84cdaa44eab5ec167f76881742e2f89414abde
llama_base=/scratch/lmbanr001/masters/sallm/checkpoints/sallm-llama-125m/final_model
llama_sha=7388b67c8fe73a1bb8f80a97b8b756de4a110e914ac8e1dfcaf0d06697a61273
keep_all=(
  finetune.training.save_strategy=epoch
  ++finetune.training.save_total_limit=null
  ++finetune.training.save_only_model=true
  ++finetune.training.load_best_model_at_end=false
  ++finetune.training.early_stopping_patience=null
)

sha_is() { [[ "$(sha256sum "$1" | cut -d' ' -f1)" == "$2" ]] || { echo "ERROR: sha256 mismatch for $1" >&2; exit 6; }; }

verify_code() {
  case "$code" in
    "$v15"|/scratch/lmbanr001/masters/sallm_snapshots/*)
      [[ "$code" != "$v15" ]] || sha_is "$code/SNAPSHOT_MANIFEST.sha256" "$v15_sha"
      (cd "$code" && sha256sum -c SNAPSHOT_MANIFEST.sha256 --quiet) ;;
    *) (cd "$code" && sha256sum -c MANIFEST.sha256 --quiet) ;;
  esac
}

run_label() {
  local out mode=train
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    mode=dry
    out="$root/tmp/mm_recipe/dryrun/$label"
    rm -rf "$out"
  else
    out="$root/train/$label"
    [[ ! -e "$out/final_adapter" ]] || { echo "already trained: $out"; exit 0; }
    [[ ! -e "$out" ]] || { echo "ERROR: partial output exists: $out" >&2; exit 5; }
  fi
  mkdir -p "$out"

  verify_code
  sha_is "$base_file" "$base_sha"
  sha_is "$tok/tokenizer.json" "$tok_sha"

  export SCRATCH=/scratch/lmbanr001
  export HF_HOME=/scratch/lmbanr001/hf HF_DATASETS_CACHE=/scratch/lmbanr001/hf/datasets
  export HF_HUB_CACHE=/scratch/lmbanr001/hf/hub HF_METRICS_CACHE=/scratch/lmbanr001/hf/metrics
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1
  export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
  export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
  export SALLM_SKIP_MAMBA_KERNEL_CHECK=1
  unset SALLM_GENERAL_SELECTION_PROTOCOL SALLM_EXECUTION_MANIFEST SALLM_DISABLE_TASK_METRICS HF_TOKEN
  export PYTHONPATH="$code/src/main"
  export TRITON_CACHE_DIR="$out/triton-cache"
  local kv
  for kv in "${env_extra[@]}"; do export "${kv?}"; done

  local python="$venv/bin/python"
  local -a full=(
    --config-name "finetune/$config"
    "${args[@]}"
    "finetune.training.output_dir=$out"
    "finetune.training.logging_dir=$out/logs"
    "hydra.run.dir=$out/hydra"
  )
  cd "$out"
  "$python" -c 'import sallm, sys; print(sallm.__file__)' > "$out/import_check.txt"
  grep -q "^$code/src/main/sallm/" "$out/import_check.txt" || { echo "ERROR: sallm not imported from $code" >&2; exit 7; }
  "$python" "$entry" "${full[@]}" --cfg job --resolve > "$out/resolved_config.stdout"
  sed -n '/^finetune:/,$p' "$out/resolved_config.stdout" > "$out/resolved_config.yaml"
  [[ -s "$out/resolved_config.yaml" ]]
  printf '%s\n' "${full[@]}" > "$out/overrides.txt"
  {
    echo "label=$label"
    echo "config=finetune/$config"
    echo "code=$code"
    if [[ -f "$code/SOURCE.txt" ]]; then echo "code_source=$(cat "$code/SOURCE.txt")"; fi
    if [[ -f "$code/MANIFEST.sha256" ]]; then echo "code_manifest_sha256=$(sha256sum "$code/MANIFEST.sha256" | cut -d' ' -f1)"; fi
    if [[ -f "$code/SNAPSHOT_MANIFEST.sha256" ]]; then echo "code_manifest_sha256=$(sha256sum "$code/SNAPSHOT_MANIFEST.sha256" | cut -d' ' -f1)"; fi
    echo "venv=$venv"
    echo "packages=$("$python" -c 'import importlib.metadata as m; print(" ".join(f"{p}={m.version(p)}" for p in ("torch","transformers","trl","peft","accelerate","datasets")))')"
    echo "entry=$entry sha256=$(sha256sum "$entry" | cut -d' ' -f1)"
    echo "launcher=$launcher"
    echo "env_extra=${env_extra[*]}"
    echo "base=$base"
    echo "base_weight=$base_file sha256=$base_sha"
    echo "tokenizer=$tok tokenizer_json_sha256=$tok_sha"
    echo "original=$original"
    echo "recipe=$recipe_note"
    echo "changes_vs_original=$changes"
    echo "deviations=$deviations"
    echo "mode=$mode slurm_job_id=${SLURM_JOB_ID:-} node=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-}"
    if [[ "$mode" == train ]]; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1; fi
    echo "start=$(date -Is)"
  } > "$out/run_info.txt"

  if [[ "$mode" == dry ]]; then
    if [[ "${DATA_CHECK:-0}" == 1 ]]; then
      CUDA_VISIBLE_DEVICES="" SALLM_DATA_CHECK=1 SALLM_DATA_CHECK_OUT="$out/data_check.json" \
        "$python" "$entry" "${full[@]}" > "$out/data_check.log" 2>&1
      [[ -s "$out/data_check.json" ]]
    fi
    echo "DRY_RUN_DONE $label $out"
    return 0
  fi

  if [[ "$launcher" == accelerate ]]; then
    "$venv/bin/accelerate" launch --num_processes 1 --num_machines 1 --mixed_precision bf16 \
      --dynamo_backend no --main_process_port "$((29500 + RANDOM % 900))" "$entry" "${full[@]}"
  else
    "$python" "$entry" "${full[@]}"
  fi
  echo "end=$(date -Is)" >> "$out/run_info.txt"
  [[ -s "$out/final_adapter/adapter_config.json" ]]
  ls -d "$out"/checkpoint-* > "$out/checkpoints.txt"
  (cd "$out/final_adapter" && find . -type f ! -path './.cache/*' | sort | xargs sha256sum) > "$out/final_adapter.sha256"
  echo "final_adapter_tree_sha256=$(sha256sum "$out/final_adapter.sha256" | cut -d' ' -f1)" >> "$out/run_info.txt"
  echo "checkpoints=$(wc -l < "$out/checkpoints.txt")" >> "$out/run_info.txt"
  echo "TRAIN_DONE $label"
}
