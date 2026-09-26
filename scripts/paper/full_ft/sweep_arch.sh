#!/bin/bash
#SBATCH --job-name=fft-t2x
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/equal_recipe_fullft_t2x_pilot_20260925/logs/%x-%j.out
# usage: sbatch --job-name=fft-<arch> --export=ALL,ARCH=<mzansilm|mamba2|xlstm|gdn> sweep_arch.sbatch
# Equal-recipe full fine-tuning LR sweep for one architecture on Monolingual T2X (isiXhosa); see recipe.md.
set -euo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1
arch="${ARCH:?ARCH is required}"

ROOT=/scratch/lmbanr001/masters/sallm/results/equal_recipe_fullft_t2x_pilot_20260925
S="$ROOT/scripts"
snapshot=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-targeted-recovery-20260916-v9
snapshot_manifest=10b40f2bd5e93f757d7bff07a93ee9c2e724838a33e72224a65d8b6056afb4c2
main_py=/home/lmbanr001/masters/sallm/.venv/bin/python
sealed_py=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
gen=/scratch/lmbanr001/masters/sallm_snapshots/generation-protocol-v3-greedy-20260924
v4=/scratch/lmbanr001/masters/sallm_snapshots/generation-protocol-v4-xlstm-bs1-20260925
eval_source=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
retained=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-retained-bindings-20260916-v1/bases
# core grid 1e-5 3e-5 1e-4; 3e-4 / 3e-6 are the edge extensions (enter selection only if the rule fires)
read -r -a LRS <<< "${LRS:-1e-5 3e-5 1e-4 3e-4 3e-6}"
export SEED="${SEED:-42}" BS="${BS:-16}"  # BS = per-device (= effective, 1 GPU, no accumulation) train batch
prefix=""
[[ "$SEED" != 42 ]] && prefix+="s${SEED}_"
[[ "$BS" != 16 ]] && prefix+="b${BS}_"
tag() { echo "${prefix}lr$1"; }

extra=()
train_env=()
case "$arch" in
  mzansilm) key=llama; base=/scratch/lmbanr001/masters/sallm/checkpoints/sallm-llama-125m/final_model
            base_tree=b75722806e9d41f903d431406c9ecb44b2ad1f6bded778d4e535c9bf692572a9
            py=$main_py; eval_runner="$S/run_generation_direct_fullft.py" ;;
  mamba2)   key=mamba2; src_base=$retained/mamba2; base=$ROOT/bases/mamba2_eosfix
            base_tree=5db6860f96ebc7ec442417e7fa696f5ad404c1a6bd5c81e36c6607594c4b520e
            py=$main_py; eval_runner="$S/run_generation_direct_fullft.py" ;;
  xlstm)    key=xlstm; base=$retained/xlstm
            base_tree=ff2e7c99151a290a7eb5428751c834316f0e70cf83c5cd966d3f31b482c41153
            py=$sealed_py; eval_runner="$v4/run_generation_direct_bs1.py"
            # xLSTM train mode needs sequence lengths that are multiples of its chunk size (64); pads are loss-masked.
            extra=(++finetune.training.pad_to_multiple_of=64) ;;
  gdn)      key=gated_deltanet; base=/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model
            base_tree=5ceca7d0d1e96bb6d330a7f49b997ede7102a2b350d9080bfb5d1c023cdefbf3
            py=$main_py; eval_runner="$S/run_generation_direct_fullft.py"
            train_env=(SALLM_SKIP_MAMBA_KERNEL_CHECK=1) ;;
  *) echo "ERROR: unknown arch $arch" >&2; exit 2 ;;
esac

echo "job=$SLURM_JOB_ID arch=$arch seed=$SEED batch=$BS lrs=${LRS[*]} node=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used --format=csv,noheader
/scratch/slurm/bin/purequota 2>/dev/null | grep -E '^/scratch' || true

[[ "$(sha256sum "$snapshot/SNAPSHOT_MANIFEST.sha256" | cut -d' ' -f1)" == "$snapshot_manifest" ]]
(cd "$snapshot" && sha256sum -c SNAPSHOT_MANIFEST.sha256 --quiet)
(cd "$S" && sha256sum -c SCRIPTS.sha256 --quiet)
(cd "$ROOT/assets/t2x_train_validation_only" && sha256sum -c "$S/T2X_ASSETS.sha256" --quiet)
tree() { "$main_py" "$S/fft_tools.py" tree "$1"; }

if [[ "$arch" == mamba2 ]]; then
  [[ "$(tree "$src_base")" == "$base_tree" ]] || { echo "ERROR: mamba2 base tree changed" >&2; exit 3; }
  if [[ ! -d "$base" ]]; then
    # The retained Mamba-2 base config has eos/pad swapped (eos 2, pad 1); the tokenizer has [EOS]=1, [PAD]=2.
    mkdir -p "$ROOT/bases"; tmp="$ROOT/bases/.mamba2_eosfix.$SLURM_JOB_ID"
    rsync -a --exclude .cache "$src_base/" "$tmp/"
    "$main_py" - "$tmp" <<'PY'
import json, sys
from pathlib import Path
for name in ("config.json", "generation_config.json"):
    p = Path(sys.argv[1]) / name
    c = json.loads(p.read_text())
    c["eos_token_id"], c["pad_token_id"] = 1, 2
    p.write_text(json.dumps(c, indent=2) + "\n")
tok = json.loads((Path(sys.argv[1]) / "tokenizer.json").read_text())
ids = {t["content"]: t["id"] for t in tok["added_tokens"]}
assert ids["[EOS]"] == 1 and ids["[PAD]"] == 2, ids
PY
    mv "$tmp" "$base"
  fi
  echo "mamba2_eosfix_tree=$(tree "$base")"
else
  observed="$(tree "$base")"
  [[ "$observed" == "$base_tree" ]] || { echo "ERROR: $arch base tree $observed != $base_tree" >&2; exit 3; }
fi
echo "base=$base tree_verified"

shm=/dev/shm/fft_${SLURM_JOB_ID}_${arch}_s${SEED}_b$BS
if [[ "$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)" -lt 24 ]]; then
  echo "ERROR: /dev/shm too small for epoch checkpoints" >&2; exit 4
fi
mkdir -p "$shm"
trap 'rm -rf "$shm"' EXIT
trap 'rm -rf "$shm"; exit 143' TERM INT

common_env=(HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled WANDB_SILENT=true TOKENIZERS_PARALLELISM=false
            PYTHONDONTWRITEBYTECODE=1 FLA_DISABLE_BACKEND_DISPATCH=1)

train_one() {  # lr precision run_dir out_dir
  local lr=$1 precision=$2 run=$3 out=$4 bf16=true
  [[ "$precision" == fp32 ]] && bf16=false
  local args=(
    --config-name finetune/llama_t2x_xho
    "finetune.model.architecture=$key"
    "finetune.model.init_checkpoint=$base"
    "finetune.tokenizer.path=$base"
    finetune.peft.method=none
    "finetune.training.output_dir=$out"
    "finetune.training.logging_dir=$run/logs"
    "finetune.training.run_name=fft-t2x-$arch-$(tag "$lr")"
    "hydra.run.dir=$run/hydra"
    finetune.training.report_to=none
    finetune.hub.enabled=false finetune.hub.push_adapter=false finetune.hub.push_merged=false
    finetune.dataset.max_seq_length=1024
    "finetune.training.learning_rate=$lr"
    ++finetune.training.optim=adamw_torch
    ++finetune.training.adam_beta1=0.9
    finetune.training.adam_beta2=0.95
    ++finetune.training.adam_epsilon=1e-8
    finetune.training.weight_decay=0.01
    finetune.training.lr_scheduler_type=cosine
    finetune.training.warmup_ratio=0.10
    "finetune.training.per_device_train_batch_size=$BS"
    finetune.training.gradient_accumulation_steps=1
    finetune.training.per_device_eval_batch_size=16
    finetune.training.num_train_epochs=4
    finetune.training.max_grad_norm=1.0
    "finetune.training.bf16=$bf16"
    finetune.training.gradient_checkpointing=false
    finetune.training.label_smoothing_factor=0.0
    finetune.training.eval_strategy=epoch
    finetune.training.save_strategy=epoch
    ++finetune.training.save_only_model=true
    ++finetune.training.save_total_limit=null
    ++finetune.training.load_best_model_at_end=false
    finetune.training.metric_for_best_model=null
    finetune.training.greater_is_better=null
    ++finetune.training.early_stopping_patience=null
    "++finetune.training.seed=$SEED"
    "++finetune.training.data_seed=$SEED"
    finetune.training.logging_steps=10
    finetune.generation_decoding.strategy=greedy
    ~finetune.generation_decoding.num_beams
    ~finetune.generation_decoding.length_penalty
    ~finetune.generation_decoding.early_stopping
    "${extra[@]}"
  )
  local envs=("${common_env[@]}" "${train_env[@]}"
    HF_HOME=/scratch/lmbanr001/hf HF_DATASETS_CACHE=/scratch/lmbanr001/hf/datasets HF_HUB_CACHE=/scratch/lmbanr001/hf/hub
    HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 SALLM_DISABLE_TASK_METRICS=1
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
    SALLM_T2X_TRAIN_VALIDATION_ONLY=1 "SALLM_T2X_CACHE_DIR=$ROOT/assets/t2x_train_validation_only"
    "PYTHONPATH=$snapshot/src/main" "TRITON_CACHE_DIR=$shm/triton")
  env "${envs[@]}" FFT_RUN_INFO=/dev/null "$py" "$S/train_fft.py" "${args[@]}" --cfg job --resolve > "$run/resolved_config.stdout" || return 1
  sed -n '/^finetune:/,$p' "$run/resolved_config.stdout" > "$run/resolved_config.yaml"
  [[ -s "$run/resolved_config.yaml" ]] || return 1
  if grep -Eiq '(^|[[:space:]])(test|test_split)[[:space:]]*:' "$run/resolved_config.yaml"; then
    echo "ERROR: held-out field in resolved config" >&2; return 6
  fi
  printf '%s\n' "${args[@]}" > "$run/overrides.txt"
  if [[ "$arch" == mamba2 ]]; then
    env "${envs[@]}" "$py" -c 'from transformers.models.mamba2 import modeling_mamba2 as m; assert m.is_fast_path_available; print("mamba fast path OK")' || return 1
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader -lms 5000 > "$run/nvidia_smi_mem.log" &
  local smi=$!
  local rc=0
  env "${envs[@]}" "FFT_RUN_INFO=$run/run_info.json" "$py" "$S/train_fft.py" "${args[@]}" > "$run/train.log" 2>&1 || rc=$?
  kill "$smi" 2>/dev/null || true
  return $rc
}

eval_val() {  # spec index output
  (cd "$gen" && env "${common_env[@]}" HF_HOME=/scratch/lmbanr001/hf-cache HF_DATASETS_CACHE=/scratch/lmbanr001/hf-cache/datasets \
     HF_HUB_CACHE=/scratch/lmbanr001/hf-cache/hub PYTHONHASHSEED=42 MAMBA_SCAN_IMPL=cuda \
     "SALLM_T2X_CACHE_DIR=$gen/data/t2x_cache" "SALLM_AFRIHG_CACHE_DIR=$gen/data/afrihg_cache" \
     "PYTHONPATH=$eval_source/src/main:$bundle" \
     "$py" "$eval_runner" --spec "$1" --index "$2" --source "$eval_source" --output "$3" \
       --split "${4:-val}" --system-prompt drop --decoding greedy)
}

sweep() {  # precision
  local precision=$1
  for lr in "${LRS[@]}"; do
    local run="$ROOT/runs/$arch/$(tag "$lr")" out="$shm/$arch/$(tag "$lr")"
    mkdir -p "$run" "$out"
    echo "TRAIN_START arch=$arch lr=$lr precision=$precision $(date -Is)"
    if ! train_one "$lr" "$precision" "$run" "$out"; then
      tail -40 "$run/train.log" >&2
      if grep -Eiq "returned nan values|'loss': nan|loss=nan|non-finite" "$run/train.log"; then return 42; fi
      return 1
    fi
    echo "TRAIN_DONE arch=$arch lr=$lr $(date -Is) $(cat "$run/run_info.json" | tr -d '\n ')"
    rm -rf "$out/final_merged_model"
    mapfile -t ckpts < <(ls -d "$out"/checkpoint-* | sort -t- -k2 -n)
    [[ ${#ckpts[@]} == 4 ]] || { echo "ERROR: expected 4 epoch checkpoints, got ${ckpts[*]}" >&2; return 1; }
    cp "${ckpts[3]}/trainer_state.json" "$run/trainer_state.json" || return 1
    cp "${ckpts[0]}/config.json" "$run/checkpoint_config.json" || return 1
    "$main_py" -c 'import json,subprocess,sys; print(json.dumps({p.rsplit("/",1)[1]: int(subprocess.check_output(["du","-sb",p]).split()[0]) for p in sys.argv[1:]}))' "${ckpts[@]}" > "$run/disk.json" || return 1
    "$main_py" "$S/fft_tools.py" valspec "$arch" "$lr" "$run/val_units.json" "${ckpts[@]}" || return 1
    for i in 0 1 2 3; do
      local unit; unit="$(jq -r ".[$i].unit_id" "$run/val_units.json")"
      local t0; t0=$(date +%s)
      eval_val "$run/val_units.json" "$i" "$run/val/$unit/greedy" > "$run/val_$unit.log" 2>&1 || { tail -40 "$run/val_$unit.log" >&2; return 1; }
      echo "VAL_DONE $unit secs=$(( $(date +%s) - t0 ))" | tee -a "$run/val_timing.txt"
    done
    local best
    best="$("$main_py" "$S/fft_tools.py" rundone "$arch" "$lr" "$precision" "$SEED" "$run" "$run/val_units.json" \
            "$run/trainer_state.json" "$run/run_info.json" "$run/disk.json" | tee /dev/stderr | sed -n 's/^BEST_CKPT //p')"
    [[ -d "$best" ]] || { echo "ERROR: no best checkpoint for lr $lr" >&2; return 1; }
    local epoch; epoch="$(jq -r .best_epoch "$run/RUN_DONE.json")"
    local dest="$ROOT/keep/$arch/$(tag "$lr")_e$epoch"
    mkdir -p "$ROOT/keep/$arch"
    rsync -a "$best/" "$dest.tmp/" && mv "$dest.tmp" "$dest" || return 1
    echo "KEPT $dest"
    rm -rf "$out"
  done
}

precision=bf16
[[ -e "$ROOT/runs/$arch/PRECISION_FP32" ]] && precision=fp32
rc=0
sweep "$precision" || rc=$?
if [[ $rc == 42 ]]; then
  echo "NONFINITE arch=$arch precision=$precision seed=$SEED: rerun this architecture's whole sweep in fp32" >&2
  [[ "$precision" == bf16 ]] && date -Is > "$ROOT/runs/$arch/PRECISION_FP32"
fi
[[ $rc == 0 ]] || { echo "ERROR: sweep failed rc=$rc" >&2; exit $rc; }
/scratch/slurm/bin/purequota 2>/dev/null | grep -E '^/scratch' || true

if [[ -n "$prefix" ]]; then
  # Extra seed / batch-size run at the selected lr: its own best epoch on validation, then TEST.
  spec="$ROOT/test_units_${arch}_${prefix%_}.json"
  "$main_py" "$S/fft_tools.py" extratest "$arch" "$(tag "${LRS[0]}")" "${LRS[0]}" "$spec"
  unit="$(jq -r '.[0].unit_id' "$spec")"
  rm -rf "$ROOT/test/$unit/greedy"; mkdir -p "$ROOT/test/$unit"
  t0=$(date +%s)
  eval_val "$spec" 0 "$ROOT/test/$unit/greedy" test > "$ROOT/logs/test_$unit.log" 2>&1
  echo "TEST_DONE $unit secs=$(( $(date +%s) - t0 ))" | tee -a "$ROOT/test_timing.txt"
  exec 9>"$ROOT/final.lock"; flock 9
  "$main_py" "$S/fft_tools.py" report
  exit 0
fi

# Seed 42 sweep: once all five lrs of this architecture are done, mark it; the last architecture runs final.sh.
exec 9>"$ROOT/final.lock"
flock 9
for lr in 1e-5 3e-5 1e-4 3e-4 3e-6; do
  [[ -f "$ROOT/runs/$arch/lr$lr/RUN_DONE.json" ]] || { echo "lr $lr of $arch not done yet"; exit 0; }
done
touch "$ROOT/runs/$arch/SWEEP_DONE"
echo "SWEEP_DONE arch=$arch $(date -Is)"
for a in mzansilm mamba2 xlstm gdn; do [[ -f "$ROOT/runs/$a/SWEEP_DONE" ]] || { echo "waiting for $a; not final"; exit 0; }; done
[[ ! -e "$ROOT/FINAL_STARTED" ]] || { echo "final already started"; exit 0; }
echo "$SLURM_JOB_ID $(date -Is)" > "$ROOT/FINAL_STARTED"
bash "$S/final.sh"
