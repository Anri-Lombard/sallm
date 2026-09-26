#!/bin/bash
#SBATCH --job-name=fft-t2x-final
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/equal_recipe_fullft_t2x_pilot_20260925/logs/%x-%j.out
# Cross-architecture edge rule + selection, then TEST scoring of exactly one checkpoint per architecture.
# Runs at the end of the last sweep_arch job (or standalone via sbatch if that job cannot).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT=/scratch/lmbanr001/masters/sallm/results/equal_recipe_fullft_t2x_pilot_20260925
S="$ROOT/scripts"
main_py=/home/lmbanr001/masters/sallm/.venv/bin/python
sealed_py=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
gen=/scratch/lmbanr001/masters/sallm_snapshots/generation-protocol-v3-greedy-20260924
v4=/scratch/lmbanr001/masters/sallm_snapshots/generation-protocol-v4-xlstm-bs1-20260925
eval_source=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
(cd "$S" && sha256sum -c SCRIPTS.sha256 --quiet)

spec="$ROOT/test_units.json"
[[ -s "$spec" ]] || "$main_py" "$S/fft_tools.py" final "$spec"

# At each architecture's selected lr: seeds 43/44 (batch 16) and batch 8/32 (seed 42); one L40S job each, submitted now.
if [[ ! -e "$ROOT/EXTRAS_SUBMITTED" ]]; then
  while IFS=$'\t' read -r a lr; do
    for extra in "43 16" "44 16" "42 8" "42 32"; do
      read -r seed bs <<< "$extra"
      j=$(sbatch --parsable --job-name="fft-$a-s$seed-b$bs" --export=ALL,ARCH=$a,SEED=$seed,BS=$bs,LRS=$lr "$S/sweep_arch.sbatch") \
        && echo "# $(date -Iseconds) sbatch extra job $a seed=$seed batch=$bs lr=$lr -> $j" | tee -a "$ROOT/COMMANDS.log" \
        || echo "WARNING: could not submit extra job $a seed=$seed batch=$bs from $(hostname)" | tee -a "$ROOT/COMMANDS.log"
    done
  done < "$ROOT/SEEDS_PLAN.tsv"
  touch "$ROOT/EXTRAS_SUBMITTED"
fi

# Delete seed-42 sweep checkpoints that were not selected.
for d in "$ROOT"/keep/*/lr*; do
  jq -e --arg d "$d" 'any(.[]; .base == $d)' "$spec" > /dev/null || { echo "DELETE unselected $d"; rm -rf "$d"; }
done

for i in 0 1 2 3; do
  unit="$(jq -r ".[$i].unit_id" "$spec")"; arch="$(jq -r ".[$i].architecture" "$spec")"
  out="$ROOT/test/$unit/greedy"
  [[ -f "$out/UNIT_DONE.json" ]] && continue
  rm -rf "$out"; mkdir -p "$(dirname "$out")"
  if [[ "$arch" == xlstm ]]; then py=$sealed_py; runner="$v4/run_generation_direct_bs1.py"
  else py=$main_py; runner="$S/run_generation_direct_fullft.py"; fi
  t0=$(date +%s)
  (cd "$gen" && env HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 \
     FLA_DISABLE_BACKEND_DISPATCH=1 HF_HOME=/scratch/lmbanr001/hf-cache HF_DATASETS_CACHE=/scratch/lmbanr001/hf-cache/datasets \
     HF_HUB_CACHE=/scratch/lmbanr001/hf-cache/hub PYTHONHASHSEED=42 MAMBA_SCAN_IMPL=cuda \
     "SALLM_T2X_CACHE_DIR=$gen/data/t2x_cache" "SALLM_AFRIHG_CACHE_DIR=$gen/data/afrihg_cache" \
     "PYTHONPATH=$eval_source/src/main:$bundle" \
     "$py" "$runner" --spec "$spec" --index "$i" --source "$eval_source" --output "$out" \
       --split test --system-prompt drop --decoding greedy) > "$ROOT/logs/test_$unit.log" 2>&1
  echo "TEST_DONE $unit secs=$(( $(date +%s) - t0 ))" | tee -a "$ROOT/test_timing.txt"
done
"$main_py" "$S/fft_tools.py" report "$spec"
touch "$ROOT/FINAL_DONE"
/scratch/slurm/bin/purequota 2>/dev/null | grep -E '^/scratch' || true
