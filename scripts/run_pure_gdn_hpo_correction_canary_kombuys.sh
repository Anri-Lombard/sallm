#!/bin/bash
set -euo pipefail

source_repo="${SALLM_CANARY_SOURCE_REPO:-/scratch/alombard/sallm}"
runtime_repo="${SALLM_CANARY_RUNTIME_REPO:-/scratch/alombard/sallm}"
python_bin="${SALLM_CANARY_PYTHON:-$runtime_repo/.venv/bin/python}"
checkpoint="${SALLM_CANARY_CHECKPOINT:-$runtime_repo/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model}"
result_root="${SALLM_CANARY_RESULT_ROOT:-$runtime_repo/results/pure_gdn_hpo_correction_canary/2026-08-09}"
attempt="${1:-attempt-2}"
if [[ ! "$attempt" =~ ^attempt-[0-9]+$ ]]; then
  echo "ERROR: attempt must match attempt-N, got '$attempt'" >&2
  exit 2
fi
output_dir="$result_root/$attempt"

if [[ -e "$output_dir" ]]; then
  echo "ERROR: preserving existing canary attempt at $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"
exec > >(tee -a "$output_dir/canary.log") 2>&1

cd "$source_repo"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$source_repo/src/main:${PYTHONPATH:-}"
export HF_HOME="$runtime_repo/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export FLA_DISABLE_BACKEND_DISPATCH=1
export SALLM_SKIP_MAMBA_KERNEL_CHECK=1
export TOKENIZERS_PARALLELISM=true

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader > "$output_dir/gpu-before.csv"

"$python_bin" scripts/create_execution_manifest.py \
  --repo-root "$source_repo" \
  --output "$output_dir/execution_manifest.json" \
  --entrypoint scripts/run_pure_gdn_hpo_correction_canary_kombuys.sh \
  --command "CUDA_VISIBLE_DEVICES=1 validation-only correction canary" \
  --artifact-root "$checkpoint"
"$python_bin" scripts/create_execution_manifest.py \
  --verify "$output_dir/execution_manifest.json"

"$python_bin" scripts/run_pure_gdn_hpo_correction_canary.py \
  --checkpoint "$checkpoint" \
  --device cuda:0 \
  --expected-gpu "NVIDIA GeForce RTX 3080 Ti" \
  --output "$output_dir/canary_result.json"

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader > "$output_dir/gpu-after.csv"
echo "PASS: pure-GDN HPO correction canary completed"
