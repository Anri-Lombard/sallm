#!/bin/bash
set -euo pipefail

repo=/scratch/alombard/sallm_snapshots/pure-gdn-a100-equivalence-20260822-38d367bf
runtime=/scratch/alombard/sallm
output=/scratch/alombard/sallm/results/cross_host_hardware_equivalence/2026-08-27/pos-rtx5090.json
checkpoint=/scratch/alombard/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model
adapter=/scratch/alombard/sallm/checkpoints/adapter_hpo_v3/pure_gdn/pos/stage_b/b0/seed_42/checkpoint-566
manifest="${output%.json}.execution_manifest.json"

for target in "$output" "$output.sha256" "$manifest" "$manifest.sha256"; do
  test ! -e "$target" || { echo "Refusing to overwrite $target" >&2; exit 2; }
done

cd "$runtime"
source .venv/bin/activate
export PYTHONPATH="$repo/src/main:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export FLA_DISABLE_BACKEND_DISPATCH=1
export SALLM_SKIP_MAMBA_KERNEL_CHECK=1
export SLURM_JOB_ID=kombuys-20260827-rtx5090-gate
export SLURM_JOB_NAME=gdn-rtx5090-equivalence
export SLURM_JOB_GRES=gpu:rtx5090:1
export SLURM_JOB_PARTITION=kombuys-local
export SLURM_JOB_QOS=local

python "$repo/scripts/create_execution_manifest.py" \
  --repo-root "$repo" \
  --output "$manifest" \
  --entrypoint scripts/run_pos_runtime_equivalence_gate.sh \
  --command "mode=full checkpoint=$checkpoint adapter=$adapter gres=$SLURM_JOB_GRES" \
  --artifact-root "$checkpoint"

python "$repo/scripts/benchmark_pos_incremental_cache.py" \
  --checkpoint "$checkpoint" \
  --adapter "$adapter" \
  --mode full \
  --output "$output"
