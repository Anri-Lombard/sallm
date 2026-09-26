#!/bin/bash
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

candidate="${1:?Usage: $0 <b0-b7>}"
[[ "$candidate" =~ ^b[0-7]$ ]] || { echo "Invalid candidate: $candidate" >&2; exit 1; }
repo="${SALLM_REPO_DIR:?immutable source snapshot is required}"
runtime="${SALLM_RUNTIME_REPO:-$HOME/masters/sallm}"
scratch="${SCRATCH:-/scratch/lmbanr001}"
base="$scratch/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model"
adapter="$scratch/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/pos/stage_b/b0/seed_42/checkpoint-566"
gate_dir="$scratch/masters/sallm/results/diagnostics/pos_runtime_gate_20260816/job-${SLURM_JOB_ID}"
mkdir -p "$gate_dir"

module load python/miniconda3-py3.12
cd "$runtime"
source .venv/bin/activate
export PYTHONPATH="$repo/src/main:${PYTHONPATH:-}"
export FLA_DISABLE_BACKEND_DISPATCH=1
export SALLM_SKIP_MAMBA_KERNEL_CHECK=1

python "$repo/scripts/create_execution_manifest.py" \
  --repo-root "$repo" \
  --output "$gate_dir/execution_manifest.json" \
  --entrypoint scripts/run_pos_stage_b_with_cache_gate.sh \
  --command "candidate=$candidate gate=checkpoint-566 then incremental_cache_last_logit_v2" \
  --artifact-root "$base"
python "$repo/scripts/benchmark_pos_incremental_cache.py" \
  --checkpoint "$base" \
  --adapter "$adapter" \
  --mode both \
  --output "$gate_dir/equivalence.json"
python "$repo/scripts/verify_pos_runtime_equivalence.py" \
  --input "$gate_dir/equivalence.json" \
  --output "$gate_dir/verification.json"

export SALLM_POS_INCREMENTAL_CACHE=1
exec bash "$repo/scripts/run_validation_hpo_trial.sh" \
  pure_gdn pos stage_b "$candidate" 42
