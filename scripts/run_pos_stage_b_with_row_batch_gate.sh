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

candidate="${1:?Usage: $0 b2}"
[[ "$candidate" == b2 ]] || { echo "Only pending b2 is authorized." >&2; exit 1; }
repo="${SALLM_REPO_DIR:?immutable source snapshot is required}"
runtime="${SALLM_RUNTIME_REPO:-$HOME/masters/sallm}"
scratch="${SCRATCH:-/scratch/lmbanr001}"
base="$scratch/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model"
adapter="$scratch/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/pos/stage_b/b0/seed_42/checkpoint-566"
gate_dir="$scratch/masters/sallm/results/diagnostics/pos_row_batch_gate_20260816/job-${SLURM_JOB_ID}"
mkdir -p "$gate_dir"

module load python/miniconda3-py3.12
cd "$runtime"
source .venv/bin/activate
export PYTHONPATH="$repo/src/main:${PYTHONPATH:-}"
export FLA_DISABLE_BACKEND_DISPATCH=1
export SALLM_SKIP_MAMBA_KERNEL_CHECK=1
export SALLM_POS_ROW_BATCH_SIZE=8
export SALLM_HPO_RUN_ID_OVERRIDE=hpo-pure_gdn-pos-stage_b-b2-seed42-rowbatch-v3
export SALLM_HPO_OUTPUT_DIR_OVERRIDE="$scratch/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/pos/stage_b/b2-rowbatch-v3/seed_42"
export SALLM_HPO_LOGGING_DIR_OVERRIDE="$scratch/masters/sallm/logs/adapter_hpo_v3/pure_gdn/pos/stage_b/b2-rowbatch-v3/seed_42"

python "$repo/scripts/create_execution_manifest.py" \
  --repo-root "$repo" \
  --output "$gate_dir/execution_manifest.json" \
  --entrypoint scripts/run_pos_stage_b_with_row_batch_gate.sh \
  --command "candidate=b2 gate=checkpoint-566 then row_batch_full_prefix_v1 batch_size=8" \
  --artifact-root "$base"
python "$repo/scripts/benchmark_pos_row_batch.py" \
  --checkpoint "$base" \
  --adapter "$adapter" \
  --output "$gate_dir/equivalence.json"
python "$repo/scripts/verify_pos_row_batch_equivalence.py" \
  --input "$gate_dir/equivalence.json" \
  --output "$gate_dir/verification.json"

exec bash "$repo/scripts/run_validation_hpo_trial.sh" \
  pure_gdn pos stage_b b2 42
