#!/bin/bash
# Run manually with sbatch; this script never submits itself.
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/home/lmbanr001/masters/sallm

set -euo pipefail

repo="${SALLM_REPO_DIR:?immutable source snapshot is required}"
runtime="${SALLM_RUNTIME_REPO:-$HOME/masters/sallm}"
output="${SALLM_POS_GATE_OUTPUT:?gate output path is required}"
mode="${SALLM_POS_GATE_MODE:?full, cached, or both is required}"
checkpoint="${SALLM_POS_GATE_CHECKPOINT:?base checkpoint is required}"
adapter="${SALLM_POS_GATE_ADAPTER:?frozen adapter checkpoint is required}"

module load python/miniconda3-py3.12
cd "$runtime"
source .venv/bin/activate
export PYTHONPATH="$repo/src/main:${PYTHONPATH:-}"
export FLA_DISABLE_BACKEND_DISPATCH=1
export SALLM_SKIP_MAMBA_KERNEL_CHECK=1

python "$repo/scripts/create_execution_manifest.py" \
  --repo-root "$repo" \
  --output "${output%.json}.execution_manifest.json" \
  --entrypoint scripts/run_pos_runtime_equivalence_gate.sh \
  --command "mode=$mode checkpoint=$checkpoint adapter=$adapter gres=${SLURM_JOB_GRES:-unknown}" \
  --artifact-root "$checkpoint"

python "$repo/scripts/benchmark_pos_incremental_cache.py" \
  --checkpoint "$checkpoint" \
  --adapter "$adapter" \
  --mode "$mode" \
  --output "$output"
