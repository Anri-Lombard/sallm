#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gdn-prompt-canary
#SBATCH --mail-type=FAIL,END

set -euo pipefail

source_repo="${SALLM_CANARY_SOURCE_REPO:?immutable source snapshot is required}"
runtime_repo="${SALLM_CANARY_RUNTIME_REPO:-$HOME/masters/sallm}"
scratch_root="${SCRATCH:-/scratch/$USER}"
checkpoint="${SALLM_CANARY_CHECKPOINT:-$scratch_root/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model}"
output_dir="${SALLM_CANARY_OUTPUT_DIR:?new canary output directory is required}"

if [[ -e "$output_dir" ]]; then
  echo "ERROR: preserving existing canary output at $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"
exec > >(tee -a "$output_dir/canary.log") 2>&1

module load python/miniconda3-py3.12
source "$runtime_repo/.venv/bin/activate"

export PYTHONPATH="$source_repo/src/main:${PYTHONPATH:-}"
export HF_HOME="$scratch_root/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export FLA_DISABLE_BACKEND_DISPATCH=1
export SALLM_SKIP_MAMBA_KERNEL_CHECK=1
export TOKENIZERS_PARALLELISM=true

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$gpu_name" != "NVIDIA A100-PCIE-40GB" ]]; then
  echo "ERROR: expected NVIDIA A100-PCIE-40GB, got '$gpu_name'" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader > "$output_dir/gpu-before.csv"

python "$source_repo/scripts/create_execution_manifest.py" \
  --repo-root "$source_repo" \
  --output "$output_dir/execution_manifest.json" \
  --entrypoint scripts/run_pure_gdn_prompt_contract_canary_hex.sh \
  --command "A100-40GB validation-only prompt-contract correction canary" \
  --artifact-root "$checkpoint"
python "$source_repo/scripts/create_execution_manifest.py" \
  --verify "$output_dir/execution_manifest.json"

python "$source_repo/scripts/run_pure_gdn_hpo_correction_canary.py" \
  --checkpoint "$checkpoint" \
  --device cuda:0 \
  --expected-gpu "$gpu_name" \
  --output "$output_dir/canary_result.json"

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader > "$output_dir/gpu-after.csv"
echo "PASS: pure-GDN prompt-contract canary completed"
