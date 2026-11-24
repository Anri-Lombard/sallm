#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=sallm-pretrain
#SBATCH --mail-type=FAIL,END

set -euo pipefail

CONFIG_PATH="${1:-src/conf/pretrain/mzansilm_wsd.json}"
DATA_DIR="${2:-${MP_DATA_DIR:-}}"
OUT_DIR="${3:-${MP_OUT_DIR:-}}"
if [[ "$#" -ge 3 ]]; then
  shift 3
elif [[ "$#" -gt 0 ]]; then
  shift "$#"
fi

if [[ -z "$DATA_DIR" || -z "$OUT_DIR" ]]; then
  echo "Usage: sbatch $0 <config_json> <packed_data_dir> <output_dir> [extra sallm.pretrain args...]" >&2
  echo "Example: sbatch $0 src/conf/pretrain/gdn_wsd.json \$SCRATCH/data/packed_bin \$SCRATCH/runs/gdn" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/auth.sh"
source "$SCRIPT_DIR/lib/env.sh"
set_sallm_cluster_env
setup_sallm_runtime "$CONFIG_PATH" all

export MP_TOKENIZER="${MP_TOKENIZER:-$SALLM_REPO_DIR/tokenizer/sallm_bpe_tokenizer}"
export MP_VAL_PARQUET="${MP_VAL_PARQUET:-$SCRATCH/data/validation.parquet}"

NUM_PROCESSES="$(detect_gpu_count)"
echo "Launching matched pretraining with ${CONFIG_PATH} on ${NUM_PROCESSES} GPUs..."

torchrun --nproc_per_node="$NUM_PROCESSES" -m sallm.pretrain \
  --config "$CONFIG_PATH" \
  --data "$DATA_DIR" \
  --out "$OUT_DIR" \
  --stack fast \
  "$@"
