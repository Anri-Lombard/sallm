#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --qos=l40sfree
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# env: ARCH BASE SPLIT OUTDIR ADAPTERS="dir dir ..."
set -euo pipefail
R=/scratch/lmbanr001/masters/sallm/results/monomulti_retrain_20260924
K=$R/sibkit
[[ "$(sha256sum $K/run_sib_natural_first_token_eval.py | cut -d' ' -f1)" == b247505aa86b9d6c6b2cdf3184bb9f6a3bf4830b8b8f17302535a3d65f2278c8 ]]
[[ "$(sha256sum $K/run_sib_balanced_code_eval.py | cut -d' ' -f1)" == 866e9d82a21a894fb21a2044c05105b6d6eeb7f20dd13a5c07d46e84b47c6ba8 ]]
source_snapshot=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
(cd "$source_snapshot" && sha256sum -c SNAPSHOT_MANIFEST.sha256 --quiet)
python=/home/lmbanr001/masters/sallm/.venv/bin/python
[[ "$ARCH" == xlstm ]] && python=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
export HF_HOME=/scratch/lmbanr001/hf-cache HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1
export FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda
export PYTHONPATH="$source_snapshot/src/main"
nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p "$OUTDIR"
read -r -a adapters <<< "$ADAPTERS"
"$python" $K/score_sib_split_hex.py --arch "$ARCH" --base "$BASE" --split "$SPLIT" --out-dir "$OUTDIR" --adapter "${adapters[@]}"
echo SIB_EVAL_DONE
