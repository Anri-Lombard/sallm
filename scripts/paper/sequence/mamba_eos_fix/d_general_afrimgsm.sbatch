#!/bin/bash
#SBATCH --job-name=eosfix-gen-mgsm
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/logs/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/logs/%x-%j.err
set -euo pipefail
source /scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/sb/common.sh
bundle=$R/kombuys_snapshot/general-prompt-official-kombuys-20260915-v1
snapshot=$R/kombuys_snapshot/general-prompt-correction-20260915-v8-sixfamily-tso-all-interface-fix
(cd "$bundle" && sha256sum --check --strict --quiet BUNDLE_MANIFEST.sha256)
export PYTHONPATH="$R/sb:$bundle:$snapshot/scripts:$snapshot/src/main"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
adapter=/scratch/lmbanr001/masters/sallm/checkpoints/ft_mamba_125m_sa_general_sixfamily_tokenbalanced_r1_kombuys_verified/final_adapter
out=$R/general_afrimgsm/raw/test/lm_eval/mamba2.json
mkdir -p "$(dirname $out)"
"$PY" $R/sb/general_afrimgsm_wrapper.py --phase test --architecture mamba2 --checkpoint "$FIX" --adapter "$adapter" \
  --selection "$bundle/selection/SELECTION.json" --dtype float32 --tie-word-embeddings false \
  --batch-size 8 --max-batch-size 8 --output "$out"
date -Is
