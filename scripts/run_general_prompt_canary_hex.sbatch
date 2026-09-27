#!/usr/bin/env bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --nodelist=srvrocgpu010
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,END

set -euo pipefail
/scratch/slurm/bin/purequota

snapshot=/scratch/lmbanr001/masters/sallm_snapshots/general-prompt-correction-20260914-v1
asset_root=/scratch/lmbanr001/masters/sallm/assets/general_prompt_correction_20260914_v1
canary_root=/scratch/lmbanr001/masters/sallm/audits/general_prompt_correction_20260914_v1/canary_ampere_v1
python_bin=/home/lmbanr001/masters/sallm/.venv/bin/python

(cd "$snapshot" && sha256sum --check --strict SNAPSHOT_EXECUTION_MANIFEST.json.sha256)
"$python_bin" "$snapshot/scripts/create_execution_manifest.py" \
  --verify "$snapshot/SNAPSHOT_EXECUTION_MANIFEST.json" \
  --expected-repo-root "$snapshot"
sha256sum --check --strict "$snapshot/.audit/general_prompt_correction_model_files_20260914.sha256"
[[ ! -e "$canary_root" ]]
mkdir -p "$canary_root"

export PYTHONPATH="$snapshot/src/main:$snapshot/scripts"
export HF_HOME=/scratch/lmbanr001/hf
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export TMPDIR="${SLURM_TMPDIR:-/scratch/lmbanr001/tmp/general-prompt-canary-$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

"$python_bin" "$snapshot/scripts/run_general_prompt_lm_eval.py" \
  --phase validation \
  --architecture mzansilm \
  --checkpoint "$asset_root/mzansilm/base" \
  --adapter "$asset_root/mzansilm/adapter" \
  --output "$canary_root/lm_eval.json" \
  --limit 1

"$python_bin" "$snapshot/scripts/run_injongointent_mean_eval.py" \
  --config "$snapshot/src/conf/finetune/llama_injongointent_all.yaml" \
  --checkpoint "$asset_root/mzansilm/base" \
  --adapter "$asset_root/mzansilm/adapter" \
  --architecture mzansilm \
  --output "$canary_root/intent.json" \
  --split validation \
  --max-length 1024 \
  --max-samples-per-lang 1

"$python_bin" "$snapshot/scripts/verify_general_prompt_canary.py" \
  --lm "$canary_root/lm_eval.json" \
  --intent "$canary_root/intent.json" \
  --output "$canary_root/VERIFIED.json"
