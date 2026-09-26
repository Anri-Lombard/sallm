#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --qos=l40sfree
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=rsel-parity2
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925/slurm/%x-%j.out
# NER parity + NER val smoke (after the offline-flag fix), and an unrestricted Kombuys run_prefix_eval.py rerun of
# GDN Intent Multi on HEX to separate the effect of restricting lm-eval to the protocol tasks from GPU/bf16 drift.
set -uo pipefail
root=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925
V9=/scratch/lmbanr001/masters/sallm/results/full_matrix_targeted_recovery_20260916_v9/train
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
step() { local t0=$(date +%s); echo "=== START $* $(date -Is)"; "$@"; local rc=$?; echo "=== END rc=$rc $(( $(date +%s) - t0 ))s $*"; }
step bash $root/jobs/stage_test.sh mzansilm_ner_multi_monorecipe --adapter $V9/mzansilm_ner_multi/final_adapter
RESELECT_TRAIN_ROOT=$root/tmp/smoke/train RESELECT_VAL_ROOT=$root/tmp/smoke/val step bash $root/jobs/stage_val.sh mzansilm_ner_multi_monorecipe
full() {
  env -u PYTHONPATH HF_HOME=/scratch/lmbanr001/hf-cache HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
    WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1 FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda \
    TMPDIR=$root/tmp/work/full-$SLURM_JOB_ID \
    PYTHONPATH=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8/src/main:/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1 \
    /home/lmbanr001/masters/sallm/.venv/bin/python $root/jobs/kit/run_prefix_eval_kombuys.py --spec $root/tmp/spec_gdn_intent.json \
    --unit-id parity-gdn-intent-multi --arch gdn --mode train --packs injongointent_all --output $root/tmp/intent_full_gdn_multi
}
mkdir -p $root/tmp/work/full-$SLURM_JOB_ID
step full
echo PARITY2_DONE
