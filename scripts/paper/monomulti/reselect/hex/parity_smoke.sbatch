#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --qos=l40sfree
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=rsel-parity
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925/slurm/%x-%j.out
# Parity (stage_test.sh --adapter on existing adapters, compare with Kombuys rescore_results.csv) and
# stage_val.sh smoke tests on fake labels under tmp/smoke (one per task), each step timed.
set -uo pipefail
root=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925
C=/scratch/lmbanr001/masters/sallm/checkpoints
V9=/scratch/lmbanr001/masters/sallm/results/full_matrix_targeted_recovery_20260916_v9/train
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
step() { local t0=$(date +%s); echo "=== START $* $(date -Is)"; "$@"; local rc=$?; echo "=== END rc=$rc $(( $(date +%s) - t0 ))s $*"; }
T="$root/jobs/stage_test.sh"
step bash $T gdn_sib_multi --adapter $C/adapter_hpo_v3/pure_gdn/sib/stage_a/a0/seed_42/final_adapter
step bash $T gdn_news_multi --adapter $C/adapter_hpo_v3/pure_gdn/news/stage_a/a0/seed_42_env_correction_20260901/final_adapter
step bash $T gdn_intent_multi --adapter $C/adapter_hpo_v3/pure_gdn/intent/stage_a/a1/seed_42/final_adapter
step bash $T mzansilm_intent_multi --adapter $C/intent_recovery_20260729/llama_clean_r2/checkpoint-1764
step bash $T mzansilm_ner_multi_monorecipe --adapter $V9/mzansilm_ner_multi/final_adapter
export RESELECT_TRAIN_ROOT=$root/tmp/smoke/train RESELECT_VAL_ROOT=$root/tmp/smoke/val
for label in gdn_sib_multi gdn_news_multi gdn_intent_multi mzansilm_ner_multi_monorecipe; do
  step bash $root/jobs/stage_val.sh $label
done
step bash $root/jobs/stage_val.sh gdn_intent_multi
unset RESELECT_TRAIN_ROOT RESELECT_VAL_ROOT
step bash $T xlstm_news_multi --adapter $C/xlstm_news_recovery_20260728/all_rolefix_val_r2/final_adapter
step bash $T xlstm_intent_multi --adapter $C/intent_recovery_20260729/xlstm_clean_r4/checkpoint-442
step bash $T mamba2_intent_multi --adapter $C/intent_recovery_20260729/mamba_clean_r2/checkpoint-888
echo PARITY_SMOKE_DONE
