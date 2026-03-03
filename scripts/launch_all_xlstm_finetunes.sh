#!/bin/bash
# Launch all xLSTM finetune jobs
#
# This script submits all xLSTM monolingual finetune jobs to SLURM
# with staggered submissions to avoid overwhelming the scheduler.

# All xLSTM finetune configs
CONFIGS=(
  # NER tasks (3)
  "xlstm_ner_xho"
  "xlstm_ner_zul"
  "xlstm_ner_tsn"

  # POS tasks (3)
  "xlstm_pos_xho"
  "xlstm_pos_zul"
  "xlstm_pos_tsn"

  # News tasks (2)
  "xlstm_news_eng"
  "xlstm_news_xho"

  # SIB200 tasks (6)
  "xlstm_sib_afr"
  "xlstm_sib_eng"
  "xlstm_sib_nso"
  "xlstm_sib_sot"
  "xlstm_sib_xho"
  "xlstm_sib_zul"
)

echo "============================================================"
echo "xLSTM Monolingual Finetune Batch Submission"
echo "Total configs: ${#CONFIGS[@]}"
echo "============================================================"
echo ""

SUBMITTED=0
FAILED=0

for cfg in "${CONFIGS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Submitting: $cfg"

  if sbatch scripts/launch_finetune.sh "$cfg"; then
    SUBMITTED=$((SUBMITTED + 1))
    echo "  ✓ Job submitted successfully"
  else
    FAILED=$((FAILED + 1))
    echo "  ✗ Job submission failed"
  fi

  # Stagger submissions by 30 seconds to avoid scheduler overload
  if [ "$cfg" != "${CONFIGS[-1]}" ]; then
    echo "  Waiting 30s before next submission..."
    sleep 30
  fi
  echo ""
done

echo "============================================================"
echo "Submission Summary"
echo "  Submitted: $SUBMITTED"
echo "  Failed: $FAILED"
echo "============================================================"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u lmbanr001 | grep 'ft-'"
echo ""
echo "Check job logs in:"
echo "  ~/masters/sallm/logs/ft-*"
echo ""
echo "Track progress on WandB:"
echo "  https://wandb.ai/anri-lombard/sallm-ft"
echo ""
