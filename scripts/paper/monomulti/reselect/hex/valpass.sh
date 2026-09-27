#!/bin/bash
# One scoring pass (run inside an existing allocation via srun --overlap): for every label under train/,
# score new epoch checkpoints on validation (partial while training runs); once final_adapter exists write the
# selection and score the selected checkpoint on test.
set -uo pipefail
root=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925
only="${1:-}"
# scorers share GPUs with training lanes: cap their memory (lm-eval auto batch then picks a smaller batch)
export RESELECT_MEMCAP="${RESELECT_MEMCAP:-0.33}"
echo "${SLURM_JOB_ID:-none} $(date +%s)" > "$root/valpass.lock"
trap 'rm -f "$root/valpass.lock"' EXIT
for d in "$root"/train/*/; do
  label=$(basename "$d")
  [[ -n "$only" && ! "$label" =~ $only ]] && continue
  ls -d "$d"checkpoint-* >/dev/null 2>&1 || continue
  if [[ -e "$d/final_adapter/adapter_config.json" ]]; then
    [[ -s "$root/test/$label.json" ]] && continue
    echo "[$(date -Is)] VAL-FINAL $label"
    bash "$root/jobs/stage_val.sh" "$label" >> "$root/logs/val_$label.log" 2>&1 || { echo "  val failed $label"; continue; }
    echo "[$(date -Is)] TEST $label"
    bash "$root/jobs/stage_test.sh" "$label" >> "$root/logs/test_$label.log" 2>&1 || { echo "  test failed $label"; continue; }
    echo "[$(date -Is)] PRUNE $label"
    bash "$root/jobs/prune.sh" "$label" --yes >> "$root/logs/prune_$label.log" 2>&1 || echo "  prune failed $label"
  else
    echo "[$(date -Is)] VAL-PARTIAL $label"
    RESELECT_ALLOW_PARTIAL=1 bash "$root/jobs/stage_val.sh" "$label" >> "$root/logs/val_$label.log" 2>&1 || echo "  partial val failed $label"
  fi
done
echo "[$(date -Is)] PASS_DONE"
