#!/bin/bash
# usage: valpass_kb.sh <gpu_index> [label_regex] [--loop SECONDS]
# Kombuys port of HEX jobs/valpass.sh (News/SIB/Intent labels only). For every label under train/: score new epoch
# checkpoints on validation (partial while training runs); once final_adapter exists, final validation -> selection ->
# test on the selected checkpoint. With --loop N it repeats every N seconds until every label has test/<label>.json.
# Pruning is separate (prune.sh <label> [--yes]).
set -uo pipefail
root=/scratch/alombard/sallm/results/monomulti_reselect_20260925
gpu="${1:?gpu index}"; only="${2:-}"; loop=0
[[ "${3:-}" == --loop ]] && loop="${4:?seconds}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
mkdir -p "$root/logs"
while true; do
  pending=0
  for d in "$root"/train/*/; do
    label=$(basename "$d")
    [[ -n "$only" && ! "$label" =~ $only ]] && continue
    [[ "$label" == *_ner_* ]] && continue
    ls -d "$d"checkpoint-* >/dev/null 2>&1 || { pending=1; continue; }
    if [[ -e "$d/final_adapter/adapter_config.json" ]]; then
      [[ -s "$root/test/$label.json" ]] && continue
      pending=1
      echo "[$(date -Is)] VAL-FINAL $label gpu=$gpu"
      bash "$root/jobs/stage_val.sh" "$label" >> "$root/logs/val_$label.log" 2>&1 || { echo "  val failed $label"; continue; }
      echo "[$(date -Is)] TEST $label gpu=$gpu"
      bash "$root/jobs/stage_test.sh" "$label" >> "$root/logs/test_$label.log" 2>&1 || echo "  test failed $label"
    else
      pending=1
      echo "[$(date -Is)] VAL-PARTIAL $label gpu=$gpu"
      RESELECT_ALLOW_PARTIAL=1 bash "$root/jobs/stage_val.sh" "$label" >> "$root/logs/val_$label.log" 2>&1 || echo "  partial val failed $label"
    fi
  done
  echo "[$(date -Is)] PASS_DONE pending=$pending"
  [[ "$loop" -gt 0 && "$pending" -eq 1 ]] || break
  sleep "$loop"
done
