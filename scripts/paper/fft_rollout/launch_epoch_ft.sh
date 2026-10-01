#!/bin/bash
# Multi-regime fine-tuning (plus T2X) of the 2-4 epoch checkpoints, seed 42, at each architecture's epoch-1 selected
# learning rates (no sweep; Sun & Dredze 2025 fix the rate across checkpoints). Plus an epoch-4 learning-rate check:
# NER and SIB-200 at one step below the selected rate (Springer et al. 2025: longer-pretrained models are more sensitive
# to the fine-tuning rate). Each run's lanes wait (afterany) for its pretraining branch job in matched_pretrain runs/ext.
# One lane per run (the 50-job submit limit; add lanes with addlane.sh later). HEX head node (bash + sbatch only): bash launch_epoch_ft.sh [ARCH ...]   (default: all four)
set -euo pipefail
R=/scratch/lmbanr001/masters/sallm/results
F=$R/fft_rollout_20260926
X=$R/matched_pretrain_20260926/runs/ext
FAMS="news sib intent ner pos afrihg nchlt_ner t2x"
LADDER=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 6e-3)
lrs_json() {  # arch -> {"family": "selected lr", ...} from the epoch-1 run, optionally one ladder step down
  local a=$1 down=${2:-0} out="" f lr i
  for f in $FAMS; do
    [ -n "${3:-}" ] && [[ " $3 " != *" $f "* ]] && continue
    lr=$(jq -r .lr "$F/runs/$a/keep/$f/SELECTED.json")
    if [ "$down" = 1 ]; then for i in "${!LADDER[@]}"; do [ "${LADDER[$i]}" = "$lr" ] && lr=${LADDER[$((i - 1))]}; done; fi
    out+="${out:+, }\"$f\": \"$lr\""
  done
  echo "{$out}"
}
branch_job() { awk -v a="$1" -v e="E=$2" '$2 == a && $3 == e {sub("branch=", "", $5); print $5}' "$X/LAUNCH.log" | tail -1; }
ARCHS=("$@"); [ ${#ARCHS[@]} -gt 0 ] || ARCHS=(mzansilm xlstm gdn mamba2)
for a in "${ARCHS[@]}"; do
  for E in 2 3 4; do
    [ -e "$F/runs/${a}_e$E/config.json" ] && continue  # already launched (rerun-safe)
    AFTER=$(branch_job "$a" "$E") FIXED_LRS=$(lrs_json "$a") MULTI_ONLY=1 SMOKE_FAMILIES="$FAMS" NICE=500 \
      bash "$F/code/fft_rollout/launch.sh" "$a" "$X/$a/e$E" 1 "${a}_e$E"
  done
  [ -e "$F/runs/${a}_e4_lrcheck/config.json" ] && continue
  AFTER=$(branch_job "$a" 4) FIXED_LRS=$(lrs_json "$a" 1 "ner sib") MULTI_ONLY=1 SMOKE_FAMILIES="ner sib" NICE=500 \
    bash "$F/code/fft_rollout/launch.sh" "$a" "$X/$a/e4" 1 "${a}_e4_lrcheck"
done
