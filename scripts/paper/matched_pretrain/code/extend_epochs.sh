#!/bin/bash
# Epoch extension of the matched WSD runs (decided 1 Oct 2026). From each 1-epoch run's pre-decay checkpoint
# (stable_step005777, optimizer included) the constant-LR trunk continues in segments that stop at the decay start of a
# 2-, 3- and 4-epoch budget (--stop-before-decay), and each segment's checkpoint branches a (1-sqrt) decay over the last
# 20% of that budget, giving finished 2-, 3- and 4-epoch models. Epochs >= 2 read the same seeded block permutation for
# all architectures (Blocks.phys). Stopping rule (fixed before any extension result): continue past epoch E unless every
# architecture's clean held-out bits per byte at E is worse than at E-1 or improves by less than 0.5%; applied to all four
# together, every epoch reported for every model. Cancel the remaining jobs by ID if the rule stops it.
# NGPU=1 for single-GPU jobs (same global batch via accumulation; easier to schedule on a busy cluster).
# HEX login node (sbatch only): [AFTER=<jobid>] [NGPU=1] bash extend_epochs.sh [ARCH ...]   (default: all four)
set -euo pipefail
R=/scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926
X=$R/runs/ext
declare -A LR=([mzansilm]=0.003 [xlstm]=0.003 [gdn]=0.003 [mamba2]=0.006) MB=([mzansilm]=12 [xlstm]=12 [gdn]=24 [mamba2]=24)
# Decay start of an E-epoch budget, as pretrain.py computes it: total = 1386480*E // 192; start = total - round(0.2*total).
TRUNK_SB=${TRUNK_SB:---gres=gpu:l40s:${NGPU:-2}}; BRANCH_SB=${BRANCH_SB:-$TRUNK_SB}
# per-architecture flags of the original 1-epoch run (xLSTM: eager, TFLA chunkwise kernel; see runs.csv / logs)
declare -A XA=([xlstm]="--no-compile --xlstm-kernel chunkwise--triton_xl_chunk")
declare -A START=([1]=5777 [2]=11554 [3]=17330 [4]=23108)
sub() {  # name dep out arch [args...]; TRUNK_SB / BRANCH_SB: extra sbatch flags (partition, account, gres) per job kind
  local sb=$TRUNK_SB; [[ $1 == *-e[0-9] ]] && sb=$BRANCH_SB
  sbatch --parsable $sb --job-name="mp-ext-$1" ${2:+--dependency=afterok:$2} --export=ALL,OUT_DIR="$3" \
    "$R/code/full.sbatch" "$4" "${LR[$4]}" "${MB[$4]}" "${@:5}"
}
mkdir -p "$X"
ARCHS=("$@"); [ ${#ARCHS[@]} -gt 0 ] || ARCHS=(mzansilm xlstm gdn mamba2)
for a in "${ARCHS[@]}"; do
  ck=$R/runs/full_${a}_wsd_lr${LR[$a]}/stable_step$(printf %06d "${START[1]}")
  [ -f "$ck/state.pt" ] || { echo "missing $ck" >&2; exit 2; }
  dep="${AFTER:-}"  # e.g. AFTER=<smoke job id>: the first trunk segment waits for it to succeed
  for E in 2 3 4; do
    t=$X/$a/trunk_e$E
    seg=$(sub "$a-t$E" "$dep" "$t" "$a" --budget-epochs "$E" --stop-before-decay --resume "$ck" ${XA[$a]:-})
    ck=$t/stable_step$(printf %06d "${START[$E]}")
    br=$(sub "$a-e$E" "$seg" "$X/$a/e$E" "$a" --budget-epochs "$E" --resume "$ck" ${XA[$a]:-})
    echo "$(date -Is) $a E=$E trunk=$seg branch=$br" | tee -a "$X/LAUNCH.log"
    dep=$seg
  done
done
