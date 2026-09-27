#!/bin/bash
# Head node: add K lanes to a run without touching its running lanes (bash + squeue/sbatch only).
#   [NICE=1000] [LANE_TIME=48:00:00] bash addlane.sh NAME K
# Reuses the smallest lane indices with no queued/running job and raises MAX_LANES to cover them.
set -euo pipefail
R=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926; NAME=$1; K=$2; T=${LANE_TIME:-48:00:00}
O=$R/runs/$NAME
[[ -e $O/config.json ]] || { echo "no run $O" >&2; exit 2; }
used=" $(squeue -u "$USER" -h -o %j | awk -v n="fft-$NAME-" 'index($1, n) == 1 {print substr($1, length(n) + 1)}' | tr '\n' ' ') "
cap=$(cat "$O/MAX_LANES" 2>/dev/null || echo 0)
h=${T%%:*}; m=${T#*:}; m=${m%%:*}; lh=$(( 10#$h * 60 + 10#$m - 10 ))
i=0; added=0
while (( added < K )); do
  if [[ "$used" != *" $i "* ]]; then
    (( i + 1 > cap )) && cap=$(( i + 1 )) && echo "$cap" > "$O/MAX_LANES"
    j=$(sbatch --parsable ${NICE:+--nice=$NICE} --time="$T" --job-name="fft-$NAME-$i" \
      --export="ALL,OUT=$O,LANE=$i,LANE_HOURS=$(( lh / 60 )).$(( (lh % 60) * 100 / 60 ))" "$R/code/fft_rollout/lane.sbatch")
    echo "$(date -Is) addlane $NAME lane $i -> job $j (MAX_LANES $cap)" | tee -a "$O/LAUNCH.log"
    added=$(( added + 1 ))
  fi
  i=$(( i + 1 ))
done
