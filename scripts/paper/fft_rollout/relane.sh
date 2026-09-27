#!/bin/bash
# Head node: restart a run's lanes after a code fix or a crash (bash + squeue/scancel/sbatch only).
#   [NICE=1000] [REPLAN=1] bash relane.sh NAME N [TIME]   cancel lanes fft-NAME-*, forget every unit that is not done (it reruns), submit N lanes
R=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926; NAME=$1; N=$2; T=${3:-48:00:00}
O=$R/runs/$NAME
ids=$(squeue -u $USER -h -o "%i %j" | awk -v n="fft-$NAME-" 'index($2, n) == 1 {print $1}'); [[ -z "$ids" ]] || scancel $ids
sleep 3
for f in $O/state/*.json; do grep -q '"state": "done"' $f || rm -f $f ${f%.json}.hb; done
# REPLAN=1: rebuild the DAG (units.json) from the current code on the next lane start; done units keep their state
[[ -n "${REPLAN:-}" ]] && mv $O/units.json $O/units.json.replaced$(date +%s)
h=${T%%:*}; m=${T#*:}; m=${m%%:*}; lh=$(( (10#$h*60 + 10#$m - 10) ))
for ((i=0;i<N;i++)); do sbatch --parsable ${NICE:+--nice=$NICE} --time=$T --job-name=fft-$NAME-$i --export=ALL,OUT=$O,LANE=$i,LANE_HOURS=$((lh/60)).$(( (lh%60)*100/60 )) $R/code/fft_rollout/lane.sbatch; done
echo "$N" > $O/MAX_LANES
