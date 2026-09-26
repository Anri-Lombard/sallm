#!/bin/bash
# usage (from login node): overlap_lane.sh TAG unit1+unit2+...  -- run inside an existing allocation via srun --overlap
root=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925
tag="$1"; IFS='+' read -ra units <<< "$2"
for u in "${units[@]}"; do
  echo "[$(date -Is)] start $u host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-} job=${SLURM_JOB_ID:-}"
  bash "$root/jobs/run_unit.sh" "$u" > "$root/logs/$u.$tag.log" 2>&1
  rc=$?
  echo "[$(date -Is)] end $u rc=$rc"
done
echo LANE_DONE
