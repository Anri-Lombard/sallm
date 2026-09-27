#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
# LANES="unitA+unitB;unitC" : lanes separated by ';' run concurrently on the one GPU, units within a lane ('+'-separated) sequentially. (No commas: sbatch --export splits on them.)
set -uo pipefail
root=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925
IFS=';' read -ra lanes <<< "${LANES:?}"
pids=()
i=0
for lane in "${lanes[@]}"; do
  (
    IFS='+' read -ra units <<< "$lane"
    for u in "${units[@]}"; do
      echo "[$(date -Is)] start $u"
      bash "$root/jobs/run_unit.sh" "$u" > "$root/logs/$u.${SLURM_JOB_ID}.log" 2>&1
      rc=$?
      echo "[$(date -Is)] end $u rc=$rc"
    done
  ) &
  pids+=($!)
  i=$((i+1))
  sleep 20
done
# periodic validation/test/prune pass inside this allocation (skipped while another pass holds the lock)
(
  while true; do
    sleep 1800
    lj=$(cut -d" " -f1 "$root/valpass.lock" 2>/dev/null)
    if [[ -n "$lj" ]] && squeue -h -j "$lj" -t R 2>/dev/null | grep -q .; then continue; fi
    bash "$root/jobs/valpass.sh" >> "$root/logs/valpass.job${SLURM_JOB_ID}.log" 2>&1
  done
) &
vp=$!
for p in "${pids[@]}"; do wait "$p"; done
echo ALL_LANES_DONE
kill "$vp" 2>/dev/null
while [[ -e "$root/valpass.lock" ]] && squeue -h -j "$(cut -d' ' -f1 "$root/valpass.lock")" -t R | grep -q . && [[ "$(cut -d' ' -f1 "$root/valpass.lock")" != "$SLURM_JOB_ID" ]]; do sleep 60; done
bash "$root/jobs/valpass.sh" >> "$root/logs/valpass.job${SLURM_JOB_ID}.log" 2>&1
echo FINAL_PASS_DONE
