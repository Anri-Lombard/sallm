#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --qos=l40sfree
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
# Runs eval_unit.sbatch for each line of $UNITS (tsv: name ARCH TASK RUNNER POSLANG CHECKPOINT ADAPTER PROTOCOL OUTPUT PHASE SEQ_BATCH1 SEQ_VALIDATION_PROMPTS SEQ_LANGUAGES), $PAR at a time, on one GPU.
set -uo pipefail
R=/scratch/lmbanr001/masters/sallm/results/monomulti_retrain_20260924
PAR=${PAR:-3}
run_line() {
  IFS=$'\t' read -r name ARCH TASK RUNNER POSLANG CHECKPOINT ADAPTER PROTOCOL OUTPUT PHASE SEQ_BATCH1 SEQ_VALIDATION_PROMPTS SEQ_LANGUAGES <<< "$1"
  [[ "$SEQ_LANGUAGES" == "-" ]] && SEQ_LANGUAGES=""
  [[ "$SEQ_VALIDATION_PROMPTS" == "-" ]] && SEQ_VALIDATION_PROMPTS=""
  [[ -e "$OUTPUT" ]] && { echo "SKIP $name"; return 0; }
  ARCH=$ARCH TASK=$TASK RUNNER=$RUNNER POSLANG=$POSLANG CHECKPOINT=$CHECKPOINT ADAPTER=$ADAPTER PROTOCOL=$PROTOCOL OUTPUT=$OUTPUT \
    PHASE=$PHASE SEQ_BATCH1=$SEQ_BATCH1 SEQ_VALIDATION_PROMPTS=$SEQ_VALIDATION_PROMPTS SEQ_LANGUAGES=$SEQ_LANGUAGES \
    bash "$R/jobs/eval_unit.sbatch" > "$R/logs/eval-$name-${SLURM_JOB_ID}.out" 2>&1
  echo "END $name rc=$? $(grep UNIT_DONE "$R/logs/eval-$name-${SLURM_JOB_ID}.out" | cut -c1-200)"
}
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  while (( $(jobs -rp | wc -l) >= PAR )); do sleep 20; done
  run_line "$line" &
  sleep 30
done < "${UNITS:?}"
wait
echo EVAL_MULTI_DONE
