#!/usr/bin/env bash
#SBATCH --account=maths
#SBATCH --partition=ada
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --mail-type=FAIL,END

set -euo pipefail
/scratch/slurm/bin/purequota

snapshot=/scratch/lmbanr001/masters/sallm_snapshots/general-prompt-correction-20260914-v1
result_root=/scratch/lmbanr001/masters/sallm/results/retained_standardized_20260913_v1/general_prompt_corrected_20260914_v1
python_bin=/home/lmbanr001/masters/sallm/.venv/bin/python

(cd "$snapshot" && sha256sum --check --strict SNAPSHOT_EXECUTION_MANIFEST.json.sha256)
"$python_bin" "$snapshot/scripts/create_execution_manifest.py" \
  --verify "$snapshot/SNAPSHOT_EXECUTION_MANIFEST.json" \
  --expected-repo-root "$snapshot"
[[ ! -e "$result_root/verified" ]]
export PYTHONPATH="$snapshot/src/main:$snapshot/scripts"
"$python_bin" "$snapshot/scripts/finalize_general_prompt_corrections.py" verify-test \
  --lm-dir "$result_root/raw/test/lm_eval" \
  --intent-dir "$result_root/raw/test/intent" \
  --selection "$result_root/selection/SELECTION.json" \
  --ready "$result_root/selection/READY_FOR_TEST.json" \
  --output-dir "$result_root/verified"
