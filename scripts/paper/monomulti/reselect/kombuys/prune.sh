#!/bin/bash
# usage: prune.sh <label> [--yes]
# After selection.json and test/<label>.json exist: copy every checkpoint's trainer_state.json to val/<label>/trainer_states/,
# then delete the non-selected checkpoint-*/ dirs (and final_adapter weights if they differ from the selected checkpoint).
# Dry run (prints the plan) unless --yes. See kit/prune.py.
set -euo pipefail
root=/scratch/alombard/sallm/results/monomulti_reselect_20260925
export PYTHONDONTWRITEBYTECODE=1
cd "$root/jobs/kit"
/scratch/alombard/sallm/.venv/bin/python prune.py "$@"
