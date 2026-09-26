#!/bin/bash
set -euo pipefail

/scratch/slurm/bin/purequota

snapshot="${SALLM_RUNTIME_GATE_SNAPSHOT:?missing immutable snapshot}"
manifest="${SALLM_RUNTIME_GATE_MANIFEST:?missing execution manifest}"
checkpoint="${SALLM_RUNTIME_GATE_CHECKPOINT:?missing pure-GDN checkpoint}"
result_root="${SALLM_RUNTIME_GATE_RESULT_ROOT:?missing result root}"
python_bin="${SALLM_RUNTIME_PYTHON:-$HOME/masters/sallm/.venv/bin/python}"

[[ "$snapshot" == "$HOME"/masters/sallm_snapshots/* ]]
[[ "$result_root" == /scratch/* ]]
[[ "$checkpoint" == /scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model ]]
[[ -d "$checkpoint" ]]
[[ ! -e "$result_root/runtime_gate.json" ]]

mkdir -p "$result_root"
export PYTHONPATH="$snapshot/src/main"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

"$python_bin" "$snapshot/scripts/create_execution_manifest.py" \
  --verify "$manifest" \
  --verify-runtime \
  --expected-repo-root "$snapshot" \
  --expected-artifact-root "$checkpoint"
"$python_bin" "$snapshot/scripts/verify_pure_gdn_runtime.py" \
  --mode post-checkpoint \
  --config "$snapshot/src/conf/base/gated_deltanet_125m_pure.yaml" \
  --checkpoint "$checkpoint" \
  | tee "$result_root/runtime_gate.json"
shasum -a 256 "$result_root/runtime_gate.json" \
  > "$result_root/runtime_gate.json.sha256"
chmod 0444 "$result_root/runtime_gate.json" "$result_root/runtime_gate.json.sha256"
