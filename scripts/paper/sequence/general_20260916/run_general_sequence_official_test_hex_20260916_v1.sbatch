#!/bin/bash
#SBATCH --job-name=general-official-test
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup
#SBATCH --gres=gpu:ampere:1
#SBATCH --array=0-7%3
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/slurm/%x-%A_%a.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail
umask 077

bundle=/scratch/lmbanr001/masters/sallm_snapshots/general-sequence-official-test-20260916-v1
source_snapshot=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
selection_root=/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection
result_root=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1
release_root="$result_root/release"
protocol="$bundle/general_sequence_validation_hex_protocol_20260915_v3.json"
runner="$bundle/run_general_sequence_test_only_20260916_v1.py"
default_python=/home/lmbanr001/masters/sallm/.venv/bin/python
xlstm_python=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
xlstm_runtime_manifest=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/runtime_manifest.json
xlstm_backend=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/lib/python3.12/site-packages/transformers/models/xlstm/modeling_xlstm.py
xlstm_runtime_manifest_sha=9f30eeb73d415bd8be3535b0f74689c5d53d04f622e0e3ddea04a8c5a2df24ab
xlstm_backend_sha=5f20208d6334bf79cd2b897c5d2d15ea76a3e10f6b0d3602824fc12da8bdf525
source_manifest_sha=f2171f80edc53767937b1d714f41abd5dbea210c46e644aecbea6085da68996b
protocol_sha=2041716c9011823cc356e95f74d3c49b78c052238229b424a05c4d38ed1a0176
general_protocol="$source_snapshot/.audit/downstream-standardized-protocol-v1-20260913.json"
general_protocol_sha=61f9639c474f0955f92ff1f6779e483611f9b41495a842afed587ddf7100d6c1
general_amendment="$source_snapshot/.audit/downstream-standardized-protocol-amendment-20260913.json"
general_amendment_sha=3a93d206c45c79ba05e3551343ed9d21972ea701014078d397f546b4faf0ff2a
mamba_training_marker=/scratch/lmbanr001/masters/sallm/checkpoints/ft_mamba_125m_sa_general_sixfamily_tokenbalanced_r1_kombuys_verified/TRAINING_VERIFIED.json
mamba_training_marker_sha=7ffb0db5b4fa6d208b7d03546b581b6f5086b8b977322d95c4d291bcd4487921

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
[[ "$task_id" =~ ^[0-7]$ ]]
architectures=(mzansilm mzansilm mamba2 mamba2 xlstm xlstm gdn gdn)
tasks=(ner pos ner pos ner pos ner pos)
architecture="${architectures[$task_id]}"
task="${tasks[$task_id]}"
python="$default_python"
if [[ "$architecture" == xlstm ]]; then
  python="$xlstm_python"
fi

[[ -x "$python" ]]
(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
sha256sum -c "$bundle/general_sequence_validation_hex_home_runtime_20260914_v2.sha256" --quiet
(cd "$source_snapshot" && sha256sum -c SNAPSHOT_MANIFEST.sha256 --quiet)
(cd "$selection_root" && sha256sum -c ARTIFACTS.sha256 --quiet)
(cd "$release_root" && sha256sum -c ARTIFACTS.sha256 --quiet)
[[ "$(sha256sum "$protocol" | cut -d' ' -f1)" == "$protocol_sha" ]]
[[ "$(sha256sum "$source_snapshot/SNAPSHOT_MANIFEST.sha256" | cut -d' ' -f1)" == "$source_manifest_sha" ]]
[[ "$(sha256sum "$general_protocol" | cut -d' ' -f1)" == "$general_protocol_sha" ]]
[[ "$(sha256sum "$general_amendment" | cut -d' ' -f1)" == "$general_amendment_sha" ]]
[[ "$(sha256sum "$mamba_training_marker" | cut -d' ' -f1)" == "$mamba_training_marker_sha" ]]
if [[ "$architecture" == xlstm ]]; then
  [[ "$(sha256sum "$xlstm_runtime_manifest" | cut -d' ' -f1)" == "$xlstm_runtime_manifest_sha" ]]
  [[ "$(sha256sum "$xlstm_backend" | cut -d' ' -f1)" == "$xlstm_backend_sha" ]]
fi

selection="$selection_root/SELECTION.json"
ready="$release_root/READY_FOR_TEST.json"
release="$release_root/$(printf '%s' "$task" | tr '[:lower:]' '[:upper:]')_TEST_ACCESS_RELEASED_V1.json"
selection_sha="$(sha256sum "$selection" | cut -d' ' -f1)"
[[ "$(jq -r .selection_sha256 "$ready")" == "$selection_sha" ]]
[[ "$(jq -r .selection_sha256 "$release")" == "$selection_sha" ]]
[[ "$(jq -r .authorized "$release")" == true ]]
[[ "$(jq -r .one_time_test "$release")" == true ]]
[[ "$(jq -r .no_score_based_retry "$release")" == true ]]

export HF_HOME=/scratch/lmbanr001/hf-cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/general-official-test-pycache-${SLURM_JOB_ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export FLA_DISABLE_BACKEND_DISPATCH=1
export MAMBA_SCAN_IMPL=cuda
export PYTHONPATH="$source_snapshot/src/main:$bundle"

"$python" - "$architecture" "$task" <<'PY'
import sys
import torch
architecture, task = sys.argv[1:]
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one visible GPU, found {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
if "A100" not in name:
    raise SystemExit(f"expected A100, found {name}")
print(f"GPU_PREFLIGHT_OK architecture={architecture} task={task} gpu={name}")
PY
"$python" "$runner" --self-check

checkpoint="$("$python" - "$protocol" "$architecture" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["models"][sys.argv[2]]["base_path"])
PY
)"
adapter="$("$python" - "$protocol" "$architecture" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["models"][sys.argv[2]]["adapter_path"])
PY
)"

mkdir -p "$result_root/raw/$task"
output="$result_root/raw/$task/$architecture.json"
[[ ! -e "$output" && ! -e "$output.tmp" ]]
"$python" "$runner" \
  --task "$task" \
  --phase test \
  --architecture "$architecture" \
  --checkpoint "$checkpoint" \
  --adapter "$adapter" \
  --protocol "$protocol" \
  --selection "$selection" \
  --release "$release" \
  --output "$output"

"$python" - "$output" "$protocol" "$selection" "$architecture" "$task" <<'PY'
import json, sys
from pathlib import Path
from finalize_general_sequence_replacement_20260914 import (
    sha256, verify_common, verify_ner, verify_pos,
)
payload_path, protocol_path, selection_path = map(Path, sys.argv[1:4])
architecture, task = sys.argv[4:]
payload = json.loads(payload_path.read_text())
protocol = json.loads(protocol_path.read_text())
verify_common(
    payload, protocol, sha256(protocol_path), architecture, task, "test",
    sha256(selection_path),
)
scores, identities = (verify_ner if task == "ner" else verify_pos)(
    payload, protocol, "test"
)
if len(scores) != 3 or len(identities) != 3 or len(payload.get("rows", [])) <= 0:
    raise SystemExit("official test coverage mismatch")
print(
    f"GENERAL_OFFICIAL_TEST_UNIT_OK architecture={architecture} task={task} "
    f"scores={len(scores)} rows={len(payload['rows'])} sha256={sha256(payload_path)}"
)
PY
