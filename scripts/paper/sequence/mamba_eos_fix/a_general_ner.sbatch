#!/bin/bash
#SBATCH --job-name=eosfix-gen-ner
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/logs/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/logs/%x-%j.err
set -euo pipefail
source /scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/sb/common.sh
bundle=/scratch/lmbanr001/masters/sallm_snapshots/general-sequence-official-test-20260916-v1
repair=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260917_v2/control
source_snapshot=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
runner=$repair/run_general_sequence_eval_20260917_v2.py
[[ "$(sha256sum $runner | cut -d' ' -f1)" == 9cad0519cf65b0cd95c9f453b7ec441a68a3e89dde456e84a39856365fe026f5 ]]
(cd "$source_snapshot" && sha256sum -c SNAPSHOT_MANIFEST.sha256 --quiet)
protocol=$R/protocols/general_ner_protocol_eosfix.json
selection=/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection/SELECTION.json
release=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/release/NER_TEST_ACCESS_RELEASED_V1.json
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export PYTHONPATH="$source_snapshot/src/main:$bundle:$repair"
"$PY" "$runner" --self-check
adapter="$(jq -r .models.mamba2.adapter_path $protocol)"
out=$R/general_ner/raw/ner/mamba2.json
mkdir -p "$(dirname $out)"
"$PY" "$runner" --task ner --phase test --architecture mamba2 --checkpoint "$FIX" --adapter "$adapter" \
  --protocol "$protocol" --selection "$selection" --release "$release" --output "$out"
"$PY" - "$out" "$protocol" "$selection" <<'PY'
import json, sys
from pathlib import Path
from finalize_general_sequence_replacement_20260914 import sha256, verify_common, verify_ner
p, proto, sel = map(Path, sys.argv[1:4])
payload = json.loads(p.read_text()); protocol = json.loads(proto.read_text())
verify_common(payload, protocol, sha256(proto), "mamba2", "ner", "test", sha256(sel))
scores, ids = verify_ner(payload, protocol, "test")
print("VERIFY_OK", len(scores), len(payload["rows"]), sha256(p), payload["reported_metrics"])
PY
date -Is
