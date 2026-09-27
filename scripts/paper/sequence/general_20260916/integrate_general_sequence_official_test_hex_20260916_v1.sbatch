#!/bin/bash
#SBATCH --job-name=general-test-integrate
#SBATCH --account=maths
#SBATCH --partition=ada
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/slurm/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/slurm/%x-%j.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail
umask 077

bundle=/scratch/lmbanr001/masters/sallm_snapshots/general-sequence-official-test-20260916-v1
selection_root=/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection
result_root=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1
release_root="$result_root/release"
integration_root="$result_root/integration"
protocol="$bundle/general_sequence_validation_hex_protocol_20260915_v3.json"
python=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python

(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
(cd "$selection_root" && sha256sum -c ARTIFACTS.sha256 --quiet)
(cd "$release_root" && sha256sum -c ARTIFACTS.sha256 --quiet)
[[ ! -e "$integration_root" ]]
export PYTHONPATH="$bundle"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/general-test-integration-pycache-${SLURM_JOB_ID}"
"$python" "$bundle/integrate_general_sequence_official_test_20260916_v1.py" \
  --ner-dir "$result_root/raw/ner" \
  --pos-dir "$result_root/raw/pos" \
  --selection "$selection_root/SELECTION.json" \
  --ready "$release_root/READY_FOR_TEST.json" \
  --ner-release "$release_root/NER_TEST_ACCESS_RELEASED_V1.json" \
  --pos-release "$release_root/POS_TEST_ACCESS_RELEASED_V1.json" \
  --protocol "$protocol" \
  --output-dir "$integration_root"
(cd "$integration_root" && sha256sum -c ARTIFACTS.sha256 --quiet)
