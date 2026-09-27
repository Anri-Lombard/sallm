#!/bin/bash
#SBATCH --job-name=general-test-integrate-v2
#SBATCH --account=maths
#SBATCH --partition=ada
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260917_v2/slurm/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260917_v2/slurm/%x-%j.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail
umask 077

bundle=/scratch/lmbanr001/masters/sallm_snapshots/general-sequence-official-test-20260916-v1
repair=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260917_v2/control
selection_root=/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection
result_root=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260917_v2
pos_root=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/raw/pos
release_root=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/release
integration_root="$result_root/integration"
protocol="$bundle/general_sequence_validation_hex_protocol_20260915_v3.json"
integrator="$repair/integrate_general_sequence_official_test_20260917_v2.py"
integrator_sha=35b822e7e33a273101a6f6640f3ce2e854e777812558d4c579103c5cf92fe79f
repair_amendment="$repair/GENERAL_NER_BOS_REPAIR_V2.json"
repair_amendment_sha=8e443261ec5461805c907508a86267664ad8920d91acad78316001126e8b5e6b
python=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python

(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
[[ "$(sha256sum "$integrator" | cut -d' ' -f1)" == "$integrator_sha" ]]
[[ "$(sha256sum "$repair_amendment" | cut -d' ' -f1)" == "$repair_amendment_sha" ]]
(cd "$selection_root" && sha256sum -c ARTIFACTS.sha256 --quiet)
(cd "$release_root" && sha256sum -c ARTIFACTS.sha256 --quiet)
[[ ! -e "$integration_root" ]]
export PYTHONPATH="$bundle:$repair"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/general-test-integration-pycache-${SLURM_JOB_ID}"
"$python" "$integrator" \
  --ner-dir "$result_root/raw/ner" \
  --pos-dir "$pos_root" \
  --selection "$selection_root/SELECTION.json" \
  --ready "$release_root/READY_FOR_TEST.json" \
  --ner-release "$release_root/NER_TEST_ACCESS_RELEASED_V1.json" \
  --pos-release "$release_root/POS_TEST_ACCESS_RELEASED_V1.json" \
  --protocol "$protocol" \
  --output-dir "$integration_root"
(cd "$integration_root" && sha256sum -c ARTIFACTS.sha256 --quiet)
