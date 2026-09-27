#!/bin/bash
#SBATCH --job-name=general-test-release
#SBATCH --account=maths
#SBATCH --partition=ada
#SBATCH --qos=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/slurm/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/slurm/%x-%j.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail
umask 077

bundle=/scratch/lmbanr001/masters/sallm_snapshots/general-sequence-official-test-20260916-v1
selection_root=/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection
result_root=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1
release_root="$result_root/release"
protocol="$bundle/general_sequence_validation_hex_protocol_20260915_v3.json"
python=/home/lmbanr001/masters/sallm/.venv/bin/python
protocol_sha=2041716c9011823cc356e95f74d3c49b78c052238229b424a05c4d38ed1a0176

(cd "$bundle" && sha256sum -c ARTIFACTS.sha256 --quiet)
[[ "$(sha256sum "$protocol" | cut -d' ' -f1)" == "$protocol_sha" ]]
(cd "$selection_root" && sha256sum -c ARTIFACTS.sha256 --quiet)
[[ ! -e "$release_root" ]]
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/general-test-release-pycache-${SLURM_JOB_ID}"
"$python" "$bundle/prepare_general_sequence_test_release_20260916_v1.py" \
  --selection-dir "$selection_root" \
  --protocol "$protocol" \
  --output-dir "$release_root"
(cd "$release_root" && sha256sum -c ARTIFACTS.sha256 --quiet)
