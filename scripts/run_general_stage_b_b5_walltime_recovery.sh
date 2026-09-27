#!/bin/bash
#SBATCH --account=nlpgroup80
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup80
#SBATCH --gres=gpu:ampere80:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/home/lmbanr001/masters/sallm
#SBATCH --job-name=hpo-gdn-general-stage-b-b5-resume
#SBATCH --output=/scratch/lmbanr001/masters/sallm/logs/jobs/hpo-gdn-general-stage-b-b5-resume-%j.out
#SBATCH --mail-type=FAIL,END

set -euo pipefail

mode="${1:-run}"
[[ "$mode" == "run" || "$mode" == "--preflight-only" ]] || {
	echo "ERROR: optional argument must be --preflight-only." >&2
	exit 1
}
candidate=b5
snapshot="${SALLM_RECOVERY_SNAPSHOT:-$HOME/masters/sallm_snapshots/uniform-adapter-hpo-general-a0-resume-correction-20260828-45fec06b}"
runtime="${SALLM_RUNTIME_REPO:-$HOME/masters/sallm}"
runtime_python="${SALLM_RUNTIME_PYTHON:-$runtime/.venv/bin/python}"
recovery_bundle="${SALLM_RECOVERY_BUNDLE:?SALLM_RECOVERY_BUNDLE must be explicit}"
scratch="${SCRATCH:-/scratch/lmbanr001}"
output="$scratch/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/general/stage_b/b5/seed_42"
checkpoint="$output/checkpoint-10912"
archive="$scratch/masters/sallm/recovery_archives/general-stage-b/b5-seed42-pre-recovery.tar"
runtime_verifier="$recovery_bundle/verify_execution_runtime.py"

check_hash() {
	local path="$1"
	local expected="$2"
	[[ -f "$path" ]] || {
		echo "ERROR: missing $path" >&2
		exit 1
	}
	[[ "$(sha256sum "$path" | cut -d' ' -f1)" == "$expected" ]] || {
		echo "ERROR: hash mismatch for $path" >&2
		exit 1
	}
}

check_hash "$snapshot/scripts/run_pure_gdn_validation_trial.sh" \
	45fec06bf2ea71d4f69a43a686c88b9b8d895750d85b797ed104ac59271cceab
check_hash "$snapshot/scripts/create_execution_manifest.py" \
	55095f225dba0710b23831fb306ca3e471cbb1bbaaeec48529dd080fec235024
check_hash "$snapshot/deployment_manifest.json" \
	8b6a744bc43d402aedccb44e6d62ef5f34144b95c7e871bfd7e2be823d0d9bb6
check_hash "$snapshot/deployment_manifest.json.sha256" \
	4f5d47a49dabd4deaebe19a36aa8fe8c9a1e1f169e27ae2622e98cdc28c9a021
check_hash "$runtime_verifier" \
	cfe2eb873f53d83de2c2c296c6a172f74a8059ee4bd11f296785c444e9525b23
check_hash "$archive" \
	3f52518bf028a1a921ea7c4718654b5bd21e3f27115aa761c8a689dd92051595
check_hash "$output/execution_manifest.json" \
	662ed16cf8705aa230f94a721fa981eae8cf13d3965d212429cd837f551d6c19
[[ ! -w "$archive" ]] || {
	echo "ERROR: recovery archive is writable: $archive" >&2
	exit 1
}
[[ -x "$runtime_python" ]] || {
	echo "ERROR: missing runtime Python $runtime_python" >&2
	exit 1
}
"$runtime_python" "$snapshot/scripts/create_execution_manifest.py" \
	--verify "$snapshot/deployment_manifest.json"
"$runtime_python" "$runtime_verifier" "$output/execution_manifest.json"
check_hash "$checkpoint/adapter_model.bin" \
	252af47bbbb411a87ed3bc59bec8596a2845d8afc9c0cdfb093759be71434d5f
check_hash "$checkpoint/optimizer.pt" \
	7853c01504a5d823055b33eb31bd10400736a1750217f0074b29d2cf04018294
check_hash "$checkpoint/scheduler.pt" \
	b403be7999c394122e0db5d991abdd751bfe923ce60b4cb961d4b1de55a58413
check_hash "$checkpoint/rng_state.pth" \
	3a5fa191be89653519d53a698bb731547927ec65d27015072e0de1b7e71c27d3
check_hash "$checkpoint/trainer_state.json" \
	51a64b5a7c3f09296dd8cb7b61bd8f4ef54bb0202bae9e695b66fc8f9d695f70

[[ ! -e "$output/final_adapter/adapter_model.safetensors" ]] || {
	echo "ERROR: b5 already has a final adapter." >&2
	exit 1
}
compgen -G "$output/execution_manifest.resume-*.json" >/dev/null && {
	echo "ERROR: b5 already has a recovery manifest." >&2
	exit 1
}

if [[ "$mode" == "--preflight-only" ]]; then
	printf 'Recovery preflight verified: candidate=%s output=%s checkpoint=%s archive=%s\n' \
		"$candidate" "$output" "$checkpoint" "$archive"
	exit 0
fi

export SALLM_REPO_DIR="$snapshot"
export SALLM_RUNTIME_REPO="$runtime"
export SALLM_HPO_REGISTRY="$snapshot/src/conf/hpo/pure_gdn_enhanced_v1.json"
export SALLM_HPO_RESUME_FROM_CHECKPOINT="$checkpoint"

exec bash "$snapshot/scripts/run_validation_hpo_trial.sh" \
	pure_gdn general stage_b b5 42
