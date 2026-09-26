#!/bin/bash
#SBATCH --account=nlpgroup80
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup80
#SBATCH --gres=gpu:ampere80:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/home/lmbanr001/masters/sallm
#SBATCH --job-name=hpo-gdn-general-stage-b-resume
#SBATCH --output=/scratch/lmbanr001/masters/sallm/logs/jobs/hpo-gdn-general-stage-b-resume-%j.out
#SBATCH --mail-type=FAIL,END

set -euo pipefail

candidate="${1:?candidate must be b1, b2, or b3}"
mode="${2:-run}"
[[ "$mode" == "run" || "$mode" == "--preflight-only" ]] || {
	echo "ERROR: optional second argument must be --preflight-only." >&2
	exit 1
}
snapshot="${SALLM_RECOVERY_SNAPSHOT:-$HOME/masters/sallm_snapshots/uniform-adapter-hpo-general-a0-resume-correction-20260828-45fec06b}"
runtime="${SALLM_RUNTIME_REPO:-$HOME/masters/sallm}"
runtime_python="${SALLM_RUNTIME_PYTHON:-$runtime/.venv/bin/python}"
recovery_bundle="${SALLM_RECOVERY_BUNDLE:-$(cd "$(dirname "$0")" && pwd)}"
scratch="${SCRATCH:-/scratch/lmbanr001}"
root="$scratch/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/general/stage_b"
output="$root/$candidate/seed_42"
checkpoint="$output/checkpoint-10912"
archive="$scratch/masters/sallm/recovery_archives/general-stage-b/$candidate-seed42-pre-recovery.tar"
runtime_verifier="$recovery_bundle/verify_execution_runtime.py"

case "$candidate" in
b1)
	expected_adapter=29aa21d2237b40efffa70f719a364d92addd03042eb3b1dffe8d88a1d12819f0
	expected_optimizer=8557158f6b2f9580089f6844f4a5b2cc0210cd9e30d5a39d1b69ed700f484f6e
	expected_scheduler=58d820ba2be4b96b51823238bab995b391147e7c05fc0223f29f94d6a6473407
	expected_rng=0d4e900b49e99f73ab4c969265a4600f448e5236d37bed123a9baa3915e9013c
	expected_trainer=2b5d9b74c8c8ba71ac86bd79dd216edabb6e98316d299944c79fcab5bbabade0
	expected_archive=0c972a36ed854b27e2512350ce947d22581a145a648b9919b2959ef4df736c27
	expected_manifest=ecb840c3aa6ba93a69daa8083b4550f4e4aafbaf1aa06fbde4faf175b96bb404
	;;
b2)
	expected_adapter=cc2a37683387dfc2ebb4969ab8337ce24f42bdf372d568be628545b3f34aedcb
	expected_optimizer=5a42ee1ef504c3a4e43990f04ad983142debf06691bde0f7ca1d2a6808bcaac8
	expected_scheduler=8ad971f1950fc2d8cbce77d99f29c9806ee058be8ba844f45d74d9a63fed6707
	expected_rng=3180a03567b7f40c2f694e184313fd4ca7212fda830074f2e9bc3350b97907e5
	expected_trainer=fe862cfbfc8430edd220ec888a7b4ed3dab4151f26e26f0aa090d9341314c804
	expected_archive=c82dc55f3ed836759015f68a9851318fc1bdd2e1584b74081cf0b45362705e96
	expected_manifest=32371bdd0c4f9a7b571a70b369cc6f1dccb9ac2665be4489b2d7be05dcec274e
	;;
b3)
	expected_adapter=e591c605336ea0fbc431199b59344e768b3242a56ad0f5e5ebe8bd3a201adfb5
	expected_optimizer=98c23a9b4f6debdfe49bc5fdd64ce0388d209354f470ce2957828fca1149edd3
	expected_scheduler=2244542dfc73c540f56c4397f999c55d9557aadb704e404ac6d3861dcd996027
	expected_rng=47a82e7665a2fa0070b6758de26b3cb3469af4e8573bfae25bf22006f339fc26
	expected_trainer=968e989e31e8f9248957eb90b7adaff293aba12d3c3cfea06b19f50c66d11031
	expected_archive=a51c35ead0e3e794be9209728c12ed806e7471a18ad7d0334aeb5c5b88f3d918
	expected_manifest=1344f1c7d2ea38b1f90ffc0986e49e8296eaa44f8f3668b6029fea1ea4255bdf
	;;
*)
	echo "ERROR: candidate must be b1, b2, or b3." >&2
	exit 1
	;;
esac

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
check_hash "$archive" "$expected_archive"
check_hash "$output/execution_manifest.json" "$expected_manifest"
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
check_hash "$checkpoint/adapter_model.bin" "$expected_adapter"
check_hash "$checkpoint/optimizer.pt" "$expected_optimizer"
check_hash "$checkpoint/scheduler.pt" "$expected_scheduler"
check_hash "$checkpoint/rng_state.pth" "$expected_rng"
check_hash "$checkpoint/trainer_state.json" "$expected_trainer"

[[ ! -e "$output/final_adapter/adapter_model.safetensors" ]] || {
	echo "ERROR: $candidate already has a final adapter." >&2
	exit 1
}
compgen -G "$output/execution_manifest.resume-*.json" >/dev/null && {
	echo "ERROR: $candidate already has a recovery manifest." >&2
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
	pure_gdn general stage_b "$candidate" 42
