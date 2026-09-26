#!/bin/bash
#SBATCH --job-name=gdn-recovery-v2-afrihg-continue
#SBATCH --account=nlpgroup80
#SBATCH --partition=a100
#SBATCH --qos=nlpgroup80
#SBATCH --gres=gpu:ampere80:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --chdir=/home/lmbanr001/masters/sallm
#SBATCH --output=/scratch/lmbanr001/masters/sallm/logs/official_test/gdn-recovery-v2-afrihg-continue-%j.log

set -euo pipefail

/scratch/slurm/bin/purequota

canonical=/scratch/lmbanr001/masters/sallm/data/official_recovery_cache/20260903_v2
runtime=/scratch/lmbanr001/masters/sallm/data/official_recovery_runtime_cache/20260903_v2/afrihg_continuation_${SLURM_JOB_ID}
overlay=/scratch/lmbanr001/masters/sallm/overlays/pure-gdn-afrihg-recovery-verifier-correction-20260903-2d58e73a

[[ ! -e "$runtime" ]]
(
  cd "$canonical"
  shasum -a 256 -c cache_tree_sha256.txt
)
cp -a "$canonical" "$runtime"
(
  cd "$runtime"
  shasum -a 256 -c cache_tree_sha256.txt
)
chmod -R u+w "$runtime"

export SALLM_OFFICIAL_SNAPSHOT=/home/lmbanr001/masters/sallm_snapshots/pure-gdn-heldout-recovery-20260903-v2-46d8cf11
export SALLM_OFFICIAL_SOURCE_MANIFEST=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_recovery_20260903_v2/common/source_manifest.json
export SALLM_OFFICIAL_BASE_MANIFEST=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_recovery_20260903_v2/common/base_manifest.json
export SALLM_OFFICIAL_DATASET_MANIFEST=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_recovery_20260903_v2/afrihg/dataset_manifest.json
export SALLM_OFFICIAL_BASE=/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model
export SALLM_OFFICIAL_DATASET_ROOT=/home/lmbanr001/masters/sallm/data/afrihg_cache
export SALLM_OFFICIAL_MULTI_ADAPTER=/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/afrihg/stage_b/b7/seed_42/final_adapter
export SALLM_OFFICIAL_MULTI_ADAPTER_MANIFEST=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_recovery_20260903_v2/afrihg/multi_manifest.json
export SALLM_OFFICIAL_MONO_XHO=/scratch/lmbanr001/masters/sallm/checkpoints/pure_gdn_mono_familywise_v1/afrihg/xho/seed_42/final_adapter
export SALLM_OFFICIAL_MONO_XHO_MANIFEST=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_recovery_20260903_v2/afrihg/mono_xho_manifest.json
export SALLM_OFFICIAL_MONO_ZUL=/scratch/lmbanr001/masters/sallm/checkpoints/pure_gdn_mono_familywise_v1/afrihg/zul/seed_42/final_adapter
export SALLM_OFFICIAL_MONO_ZUL_MANIFEST=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_recovery_20260903_v2/afrihg/mono_zul_manifest.json
export SALLM_OFFICIAL_RESULT_ROOT=/scratch/lmbanr001/masters/sallm/results/official_test/familywise_recovery_20260903_v2/afrihg
export SALLM_OFFICIAL_PROTOCOL_ROOT=/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_recovery_20260903_v2/afrihg
export SALLM_OFFICIAL_RECOVERY_ID=pure-gdn-heldout-recovery-20260903-v2
export SALLM_OFFICIAL_RUNTIME_CACHE="$runtime"
export SALLM_OFFICIAL_CONTINUE_AFTER_VERIFIER_FIX=1
export SALLM_OFFICIAL_CORRECTION_ROOT="$overlay"
export SALLM_OFFICIAL_GENERATION_VERIFIER="$overlay/verify_official_generation_eval.py"
export SALLM_RUNTIME_PYTHON=/home/lmbanr001/masters/sallm/.venv/bin/python

exec bash "$overlay/run_pure_gdn_afrihg_official_test.sh"
