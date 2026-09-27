#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/monomulti_rescore_20260924/slurm/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/monomulti_rescore_20260924/slurm/%x-%j.err
# env: ARCH TASK RUNNER(v2|bounded) POSLANG(for bounded) CHECKPOINT ADAPTER PROTOCOL OUTPUT
set -euo pipefail
R=/scratch/lmbanr001/masters/sallm/results/monomulti_rescore_20260924
bundle=/scratch/lmbanr001/masters/sallm_snapshots/general-sequence-official-test-20260916-v1
repair=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260917_v2/control
source_snapshot=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
bounded=/scratch/lmbanr001/masters/sallm_snapshots/gdn-mono-pos-official-20260920-v2/run_gdn_mono_pos.py
selection=/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection/SELECTION.json
release_root=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/release
v2=$repair/run_general_sequence_eval_20260917_v2.py
[[ "$(sha256sum $v2 | cut -d' ' -f1)" == 9cad0519cf65b0cd95c9f453b7ec441a68a3e89dde456e84a39856365fe026f5 ]]
(cd /scratch/lmbanr001/masters/sallm_snapshots/gdn-mono-pos-official-20260920-v2 && sha256sum -c ARTIFACTS.sha256 --quiet)
(cd "$source_snapshot" && sha256sum -c SNAPSHOT_MANIFEST.sha256 --quiet)
[[ "$(sha256sum $selection | cut -d' ' -f1)" == ce45c0b2bb290b0e4f084e4973eaa8df71fed1ad60f20e17b7be6a223f647994 ]]
python=/home/lmbanr001/masters/sallm/.venv/bin/python
[[ "$ARCH" == xlstm ]] && python=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
if [[ "$ARCH" == mamba2 && "$TASK" == ner ]]; then
  FIX=/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/bases/mamba2_eosfix
  (cd $FIX && sha256sum -c /scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/bases/mamba2_eosfix.FILES.sha256 --quiet)
fi
release="$release_root/$(printf '%s' "$TASK" | tr '[:lower:]' '[:upper:]')_TEST_ACCESS_RELEASED_V1.json"
export HF_HOME=/scratch/lmbanr001/hf-cache HF_DATASETS_CACHE=/scratch/lmbanr001/hf-cache/datasets HF_HUB_CACHE=/scratch/lmbanr001/hf-cache/hub
export HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda
export PYTHONPATH="$source_snapshot/src/main:$bundle:$repair"
nvidia-smi --query-gpu=name --format=csv,noheader
"$python" -c "import torch; n=torch.cuda.get_device_name(0); assert '${EXPECT_GPU:-L40S}' in n, n; print('GPU_OK', n)"
[[ ! -e "$OUTPUT" && ! -e "$OUTPUT.tmp" ]]
mkdir -p "$(dirname "$OUTPUT")"
date -Is
if [[ "$RUNNER" == bounded ]]; then
  "$python" "$bounded" --task pos --phase test --architecture "$ARCH" --language "$POSLANG" \
    --checkpoint "$CHECKPOINT" --adapter "$ADAPTER" --protocol "$PROTOCOL" --selection "$selection" --release "$release" --output "$OUTPUT"
elif [[ "$RUNNER" == v2bs1 ]]; then
  "$python" $R/scripts/run_general_sequence_eval_v2_bs1.py --task "$TASK" --phase test --architecture "$ARCH"     --checkpoint "$CHECKPOINT" --adapter "$ADAPTER" --protocol "$PROTOCOL" --selection "$selection" --release "$release" --output "$OUTPUT"
else
  "$python" "$v2" --task "$TASK" --phase test --architecture "$ARCH" \
    --checkpoint "$CHECKPOINT" --adapter "$ADAPTER" --protocol "$PROTOCOL" --selection "$selection" --release "$release" --output "$OUTPUT"
fi
date -Is
"$python" -c "
import json,sys; d=json.load(open('$OUTPUT')); print('UNIT_DONE', d['elapsed_seconds'], json.dumps(d['reported_metrics']))"
