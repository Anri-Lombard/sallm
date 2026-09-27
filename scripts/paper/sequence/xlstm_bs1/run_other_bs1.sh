#!/bin/bash
# usage: run_other_bs1.sh general_ner|base_ner|base_afrimgsm|general_afrimgsm
set -uo pipefail
here=/scratch/lmbanr001/masters/sallm_snapshots/xlstm-bs1-other-20260925
R=/scratch/lmbanr001/masters/sallm/results/xlstm_bs1_other_20260925
bundle=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1
seqbundle=/scratch/lmbanr001/masters/sallm_snapshots/general-sequence-official-test-20260916-v1
repair=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260917_v2/control
source=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
fm=/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1
selection=/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection/SELECTION.json
kb=/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/kombuys_snapshot
python=/scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
export HF_HOME=/scratch/lmbanr001/hf-cache HF_DATASETS_CACHE=/scratch/lmbanr001/hf-cache/datasets HF_HUB_CACHE=/scratch/lmbanr001/hf-cache/hub
export HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda
mkdir -p $R/logs
what=$1
start=$(date +%s)
echo "START $what $(date -Is) job=${SLURM_JOB_ID:-}" >> $R/status.txt
case $what in
  general_ner)
    protocol=$seqbundle/general_sequence_validation_hex_protocol_20260915_v3.json
    mkdir -p $R/general_ner
    PYTHONPATH="$source/src/main:$seqbundle:$repair" "$python" $here/run_general_sequence_eval_20260917_v2_bs1.py --task ner --phase test \
      --architecture xlstm --checkpoint "$(jq -r .models.xlstm.base_path $protocol)" --adapter "$(jq -r .models.xlstm.adapter_path $protocol)" \
      --protocol $protocol --selection $selection \
      --release /scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/release/NER_TEST_ACCESS_RELEASED_V1.json \
      --output $R/general_ner/xlstm.json ;;
  base_ner)
    protocol=$fm/base-mamba-xlstm-20260924/sequence_protocols/u013-base-xlstm-ner.json
    mkdir -p $R/base_ner
    PYTHONPATH="$source/src/main:$bundle" "$python" $here/run_general_sequence_eval_20260914_bs1.py --task ner --phase test \
      --architecture xlstm --checkpoint "$(jq -r .models.xlstm.base_path $protocol)" --protocol $protocol --selection $selection \
      --release $fm/partial-ready-1r1/release/NER_TEST_ACCESS_RELEASED.json --output $R/base_ner/13.json ;;
  base_afrimgsm)
    mkdir -p $R/base_afrimgsm
    PYTHONPATH="$source/src/main:$bundle" "$python" $here/run_lm_eval_unit_bs1.py --mode official --index 15 \
      --inventory $here/u015_inventory --bindings $fm/base-mamba-xlstm-20260924/bindings/BINDINGS.json --source $source \
      --release $fm/base-mamba-xlstm-20260924/release/READY_FOR_OFFICIAL.json --output $R/base_afrimgsm/15.json ;;
  general_afrimgsm)
    b=$kb/general-prompt-official-kombuys-20260915-v1; s=$kb/general-prompt-correction-20260915-v8-sixfamily-tso-all-interface-fix
    protocol=$b/general_prompt_official_kombuys_20260915_protocol_v1.json
    mkdir -p $R/general_afrimgsm
    PYTHONPATH="/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/sb:$b:$s/scripts:$s/src/main" "$python" \
      /scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/sb/general_afrimgsm_wrapper.py --phase test --architecture xlstm \
      --checkpoint /scratch/lmbanr001/masters/sallm/results/downstream_standardized_20260913_v1/bases/xlstm \
      --adapter /scratch/lmbanr001/masters/sallm/retained_transfer_20260913/xlstm/general \
      --selection $b/selection/SELECTION.json --dtype float32 --merge-lora --tie-word-embeddings false \
      --batch-size 1 --max-batch-size 1 --output $R/general_afrimgsm/xlstm.json ;;
esac > $R/logs/$what.log 2>&1
rc=$?
echo "$([ $rc -eq 0 ] && echo DONE || echo FAIL) $what rc=$rc secs=$(( $(date +%s) - start )) $(date -Is)" >> $R/status.txt
