#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=mz-tt-ner
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/mzansilm_trainingtemplate_20260925/slurm/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/mzansilm_trainingtemplate_20260925/slurm/%x-%j.err
set -uo pipefail
W=/scratch/lmbanr001/masters/sallm/results/mzansilm_trainingtemplate_20260925
T=$W/templates
PY=/home/lmbanr001/masters/sallm/.venv/bin/python
BASE=/scratch/lmbanr001/masters/sallm/results/downstream_standardized_20260913_v1/bases/mzansilm
H=/scratch/lmbanr001/masters/sallm/results/historical_selected_adapter_recovery_20260917_hex_v3/train
SM=/scratch/lmbanr001/masters/sallm_recovery/mzansilm_sib_mono_20260916_v4r2/runs
R=/scratch/lmbanr001/masters/sallm/results/monomulti_rescore_20260924
export HF_HOME=/scratch/lmbanr001/hf-cache HF_DATASETS_CACHE=/scratch/lmbanr001/hf-cache/datasets HF_HUB_CACHE=/scratch/lmbanr001/hf-cache/hub
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export FLA_DISABLE_BACKEND_DISPATCH=1 SALLM_SKIP_MAMBA_KERNEL_CHECK=1 MAMBA_SCAN_IMPL=cuda
nvidia-smi --query-gpu=name --format=csv,noheader
stamp() { date -Is; echo "RUN $*"; }

# NER (full test, v5 training template), BOS-repair runner
bundle=/scratch/lmbanr001/masters/sallm_snapshots/general-sequence-official-test-20260916-v1
repair=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260917_v2/control
gen=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8
selection=/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection/SELECTION.json
release=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/release/NER_TEST_ACCESS_RELEASED_V1.json
for l in tsn xho zul; do
  stamp ner $l
  env PYTHONPATH="$gen/src/main:$bundle:$repair" NER_LANGS=$l NER_TEMPLATE_$l=$T/ner_v5.jinja \
    $PY $W/scripts/run_ner_template_check.py --task ner --phase test --architecture mzansilm \
    --checkpoint $BASE --adapter $H/mzansilm_ner_mono_$l/final_adapter --protocol $R/protocols/mzansilm_ner_mono_$l.json \
    --selection $selection --release $release --output $W/out/ner/mzansilm_ner_mono_$l.v5.json \
    && $PY -c "import json;d=json.load(open('$W/out/ner/mzansilm_ner_mono_$l.v5.json'));print('RESULT ner $l',d['elapsed_seconds'],d['reported_metrics'])"
done

date -Is; echo ALL_DONE
