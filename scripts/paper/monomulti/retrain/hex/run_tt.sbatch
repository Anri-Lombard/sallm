#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=mz-tt
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

# News (natural first-token, v15 scorer), training template v5 then shared-prompt control
NEWSPP=/scratch/lmbanr001/masters/sallm/results/anomaly_audit_20260925/newskit/v15/src/main
for mode in train shared; do
  if [ $mode = train ]; then TT="--train-template $T/news_v5.yaml"; else TT=""; fi
  stamp news mono_eng $mode; env PYTHONPATH=$NEWSPP $PY $W/scripts/score_news_tt.py --arch mzansilm --base $BASE --adapter $H/mzansilm_news_mono_eng/final_adapter --langs eng $TT --out $W/out/news/mzansilm_news_mono_eng.$mode.json
  stamp news mono_xho $mode; env PYTHONPATH=$NEWSPP $PY $W/scripts/score_news_tt.py --arch mzansilm --base $BASE --adapter $H/mzansilm_news_mono_xho/final_adapter --langs xho $TT --out $W/out/news/mzansilm_news_mono_xho.$mode.json
  stamp news multi $mode; env PYTHONPATH=$NEWSPP $PY $W/scripts/score_news_tt.py --arch mzansilm --base $BASE --adapter $H/mzansilm_news_multi/final_adapter --langs eng,xho $TT --out $W/out/news/mzansilm_news_multi.$mode.json
done

# SIB (natural first-token, fp32), training templates v1 (primary) and v5, then shared-prompt control
SIBPP=$gen/src/main
for tag in v1 v5 shared; do
  if [ $tag = shared ]; then TT=""; else TT="--train-template $T/sib_$tag.yaml"; fi
  for l in afr eng nso sot xho zul; do
    stamp sib mono_$l $tag
    mkdir -p $W/out/sib/mono_$l
    env PYTHONPATH=$SIBPP $PY $W/scripts/score_sib_tt.py --arch mzansilm --base $BASE --split test --langs $l --tag $tag $TT --out-dir $W/out/sib/mono_$l --adapter $SM/$l/output/final_adapter
  done
  stamp sib multi $tag
  env PYTHONPATH=$SIBPP $PY $W/scripts/score_sib_tt.py --arch mzansilm --base $BASE --split test --tag $tag $TT --out-dir $W/out/sib/multi --adapter $H/mzansilm_sib_multi/final_adapter
done
date -Is; echo ALL_DONE
