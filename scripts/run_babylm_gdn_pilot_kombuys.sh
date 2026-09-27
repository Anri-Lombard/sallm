#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/alombard/babylm-gdn"
SEQ_LENGTH="${SEQ_LENGTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-6500}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-12}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-8}"
FULL_ATTENTION_EVERY="${FULL_ATTENTION_EVERY:-4}"
EMPTY_STATE_EMPHASIS_REPEAT="${EMPTY_STATE_EMPHASIS_REPEAT:-0}"
EMPTY_STATE_EMPHASIS_REGEX="${EMPTY_STATE_EMPHASIS_REGEX:-\\b(nothing|empty|emptied|none|nobody|no one|without)\\b}"
SYNTHETIC_ENTITY_EXAMPLES="${SYNTHETIC_ENTITY_EXAMPLES:-0}"
SYNTHETIC_ENTITY_EMPTY_RATE="${SYNTHETIC_ENTITY_EMPTY_RATE:-0.5}"
REVISION_NAME="${REVISION_NAME:-pilot}"
RUN_NAME="${RUN_NAME:-gdn-h${HIDDEN_SIZE}l${NUM_HIDDEN_LAYERS}-s${SEQ_LENGTH}-packed-strict-small-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="$ROOT/outputs/$RUN_NAME"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/$RUN_NAME.log"

mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$ROOT/cache/hf"
export HF_DATASETS_CACHE="$ROOT/cache/hf_datasets"
export TRANSFORMERS_CACHE="$ROOT/cache/transformers"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

{
  echo "RUN_NAME=$RUN_NAME"
  echo "OUT_DIR=$OUT_DIR"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "SEQ_LENGTH=$SEQ_LENGTH"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "MAX_STEPS=$MAX_STEPS"
  echo "HIDDEN_SIZE=$HIDDEN_SIZE"
  echo "NUM_HIDDEN_LAYERS=$NUM_HIDDEN_LAYERS"
  echo "EMPTY_STATE_EMPHASIS_REPEAT=$EMPTY_STATE_EMPHASIS_REPEAT"
  echo "SYNTHETIC_ENTITY_EXAMPLES=$SYNTHETIC_ENTITY_EXAMPLES"
  echo "SYNTHETIC_ENTITY_EMPTY_RATE=$SYNTHETIC_ENTITY_EMPTY_RATE"
  date

  "$ROOT/.venv/bin/python" "$ROOT/scripts/babylm_gdn_smoke_train.py" \
    --packed \
    --train-samples 0 \
    --seq-length "$SEQ_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --max-steps "$MAX_STEPS" \
    --learning-rate "$LEARNING_RATE" \
    --hidden-size "$HIDDEN_SIZE" \
    --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
    --num-attention-heads "$NUM_ATTENTION_HEADS" \
    --full-attention-every "$FULL_ATTENTION_EVERY" \
    --empty-state-emphasis-repeat "$EMPTY_STATE_EMPHASIS_REPEAT" \
    --empty-state-emphasis-regex "$EMPTY_STATE_EMPHASIS_REGEX" \
    --synthetic-entity-examples "$SYNTHETIC_ENTITY_EXAMPLES" \
    --synthetic-entity-empty-rate "$SYNTHETIC_ENTITY_EMPTY_RATE" \
    --output-dir "$OUT_DIR"

  cd "$ROOT/repos/babylm-eval/strict"
  env PATH="$ROOT/.venv/bin:$PATH" \
    bash scripts/eval_zero_shot_fast.sh "$OUT_DIR" "$REVISION_NAME" causal

  date
} 2>&1 | tee "$LOG_FILE"
