#!/bin/bash

set -e

MODEL_PATH=$1
CONDA_ENV_NAME="sallm-ner"
TASK="masakhaner_ner_prompt_1"
BATCH_SIZE=8

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Path to the model checkpoint must be provided."
    echo "Usage: $0 /path/to/best_model.pt"
    exit 1
fi

echo "--- Running lm-evaluation-harness ---"
echo "Model: ${MODEL_PATH}"
echo "Task: ${TASK}"
echo "Batch Size: ${BATCH_SIZE}"

conda run -n "$CONDA_ENV_NAME" lm_eval \
    --model sallm_adapter \
    --model_args pretrained=${MODEL_PATH} \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --output_path "evaluation_results.json" \
    --log_samples

echo "--- Evaluation finished. Results saved to evaluation_results.json ---"
