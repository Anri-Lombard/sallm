#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:ampere:1
#SBATCH --time=08:00:00
#SBATCH --job-name="ner_finetuning"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

set -e

TASK_NAME=$1
MODE=$2
CONDA_ENV_NAME="sallm-ner"

if [ -z "$TASK_NAME" ] || [ -z "$MODE" ]; then
	echo "Error: Task name and mode must be specified."
	echo "Usage: $0 {task_name} {hpo|train|evaluate}"
	exit 1
fi

TASK_DIR="tasks/${TASK_NAME}"
if [ ! -d "$TASK_DIR" ]; then
	echo "Error: Task directory '${TASK_DIR}' not found."
	exit 1
fi

echo "Loading Python environment..."
module load python/miniconda3-py3.12

cd "$TASK_DIR"
echo "Changed directory to $(pwd)"

OUTPUT_DIR="./finetuning_results"
BEST_MODEL_PATH="${OUTPUT_DIR}/best_model.pt"

case "$MODE" in
hpo)
	echo "--- Starting Hyperparameter Optimization (HPO) Mode for task: ${TASK_NAME} ---"

	SWEEP_FILE="sweep.yaml"
	if [ ! -f "$SWEEP_FILE" ]; then
		echo "Error: sweep.yaml not found in ${TASK_DIR}"
		exit 1
	fi

	WANDB_PROJECT=$(grep 'project:' sweep.yaml | awk '{print $2}')
	if [ -z "$WANDB_PROJECT" ]; then
		echo "Error: Could not determine wandb project from sweep.yaml"
		exit 1
	fi

	HPO_RUN_COUNT=${3:-20}

	SWEEP_ID=$(conda run -n "$CONDA_ENV_NAME" wandb sweep --project "$WANDB_PROJECT" "$SWEEP_FILE" 2>&1 | grep 'wandb agent' | awk '{print $NF}')

	if [ -z "$SWEEP_ID" ]; then
		echo "Error: Failed to create W&B sweep or extract Sweep ID."
		exit 1
	fi

	echo "Sweep created successfully with ID: ${SWEEP_ID}"
	echo "Starting agent for ${HPO_RUN_COUNT} runs..."

	conda run -n "$CONDA_ENV_NAME" wandb agent "$SWEEP_ID" --count ${HPO_RUN_COUNT}

	echo "--- HPO sweep finished. ---"
	;;

train)
	echo "--- Starting Standard Training Mode for task: ${TASK_NAME} ---"

	echo "Starting final training run..."
	conda run -n "$CONDA_ENV_NAME" python run.py
	echo "--- Final training run finished. ---"

    if [ -f "$BEST_MODEL_PATH" ]; then
        echo "--- Automatically starting evaluation on the best model ---"
        cd ../..
        ./run_evaluation.sh "$BEST_MODEL_PATH"
    else
        echo "Warning: Best model not found at ${BEST_MODEL_PATH}. Skipping evaluation."
    fi
	;;

evaluate)
    echo "--- Starting Evaluation Mode ---"
    MODEL_TO_EVAL_PATH=${3:-$BEST_MODEL_PATH}

    if [ ! -f "$MODEL_TO_EVAL_PATH" ]; then
        echo "Error: Model file not found at '${MODEL_TO_EVAL_PATH}'"
        echo "Usage: $0 ${TASK_NAME} evaluate /path/to/your/model.pt"
        exit 1
    fi
    cd ../..
    ./run_evaluation.sh "$MODEL_TO_EVAL_PATH"
    ;;

*)
	echo "Error: Invalid mode '$MODE' specified. Usage: $0 {task_name} {hpo|train|evaluate}"
	exit 1
	;;
esac