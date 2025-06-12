#!/bin/sh
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=8 --gres=gpu:l40s:4
#SBATCH --time=48:00:00
#SBATCH --job-name="sallm_transformer_tune"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL
# Load Python environment
module load python/miniconda3-py3.12

# Set environment variables for file paths and WandB configuration
TRAIN_FILES="/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_train_*.bin"
VAL_FILES="/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_val_*.bin"
WANDB_PROJECT="sallm-transformer-hyperparameter-tuning"
WANDB_ENTITY="anri-m-lombard"

# Default parameters for study
N_TRIALS=50
EPOCHS=5
MAX_CONCURRENT=1

# Parse command line arguments for continuation
CONTINUE_STUDY=false
EXISTING_STUDY=""
START_TRIAL=0

# Process command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
	--continue)
		CONTINUE_STUDY=true
		shift
		;;
	--study)
		EXISTING_STUDY="$2"
		shift 2
		;;
	--start-trial)
		START_TRIAL="$2"
		shift 2
		;;
	--trials)
		N_TRIALS="$2"
		shift 2
		;;
	--epochs)
		EPOCHS="$2"
		shift 2
		;;
	*)
		echo "Unknown option: $1"
		echo "Usage: sbatch run_optimizer.sh [--continue --study STUDY_NAME --start-trial N --trials N --epochs N]"
		exit 1
		;;
	esac
done

# Set study name based on whether we're continuing
if [ "$CONTINUE_STUDY" = true ]; then
	if [ -z "$EXISTING_STUDY" ]; then
		echo "Error: --continue specified but no --study provided"
		echo "Usage: sbatch run_optimizer.sh --continue --study STUDY_NAME [--start-trial N]"
		exit 1
	fi
	STUDY_NAME="$EXISTING_STUDY"
	echo "Continuing existing study: $STUDY_NAME from trial $START_TRIAL"
else
	# Create a new study with timestamp
	TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
	STUDY_NAME="sallm_125m_opt_${TIMESTAMP}"
	echo "Creating new study: $STUDY_NAME"
fi

# Log start time and configuration
echo "Starting Bayesian Optimization at $(date)"
echo "Study name: ${STUDY_NAME}"
echo "Number of trials: ${N_TRIALS}"
echo "Epochs per trial: ${EPOCHS}"
if [ "$CONTINUE_STUDY" = true ]; then
	echo "Continuing from trial: ${START_TRIAL}"
	echo "Database file: ${STUDY_NAME}.db"
fi

# Create directories for logs and results
mkdir -p optuna_logs
mkdir -p optuna_results

# Install required packages if not already installed
pip install optuna wandb plotly

# Run the Bayesian optimizer with continuation support if needed
if [ "$CONTINUE_STUDY" = true ]; then
	python bayesian_optimizer.py \
		--n_trials ${N_TRIALS} \
		--study_name ${STUDY_NAME} \
		--storage "sqlite:///${STUDY_NAME}.db" \
		--wandb_project ${WANDB_PROJECT} \
		--wandb_entity ${WANDB_ENTITY} \
		--train_files "${TRAIN_FILES}" \
		--val_files "${VAL_FILES}" \
		--epochs ${EPOCHS} \
		--max_concurrent ${MAX_CONCURRENT} \
		--continue_study \
		--start_trial ${START_TRIAL}
else
	python bayesian_optimizer.py \
		--n_trials ${N_TRIALS} \
		--study_name ${STUDY_NAME} \
		--storage "sqlite:///${STUDY_NAME}.db" \
		--wandb_project ${WANDB_PROJECT} \
		--wandb_entity ${WANDB_ENTITY} \
		--train_files "${TRAIN_FILES}" \
		--val_files "${VAL_FILES}" \
		--epochs ${EPOCHS} \
		--max_concurrent ${MAX_CONCURRENT}
fi

# Log completion
echo "Bayesian Optimization completed at $(date)"
echo "Results saved to optuna_results/${STUDY_NAME}_best_params.json"
echo "Training script generated: run_best_125m_model.sh"

# Print summary of best parameters
echo "Summary of best parameters:"
cat optuna_results/${STUDY_NAME}_best_params.json | grep -v '"language_metrics"' | grep -v '},'

# Make the generated training script executable
chmod +x run_best_125m_model.sh

# Optional: Display information about submitting the final training job
echo ""
echo "To submit the final training job with best parameters, use:"
echo "  sbatch run_best_125m_model.sh"

# Instructions for continuing this study if needed
echo ""
echo "If you need to continue this study later, use:"
echo "  sbatch run_optimizer.sh --continue --study ${STUDY_NAME} --start-trial <NUMBER>"
