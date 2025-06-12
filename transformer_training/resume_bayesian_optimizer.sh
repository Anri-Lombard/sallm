#!/bin/sh
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=8 --gres=gpu:l40s:4
#SBATCH --time=48:00:00
#SBATCH --job-name="sallm_bayesian_opt_cont"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# Load Python environment
module load python/miniconda3-py3.12

# Create a unique study name based on the current date and time
STUDY_NAME="transformer_opt_continued_$(date +%Y%m%d%H%M%S)"
N_TRIALS=30
EPOCHS=5

# Previous study information
PREVIOUS_STUDY_NAME="optimizer_params_20250313181737"                 # REPLACE WITH YOUR ACTUAL STUDY NAME
PREVIOUS_STORAGE="sqlite:///optuna_results/${PREVIOUS_STUDY_NAME}.db" # Path to your previous study's database

# Configure paths and other settings
TRAIN_FILES="/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_train_*.bin"
VAL_FILES="/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_val_*.bin"
WANDB_PROJECT="sallm-transformer-hyperparameter-tuning"
WANDB_ENTITY="anri-m-lombard"

# Create directories for logs and results
mkdir -p optuna_logs optuna_results

# Run the resumed Bayesian optimization
python resume_bayesian_optimizer.py \
	--n_trials $N_TRIALS \
	--previous_study_name $PREVIOUS_STUDY_NAME \
	--previous_storage "$PREVIOUS_STORAGE" \
	--study_name $STUDY_NAME \
	--storage "sqlite:///optuna_results/${STUDY_NAME}.db" \
	--wandb_project $WANDB_PROJECT \
	--wandb_entity $WANDB_ENTITY \
	--train_files "$TRAIN_FILES" \
	--val_files "$VAL_FILES" \
	--epochs $EPOCHS \
	--last_trial_to_keep 4 \
	--start_trial 5

echo "Bayesian optimization resumed and completed. Results are in optuna_results/${STUDY_NAME}_best_params.json"
echo "A script to run the model with the best parameters has been generated: run_best_model.sh"
