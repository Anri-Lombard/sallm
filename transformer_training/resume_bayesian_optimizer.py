#!/usr/bin/env python3
"""
Resume a Bayesian Hyperparameter Optimization study for South African Language Model
Loads trials up to a specific number from an existing Optuna study and continues from there
Includes batch size optimization along with optimizer parameters
"""

import os
import sys
import json
import time
import optuna
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import wandb

# Create an argument parser
parser = argparse.ArgumentParser(
    description="Resume Bayesian Hyperparameter Optimization for Transformer Model"
)
parser.add_argument(
    "--n_trials", type=int, default=15, help="Number of new optimization trials to run"
)
parser.add_argument(
    "--previous_study_name",
    type=str,
    required=True,
    help="Name of the previous optimization study to continue from",
)
parser.add_argument(
    "--previous_storage",
    type=str,
    required=True,
    help="Storage URL for previous Optuna study (e.g., sqlite:///previous_study.db)",
)
parser.add_argument(
    "--study_name",
    type=str,
    default=f"batch_optimizer_tuning_cont_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    help="Name of the new optimization study",
)
parser.add_argument(
    "--storage",
    type=str,
    default="sqlite:///batch_optimizer_tuning_continued.db",
    help="Storage URL for new Optuna study",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="sallm-transformer-hyperparameter-tuning",
    help="Weights & Biases project name",
)
parser.add_argument(
    "--wandb_entity",
    type=str,
    default="anri-m-lombard",
    help="Weights & Biases entity name",
)
parser.add_argument(
    "--train_files",
    type=str,
    default="/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_train_*.bin",
    help="Pattern for training files",
)
parser.add_argument(
    "--val_files",
    type=str,
    default="/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_val_*.bin",
    help="Pattern for validation files",
)
parser.add_argument(
    "--epochs", type=int, default=5, help="Number of epochs for each trial"
)
parser.add_argument(
    "--max_concurrent",
    type=int,
    default=1,
    help="Maximum number of concurrent trials (limited by available GPUs)",
)
parser.add_argument(
    "--last_trial_to_keep",
    type=int,
    default=15,
    help="The last trial number to import from the previous study (inclusive)",
)
parser.add_argument(
    "--start_trial",
    type=int,
    default=15,
    help="The trial number to start from in the new study",
)
args = parser.parse_args()


# We're not using model architecture combinations in this script since
# we're focusing on optimizer parameters instead
# Keeping this empty to avoid confusion
VALID_CONFIGS = []


# Define parameter search space
def create_param_space(trial):
    """Define the hyperparameter search space for Optuna"""

    params = {
        # Fixed parameters (previously optimized)
        "model_dim": 1024,
        "num_heads": 16,
        "num_layers": 16,
        "embed_lr": 0.6445459064096828,
        "hidden_lr": 0.018533079377142877,
        "seq_len": 1024,
        "cooldown_frac": 0.3511943168510717,
        # Parameters to optimize now
        "batch_size": trial.suggest_categorical(
            "batch_size", [65536, 98304, 131072, 163840, 196608]
        ),
        "head_lr": trial.suggest_float("head_lr", 0.004, 0.016, log=True),
        "scalar_lr": trial.suggest_float("scalar_lr", 0.02, 0.08, log=True),
        "adam_beta1": trial.suggest_float("adam_beta1", 0.7, 0.9),
        "adam_beta2": trial.suggest_float("adam_beta2", 0.9, 0.99),
        "momentum_start": trial.suggest_float("momentum_start", 0.75, 0.9),
        "momentum_end": trial.suggest_float("momentum_end", 0.85, 0.99),
        # Fixed parameters (not optimized)
        "vocab_size": 50257,
        "adam_eps": 1e-10,
        "val_loss_every": 500,  # More frequent validation for hyperparameter tuning
        "epochs": args.epochs,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
    }
    return params


def objective(trial):
    """Objective function for Optuna optimization"""
    try:
        # Get hyperparameters for this trial
        params = create_param_space(trial)

        # Calculate the actual trial number
        # For new trials, we want them to start from args.start_trial and increment
        actual_trial_num = args.start_trial + (trial.number - len(filtered_trials))

        # Create a unique name for this trial - include batch size in the name
        trial_name = f"trial_{actual_trial_num}_batch{params['batch_size']}_headlr{params['head_lr']:.6f}_scalarlr{params['scalar_lr']:.6f}_beta1{params['adam_beta1']:.3f}"
        params["wandb_name"] = f"bayesian_opt_{trial_name}"

        # Build command for running torchrun with these hyperparameters
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=4",
            "train_transformers.py",
        ]

        # Add all parameters as command line arguments
        for key, value in params.items():
            if (
                key != "model_config_idx"
            ):  # Skip the model_config_idx parameter since it's not a real parameter
                cmd.append(f"--{key}")
                cmd.append(str(value))

        # Add train and val files
        cmd.append("--train_files")
        cmd.append(args.train_files)
        cmd.append("--val_files")
        cmd.append(args.val_files)

        # Create log directory for this trial
        log_dir = Path(f"optuna_logs/{trial_name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "output.log"

        # Log the hyperparameters for this trial
        with open(log_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)

        print(f"Starting trial {actual_trial_num} with parameters:")
        for k, v in params.items():
            if k != "model_config_idx":  # Skip the model_config_idx parameter in output
                print(f"  {k}: {v}")

        # Launch the training process and capture output
        start_time = time.time()

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Monitor the process output to extract metrics
            for line in process.stdout:
                f.write(line)
                f.flush()
                sys.stdout.write(line)
                sys.stdout.flush()

                # Look for validation metrics to report to Optuna
                if "overall_val_loss:" in line:
                    try:
                        # Extract overall validation loss
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.startswith("overall_val_loss:"):
                                val_loss = float(part.split(":")[1])
                                # Report intermediate value to Optuna
                                step = int(parts[0].split(":")[1].split("/")[0])
                                trial.report(val_loss, step=step)
                    except Exception as e:
                        print(f"Error parsing validation metrics: {e}")

                # Check if we should prune this trial
                if trial.should_prune():
                    process.terminate()
                    raise optuna.exceptions.TrialPruned()

        # Calculate total running time
        duration = time.time() - start_time

        # Extract final validation loss from WandB API (requires wandb to be properly set up)
        try:
            api = wandb.Api()
            run = api.run(
                f"{args.wandb_entity}/{args.wandb_project}/{params['wandb_name']}"
            )

            # Get the final validation metrics
            summary = run.summary
            val_loss = summary.get("val/overall_loss", float("inf"))
            val_perplexity = summary.get("val/overall_perplexity", float("inf"))
            val_accuracy = summary.get("val/overall_accuracy", 0.0)

            # Log the metrics back to the trial
            with open(log_dir / "final_metrics.json", "w") as f:
                json.dump(
                    {
                        "val_loss": val_loss,
                        "val_perplexity": val_perplexity,
                        "val_accuracy": val_accuracy,
                        "duration_seconds": duration,
                    },
                    f,
                    indent=2,
                )

            # Store additional metrics in the Optuna trial
            trial.set_user_attr("val_perplexity", val_perplexity)
            trial.set_user_attr("val_accuracy", val_accuracy)
            trial.set_user_attr("duration_seconds", duration)

            # Collect per-language metrics
            language_metrics = {}
            for key in summary.keys():
                if (
                    key.startswith("val/")
                    and "/loss" in key
                    and key != "val/overall_loss"
                ):
                    lang = key.split("/")[1]
                    language_metrics[lang] = summary[key]
                    trial.set_user_attr(f"val_loss_{lang}", summary[key])

            with open(log_dir / "language_metrics.json", "w") as f:
                json.dump(language_metrics, f, indent=2)

            print(f"Trial {actual_trial_num} completed in {duration:.2f} seconds")
            print(f"  Overall validation loss: {val_loss:.4f}")
            print(f"  Overall perplexity: {val_perplexity:.4f}")
            print(f"  Overall accuracy: {val_accuracy:.4f}")

            return val_loss

        except Exception as e:
            print(f"Error retrieving metrics from WandB: {e}")
            # If WandB fails, try to parse the log file for the last reported validation loss
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    val_losses = []
                    for line in lines:
                        if "overall_val_loss:" in line:
                            parts = line.split()
                            for part in parts:
                                if part.startswith("overall_val_loss:"):
                                    val_losses.append(float(part.split(":")[1]))

                    if val_losses:
                        return val_losses[
                            -1
                        ]  # Return the last reported validation loss
                    else:
                        return float("inf")  # If no validation loss was found
            except:
                return float("inf")  # If all else fails

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float("inf")  # Return a high loss for failed trials


# We don't need this function anymore since we're not using model_config_idx
# Keeping an empty function for compatibility with existing code calls
def map_model_config_to_idx(config):
    """This function is not used in the optimizer parameter version"""
    return 0


def main():
    """Main function to run the continued Bayesian optimization"""
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"batch_optuna_optimization_continued_{args.study_name}",
        config={
            "optimization_type": "bayesian_batch_optimizer_continued",
            "n_trials": args.n_trials,
            "previous_study_name": args.previous_study_name,
            "study_name": args.study_name,
            "epochs": args.epochs,
            "last_trial_to_keep": args.last_trial_to_keep,
        },
    )

    print(f"Loading previous study: {args.previous_study_name}")
    # Load the previous study to extract trial information
    previous_study = optuna.load_study(
        study_name=args.previous_study_name, storage=args.previous_storage
    )

    print(f"Found {len(previous_study.trials)} trials in previous study")

    # Filter completed trials and keep only up to last_trial_to_keep
    global filtered_trials  # Make this global so objective function can access it
    filtered_trials = []
    for trial in previous_study.trials:
        if (
            trial.number <= args.last_trial_to_keep
            and trial.state == optuna.trial.TrialState.COMPLETE
        ):
            filtered_trials.append(trial)

    print(
        f"Keeping {len(filtered_trials)} successfully completed trials (up to trial {args.last_trial_to_keep})"
    )

    # Initialize the TPE sampler with seed for reproducibility
    sampler = optuna.samplers.TPESampler(seed=42)

    # Create a new study
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=0, interval_steps=1
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=False,  # Create a new study
        direction="minimize",
        pruner=pruner,
        sampler=sampler,
    )

    print("Transferring selected trials to the new study...")
    # Transfer trials from the previous study to the new one
    for i, prev_trial in enumerate(filtered_trials):
        try:
            # Create a new trial with optimizer parameters including batch size
            # Use a default batch size of 131072 if it doesn't exist in the previous trial
            new_trial = optuna.trial.create_trial(
                params={
                    "batch_size": prev_trial.params.get("batch_size", 131072),
                    "head_lr": prev_trial.params.get("head_lr", 0.008),
                    "scalar_lr": prev_trial.params.get("scalar_lr", 0.04),
                    "adam_beta1": prev_trial.params.get("adam_beta1", 0.8),
                    "adam_beta2": prev_trial.params.get("adam_beta2", 0.95),
                    "momentum_start": prev_trial.params.get("momentum_start", 0.85),
                    "momentum_end": prev_trial.params.get("momentum_end", 0.95),
                },
                distributions={
                    "batch_size": optuna.distributions.CategoricalDistribution(
                        [65536, 98304, 131072, 163840, 196608]
                    ),
                    "head_lr": optuna.distributions.LogUniformDistribution(
                        0.004, 0.016
                    ),
                    "scalar_lr": optuna.distributions.LogUniformDistribution(
                        0.02, 0.08
                    ),
                    "adam_beta1": optuna.distributions.UniformDistribution(0.7, 0.9),
                    "adam_beta2": optuna.distributions.UniformDistribution(0.9, 0.99),
                    "momentum_start": optuna.distributions.UniformDistribution(
                        0.75, 0.9
                    ),
                    "momentum_end": optuna.distributions.UniformDistribution(
                        0.85, 0.99
                    ),
                },
                value=prev_trial.value,
                state=optuna.trial.TrialState.COMPLETE,
            )
            study.add_trial(new_trial)
            print(f"  Transferred trial {prev_trial.number} (original) -> {i} (new)")
        except Exception as e:
            print(f"  Error transferring trial {prev_trial.number}: {e}")

    print(f"Successfully transferred {len(filtered_trials)} trials to the new study")
    print(f"Best value so far: {study.best_value}")
    print("Best parameters so far:")

    # Display best parameters (no conversion needed)
    best_params = study.best_params

    for param, value in best_params.items():
        print(f"  {param}: {value}")

    print(f"\nStarting new trials from trial {args.start_trial}...")

    # Callback to log results to WandB
    def wandb_callback(study, trial):
        # Skip logging for transferred trials
        if trial.number < len(filtered_trials):
            return

        trial_value = trial.value if trial.value is not None else float("inf")
        # Calculate the absolute trial number
        absolute_trial_num = args.start_trial + (trial.number - len(filtered_trials))

        # No need to convert parameters, just use them directly
        params = trial.params.copy()

        # Log the current trial
        wandb.log(
            {
                "trial": absolute_trial_num,
                "study_trial_number": trial.number,
                "best_value": study.best_value,
                "current_value": trial_value,
                **{f"params/{param}": value for param, value in params.items()},
                **{
                    f"user_attr/{key}": value for key, value in trial.user_attrs.items()
                },
            }
        )

        # Update the best parameters
        if study.best_trial.number == trial.number:
            wandb.run.summary["best_trial"] = absolute_trial_num
            wandb.run.summary["best_value"] = study.best_value

            # No conversion needed for optimizer parameters
            best_params = study.best_params.copy()

            for param, value in best_params.items():
                wandb.run.summary[f"best_params/{param}"] = value

    # Determine how many new trials to run (accounting for transferred trials)
    new_trials_to_run = args.n_trials

    # Run the optimization - continue with additional trials
    study.optimize(objective, n_trials=new_trials_to_run, callbacks=[wandb_callback])

    # Print the best parameters and results
    print("Study statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best validation loss: {study.best_value:.4f}")
    print("Best hyperparameters:")

    # Save the best parameters to a file
    best_params = study.best_params.copy()

    # No conversion needed for optimizer parameters

    best_attrs = study.best_trial.user_attrs
    results = {
        "best_params": best_params,
        "metrics": {
            "val_loss": study.best_value,
            "val_perplexity": best_attrs.get("val_perplexity", "N/A"),
            "val_accuracy": best_attrs.get("val_accuracy", "N/A"),
            "duration_seconds": best_attrs.get("duration_seconds", "N/A"),
        },
        "language_metrics": {
            key.replace("val_loss_", ""): value
            for key, value in best_attrs.items()
            if key.startswith("val_loss_")
        },
    }

    # Create the output directory if it doesn't exist
    os.makedirs("optuna_results", exist_ok=True)

    # Save the best parameters to a file
    with open(f"optuna_results/{args.study_name}_best_params.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print the best parameters
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Visualize the optimization results (requires plotly)
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_contour,
            plot_parallel_coordinate,
        )

        # Generate and save plots
        fig1 = plot_optimization_history(study)
        fig1.write_html(f"optuna_results/{args.study_name}_optimization_history.html")

        fig2 = plot_param_importances(study)
        fig2.write_html(f"optuna_results/{args.study_name}_param_importances.html")

        # Plot the relationship between batch size and learning rates
        fig3 = plot_contour(study, params=["batch_size", "head_lr"])
        fig3.write_html(f"optuna_results/{args.study_name}_batch_headlr_contour.html")

        fig4 = plot_contour(study, params=["batch_size", "scalar_lr"])
        fig4.write_html(f"optuna_results/{args.study_name}_batch_scalarlr_contour.html")

        # Plot parallel coordinates to visualize parameter relationships
        fig5 = plot_parallel_coordinate(
            study,
            params=["batch_size", "head_lr", "scalar_lr", "adam_beta1", "adam_beta2"],
        )
        fig5.write_html(f"optuna_results/{args.study_name}_parallel_coordinates.html")

        print(f"Visualizations saved to optuna_results/{args.study_name}_*.html")
    except Exception as e:
        print(f"Could not generate visualization plots: {e}. Please install plotly.")

    # Close WandB
    wandb.finish()

    # Generate a script with the best parameters for final training
    generate_best_params_script(results)


def generate_best_params_script(results):
    """Generate a script to run the model with the best parameters"""
    best_params = results["best_params"]
    script = f"""#!/bin/sh
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=4 --gres=gpu:l40s:4
#SBATCH --time=48:00:00
#SBATCH --job-name="sallm_best_transformer"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

module load python/miniconda3-py3.12

# Best parameters found by Bayesian optimization
TRAIN_FILES="{args.train_files}"
VAL_FILES="{args.val_files}"
BATCH_SIZE={best_params.get("batch_size", 131072)}
SEQ_LEN=1024
EPOCHS=30  # Use full 30 epochs for final training

# Model architecture parameters (fixed)
VOCAB_SIZE=50257
NUM_LAYERS=16
NUM_HEADS=16
MODEL_DIM=1024

# Optimizer parameters (from Bayesian optimization)
HEAD_LR={best_params.get("head_lr")}
EMBED_LR=0.6445459064096828
SCALAR_LR={best_params.get("scalar_lr")}
HIDDEN_LR=0.018533079377142877
MOMENTUM_START={best_params.get("momentum_start")}
MOMENTUM_END={best_params.get("momentum_end")}
ADAM_BETA1={best_params.get("adam_beta1")}
ADAM_BETA2={best_params.get("adam_beta2")}
ADAM_EPS=1e-10
COOLDOWN_FRAC={best_params.get("cooldown_frac", 0.3511943168510717)}
VAL_LOSS_EVERY=2000

# Logging parameters
WANDB_PROJECT="{args.wandb_project}"
WANDB_ENTITY="{args.wandb_entity}"
WANDB_NAME="final_best_batch_optimizer_model"

# Run the training script with the best parameters
torchrun --standalone --nproc_per_node=4 train_transformers.py \\
    --train_files "$TRAIN_FILES" \\
    --val_files "$VAL_FILES" \\
    --batch_size $BATCH_SIZE \\
    --seq_len $SEQ_LEN \\
    --epochs $EPOCHS \\
    --vocab_size $VOCAB_SIZE \\
    --num_layers $NUM_LAYERS \\
    --num_heads $NUM_HEADS \\
    --model_dim $MODEL_DIM \\
    --head_lr $HEAD_LR \\
    --embed_lr $EMBED_LR \\
    --scalar_lr $SCALAR_LR \\
    --hidden_lr $HIDDEN_LR \\
    --momentum_start $MOMENTUM_START \\
    --momentum_end $MOMENTUM_END \\
    --adam_beta1 $ADAM_BETA1 \\
    --adam_beta2 $ADAM_BETA2 \\
    --adam_eps $ADAM_EPS \\
    --cooldown_frac $COOLDOWN_FRAC \\
    --val_loss_every $VAL_LOSS_EVERY \\
    --wandb_project "$WANDB_PROJECT" \\
    --wandb_entity "$WANDB_ENTITY" \\
    --wandb_name "$WANDB_NAME" \\
    --save_checkpoint
"""

    # Save the script
    with open("run_best_batch_optimizer_model.sh", "w") as f:
        f.write(script)

    # Make the script executable
    os.chmod("run_best_batch_optimizer_model.sh", 0o755)

    print("Best parameters script generated: run_best_batch_optimizer_model.sh")


if __name__ == "__main__":
    main()
