#!/usr/bin/env python3
"""
Bayesian Hyperparameter Optimization for South African Language Model
Optimizes training hyperparameters across three different 125M parameter model architectures
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
    description="Bayesian Hyperparameter Optimization for South African Language Model"
)
parser.add_argument(
    "--n_trials", type=int, default=50, help="Number of optimization trials to run"
)
parser.add_argument(
    "--study_name",
    type=str,
    default=f"sallm_125m_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    help="Name of the optimization study",
)
parser.add_argument(
    "--storage",
    type=str,
    default="sqlite:///sallm_125m_optimization.db",
    help="Storage URL for Optuna study",
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
    "--continue_study",
    action="store_true",
    help="Continue from an existing study",
)
parser.add_argument(
    "--start_trial",
    type=int,
    default=0,
    help="Trial number to start from when continuing a study",
)
args = parser.parse_args()


# Define parameter search space including architecture selection
def create_param_space(trial):
    """Define the hyperparameter search space for Optuna"""

    # First, select one of the three architectural configurations (all ~125M params)
    arch_config = trial.suggest_categorical("arch_config", [1, 2, 3])

    # Fixed vocabulary size for all configurations
    vocab_size = 50257

    # Set architectural parameters based on selected configuration
    if arch_config == 1:
        # GPT-2 Style: ~124M parameters
        model_dim = 768
        num_heads = 12
        num_layers = 12
    elif arch_config == 2:
        # Wider & Shallower: ~125M parameters
        model_dim = 1024
        num_heads = 16
        num_layers = 8
    else:  # arch_config == 3
        # Narrower & Deeper: ~125M parameters
        model_dim = 512
        num_heads = 8
        num_layers = 24

    # Parameters to optimize - all training hyperparameters
    params = {
        # Model architecture (fixed per config)
        "model_dim": model_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        # Batch size and sequence length
        "batch_size": trial.suggest_categorical(
            "batch_size", [32768, 65536, 98304, 131072, 163840, 196608]
        ),
        "seq_len": trial.suggest_categorical("seq_len", [512, 1024, 2048]),
        # Learning rates for different parameter groups
        "head_lr": trial.suggest_float("head_lr", 0.001, 0.02, log=True),
        "embed_lr": trial.suggest_float("embed_lr", 0.1, 1.0, log=True),
        "scalar_lr": trial.suggest_float("scalar_lr", 0.01, 0.1, log=True),
        "hidden_lr": trial.suggest_float("hidden_lr", 0.005, 0.05, log=True),
        # Optimizer parameters
        "adam_beta1": trial.suggest_float("adam_beta1", 0.7, 0.95),
        "adam_beta2": trial.suggest_float("adam_beta2", 0.9, 0.999),
        "adam_eps": trial.suggest_float("adam_eps", 1e-10, 1e-8, log=True),
        "momentum_start": trial.suggest_float("momentum_start", 0.7, 0.9),
        "momentum_end": trial.suggest_float("momentum_end", 0.8, 0.99),
        # Learning rate schedule
        "cooldown_frac": trial.suggest_float("cooldown_frac", 0.1, 0.5),
        # Fixed parameters
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

        # Create a unique name for this trial
        arch_name = {768: "gpt2small", 1024: "wide", 512: "deep"}[params["model_dim"]]

        trial_name = f"trial_{trial.number}_arch{arch_name}_bs{params['batch_size']}_seq{params['seq_len']}"
        params["wandb_name"] = f"sallm_125m_{trial_name}"

        # Build command for running torchrun with these hyperparameters
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=4",
            "train_transformers.py",
        ]

        # Add all parameters as command line arguments
        for key, value in params.items():
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

        print(
            f"Starting trial {trial.number} with architecture {arch_name} and parameters:"
        )
        print(
            f"  Model: {params['num_layers']} layers, {params['num_heads']} heads, {params['model_dim']} dim"
        )
        print(
            f"  Batch size: {params['batch_size']}, Sequence length: {params['seq_len']}"
        )
        print(
            f"  Learning rates - head: {params['head_lr']:.6f}, embed: {params['embed_lr']:.6f}, scalar: {params['scalar_lr']:.6f}, hidden: {params['hidden_lr']:.6f}"
        )

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
            last_val_loss = float("inf")
            val_steps = 0

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
                                # Get step information
                                step = int(parts[0].split(":")[1].split("/")[0])

                                # Report to Optuna
                                trial.report(val_loss, step=step)
                                last_val_loss = val_loss
                                val_steps += 1

                                # Check for divergence - if loss doubles, stop early
                                if val_steps > 1 and val_loss > 2.0 * last_val_loss:
                                    print(
                                        f"Trial {trial.number} is diverging, stopping early."
                                    )
                                    process.terminate()
                                    raise optuna.exceptions.TrialPruned()
                    except Exception as e:
                        print(f"Error parsing validation metrics: {e}")

                # Check if we should prune this trial
                if trial.should_prune():
                    process.terminate()
                    raise optuna.exceptions.TrialPruned()

            # Wait for process to complete
            process.wait()

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
            trial.set_user_attr("arch_config", arch_name)
            trial.set_user_attr("val_perplexity", val_perplexity)
            trial.set_user_attr("val_accuracy", val_accuracy)
            trial.set_user_attr("duration_seconds", duration)
            trial.set_user_attr("model_dim", params["model_dim"])
            trial.set_user_attr("num_layers", params["num_layers"])
            trial.set_user_attr("num_heads", params["num_heads"])

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

            print(f"Trial {trial.number} completed in {duration:.2f} seconds")
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


def main():
    """Main function to run the Bayesian optimization"""
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"optuna_optimization_{args.study_name}",
        config={
            "optimization_type": "125M_model_hyperparameters",
            "n_trials": args.n_trials,
            "study_name": args.study_name,
            "epochs": args.epochs,
        },
    )

    # Initialize the TPE sampler with seed for reproducibility
    sampler = optuna.samplers.TPESampler(seed=42)

    # Create a new study with a sophisticated pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8,  # Allow more exploration before pruning
        n_warmup_steps=2,  # Allow 2 validation checks before pruning
        interval_steps=1,  # Check for pruning at each validation
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.continue_study,  # True if continuing, False if new
        direction="minimize",
        pruner=pruner,
        sampler=sampler,
    )

    # Callback to log results to WandB
    def wandb_callback(study, trial):
        trial_value = trial.value if trial.value is not None else float("inf")

        # Get architecture config information
        model_dim = trial.params.get("model_dim", 0)
        if model_dim == 768:
            arch_type = "GPT-2 Small (12 layers, 768 dim)"
        elif model_dim == 1024:
            arch_type = "Wide (8 layers, 1024 dim)"
        elif model_dim == 512:
            arch_type = "Deep (24 layers, 512 dim)"
        else:
            arch_type = "Unknown"

        # Log the current trial
        wandb.log(
            {
                "trial": trial.number,
                "best_value": study.best_value,
                "current_value": trial_value,
                "architecture": arch_type,
                **{f"params/{param}": value for param, value in trial.params.items()},
                **{
                    f"user_attr/{key}": value for key, value in trial.user_attrs.items()
                },
            }
        )

        # Update the best parameters
        if study.best_trial.number == trial.number:
            wandb.run.summary["best_trial"] = trial.number
            wandb.run.summary["best_value"] = study.best_value
            wandb.run.summary["best_architecture"] = arch_type

            for param, value in study.best_params.items():
                wandb.run.summary[f"best_params/{param}"] = value

    # Handle continuation of studies
    if args.continue_study:
        print(f"Continuing study {args.study_name} from trial {args.start_trial}")

        # Calculate completed trials and remaining trials
        completed_trials = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        print(f"Study has {completed_trials} completed trials")

        if args.start_trial > completed_trials:
            print(
                f"Warning: Requested to start from trial {args.start_trial} but only {completed_trials} trials completed"
            )
            print(f"Starting from trial {completed_trials}")
            args.start_trial = completed_trials

        n_trials_to_run = args.n_trials - args.start_trial

        if n_trials_to_run <= 0:
            print(
                f"No new trials to run. Requested trials: {args.n_trials}, Starting from: {args.start_trial}"
            )
            print("Exiting optimization.")
            return

        print(f"Running {n_trials_to_run} additional trials")
        print(
            f"Current best value: {study.best_value}"
            if completed_trials > 0
            else "No completed trials yet"
        )

        # Run optimization with adjusted trial count
        study.optimize(objective, n_trials=n_trials_to_run, callbacks=[wandb_callback])
    else:
        # Run optimization normally
        study.optimize(objective, n_trials=args.n_trials, callbacks=[wandb_callback])

    # Print the best parameters and results
    print("Study statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best validation loss: {study.best_value:.4f}")
    print("Best hyperparameters:")

    # Get architecture information from best trial
    best_model_dim = study.best_params.get("model_dim", 0)
    if best_model_dim == 768:
        arch_type = "GPT-2 Small (12 layers, 768 dim)"
    elif best_model_dim == 1024:
        arch_type = "Wide (8 layers, 1024 dim)"
    elif best_model_dim == 512:
        arch_type = "Deep (24 layers, 512 dim)"
    else:
        arch_type = "Unknown"

    print(f"  Architecture: {arch_type}")

    # Save the best parameters to a file
    best_params = study.best_params.copy()
    best_attrs = study.best_trial.user_attrs

    results = {
        "best_params": best_params,
        "architecture": arch_type,
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
            plot_slice,
        )

        # Generate and save plots
        fig1 = plot_optimization_history(study)
        fig1.write_html(f"optuna_results/{args.study_name}_optimization_history.html")

        fig2 = plot_param_importances(study)
        fig2.write_html(f"optuna_results/{args.study_name}_param_importances.html")

        # Plot the relationship between architecture and learning rates
        fig3 = plot_contour(study, params=["model_dim", "head_lr"])
        fig3.write_html(f"optuna_results/{args.study_name}_model_headlr_contour.html")

        fig4 = plot_contour(study, params=["batch_size", "seq_len"])
        fig4.write_html(f"optuna_results/{args.study_name}_batch_seq_contour.html")

        # Plot how learning rates vary by model architecture
        fig5 = plot_slice(
            study, params=["head_lr", "embed_lr", "scalar_lr", "hidden_lr"]
        )
        fig5.write_html(f"optuna_results/{args.study_name}_learning_rates_slice.html")

        # Plot parallel coordinates to visualize parameter relationships
        fig6 = plot_parallel_coordinate(
            study,
            params=[
                "model_dim",
                "batch_size",
                "seq_len",
                "head_lr",
                "embed_lr",
                "scalar_lr",
                "hidden_lr",
                "adam_beta1",
                "adam_beta2",
            ],
        )
        fig6.write_html(f"optuna_results/{args.study_name}_parallel_coordinates.html")

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
#SBATCH --time=72:00:00
#SBATCH --job-name="sallm_best_125m"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

module load python/miniconda3-py3.12

# Best parameters found by Bayesian optimization
TRAIN_FILES="{args.train_files}"
VAL_FILES="{args.val_files}"
BATCH_SIZE={best_params.get("batch_size")}
SEQ_LEN={best_params.get("seq_len")}
EPOCHS=30  # Use full 30 epochs for final training

# Model architecture parameters - {results["architecture"]}
VOCAB_SIZE=50257
NUM_LAYERS={best_params.get("num_layers")}
NUM_HEADS={best_params.get("num_heads")}
MODEL_DIM={best_params.get("model_dim")}

# Optimizer parameters (from Bayesian optimization)
HEAD_LR={best_params.get("head_lr")}
EMBED_LR={best_params.get("embed_lr")}
SCALAR_LR={best_params.get("scalar_lr")}
HIDDEN_LR={best_params.get("hidden_lr")}
MOMENTUM_START={best_params.get("momentum_start")}
MOMENTUM_END={best_params.get("momentum_end")}
ADAM_BETA1={best_params.get("adam_beta1")}
ADAM_BETA2={best_params.get("adam_beta2")}
ADAM_EPS={best_params.get("adam_eps")}
COOLDOWN_FRAC={best_params.get("cooldown_frac")}
VAL_LOSS_EVERY=2000

# Logging parameters
WANDB_PROJECT="{args.wandb_project}"
WANDB_ENTITY="{args.wandb_entity}"
WANDB_NAME="final_best_125m_model"

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
    with open("run_best_125m_model.sh", "w") as f:
        f.write(script)

    # Make the script executable
    os.chmod("run_best_125m_model.sh", 0o755)

    print("Best parameters script generated: run_best_125m_model.sh")


if __name__ == "__main__":
    main()
